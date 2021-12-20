import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torchvision import models
from util import make_warped_stack

class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                                norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        output = self.model(input)
        return output

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)
        if outermost:
            # import pdb; pdb.set_trace()
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            # up = [uprelu, upsample, upconv, upnorm]
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

# UNet with residual blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            # hard to converge with out batch or instance norm
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class VGGLoss(nn.Module):
    def __init__(self, model=None):
        super(VGGLoss, self).__init__()
        if model is None:
            self.vgg = Vgg19()
        else:
            self.vgg = model

        self.vgg.cuda()
        self.vgg.eval()
        self.criterion = nn.L1Loss()
        self.style_criterion = StyleLoss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.style_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        # self.weights = [5.0, 1.0, 0.5, 0.4, 0.8]
        # self.style_weights = [10e4, 1000, 50, 15, 50]

    def forward(self, x, y, style=False):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if style:
            # return both perceptual loss and style loss.
            style_loss = 0
            for i in range(len(x_vgg)):
                this_loss = (self.weights[i] *
                             self.criterion(x_vgg[i], y_vgg[i].detach()))
                this_style_loss = (self.style_weights[i] *
                                   self.style_criterion(x_vgg[i], y_vgg[i].detach()))
                loss += this_loss
                style_loss += this_style_loss
            return loss, style_loss

        for i in range(len(x_vgg)):
            this_loss = (self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()))
            loss += this_loss
        return loss

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, x, y):
        Gx = gram_matrix(x)
        Gy = gram_matrix(y)
        return F.mse_loss(Gx, Gy) * 600000


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        #in 6*640*640
        self.resDown1 = ResBlockDown(3, 64) #out 64*320*320
        self.resDown2 = ResBlockDown(64, 128) #out 128*160*160
        self.resDown3 = ResBlockDown(128, 256) #out 256*80*80
        self.resDown4 = ResBlockDown(256, 256) #out 256*40*40
        self.self_att = SelfAttention(256) #out 256*40*40
        self.resDown5 = ResBlockDown(256, 512) #out 512*20*20
        self.resDown6 = ResBlockDown(512, 512) #out 512*10*10
        self.resDown7 = ResBlockDown(512, 512) #out 512*5*5
        self.sum_pooling = nn.AdaptiveAvgPool2d((1,1)) #out 512*1*1
        self.fc = nn.Linear(512, 1)
        
        
    def forward(self, y):
        # out = torch.cat((x,y), dim=-3)
        out = y
        out = self.resDown1(out) 
        out = self.resDown2(out)       
        out = self.resDown3(out)
        out = self.resDown4(out)
        out = self.self_att(out)
        out = self.resDown5(out)
        out = self.resDown6(out)
        out = self.resDown7(out)

        out = self.sum_pooling(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(ResBlockDown, self).__init__()
        
        self.relu = nn.ReLU(inplace = False)
        self.relu_inplace = nn.ReLU(inplace = True)
        self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1,))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))

    def forward(self, x):
        res = x
        
        #left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)
        
        #right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)
        
        #merge
        out = out_res + out
        
        return out
    
class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()
        
        #conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        
        self.softmax = nn.Softmax(-2) #sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x) #BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x) #BxC'xHxW
        h_projection = self.conv_h(x) #BxCxHxW
        
        f_projection = torch.transpose(f_projection.view(B,-1,H*W), 1, 2) #BxNxC', N=H*W
        g_projection = g_projection.view(B,-1,H*W) #BxC'xN
        h_projection = h_projection.view(B,-1,H*W) #BxCxN
        
        attention_map = torch.bmm(f_projection, g_projection) #BxNxN
        attention_map = self.softmax(attention_map) #sum_i_N (A i,j) = 1
        
        #sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map) #BxCxN
        out = out.view(B,C,H,W)
        
        out = self.gamma*out + x
        return out
    
class WarpModel(nn.Module):
    def __init__(self, n_joints, n_limbs):
        super(WarpModel, self).__init__()
        self.unet_stageA = ResUnetGenerator(n_joints+3, n_limbs+1, 3) 
        self.unet_stageC = ResUnetGenerator(3*n_limbs+n_joints, 4, 3) #output mask+ target foreground
        self.unet_stageD = ResUnetGenerator(4+n_joints, 3, 3) # output background

    # src_in : B 3 H W
    # pose_src  : B n_joints H W
    # pose_target : B n_joints H W
    # src_mask_prior : B n_limbs+1 H W
    # trans_in : B 2 3 n_limbs+1 
    def forward(self, src_in, pose_src, pose_target, src_mask_prior, trans_in):
        stageA_in = torch.cat([src_in, pose_src], dim = 1)
        src_mask_delta = self.unet_stageA(stageA_in)
        src_mask = src_mask_prior + src_mask_delta
        src_mask = torch.softmax(src_mask, dim =1)

        warped_stack = make_warped_stack(src_mask, src_in, trans_in)
        fg_stack = warped_stack[:,3:,:,:]
        bg_src = warped_stack[:,0:3,:,:]
        bg_src_mask = src_mask[:,0,:,:].unsqueeze(1)

        #stage C fg
        stageC_in = torch.cat([fg_stack,pose_target], dim = 1)
        stageC_out = self.unet_stageC(stageC_in)

        fg_tgt = stageC_out[:,0:3,:,:]
        fg_tgt = torch.tanh(fg_tgt)
        fg_mask = stageC_out[:,-1,:,:].unsqueeze(1)
        fg_mask = torch.sigmoid(fg_mask).repeat(1,3,1,1)
        bg_mask = 1-fg_mask
        
        #stage D bg
        stageD_in = torch.cat([bg_src, bg_src_mask, pose_src],dim = 1)
        bg_tgt = self.unet_stageD(stageD_in)
        bg_tgt = torch.tanh(bg_tgt)

        # Merge bg and fg
        y = fg_tgt * fg_mask + bg_tgt * bg_mask
        return y


if __name__=="__main__":
    # src_in : B 3 H W
    # pose_src  : B n_joints H W
    # pose_target : B n_joints H W
    # src_mask_prior : B n_limbs+1 H W
    # trans_in : B 2 3 n_limbs+1 
    src_in = torch.randn(1,3,640,640).cuda()
    pose_src = torch.randn(1,13,640,640).cuda()
    pose_target = torch.randn(1,13,640,640).cuda()
    src_mask_prior = torch.randn(1,9,640,640).cuda()
    trans_in = torch.randn(1,2,3,8).cuda()
    model = WarpModel(n_joints = 13, n_limbs = 8).cuda()
    out = model(src_in, pose_src, pose_target, src_mask_prior, trans_in)
    print(out.shape)






    