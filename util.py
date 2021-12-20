import torch
import pdb
import torch.nn.functional as F
import numpy as np
import transformations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_warped_stack(mask, src_in, trans_in):
    for i in range(mask.shape[1]):
        mask_i = mask[:, i,:, :].unsqueeze(1).repeat(1,3,1,1)
        src_masked = mask_i * src_in
        if i == 0:
            warps = src_masked
        else:
            warp_i = affine_warp(src_masked, trans_in[:, :, :, i-1])
            warps = torch.cat([warps, warp_i], 1)
    return warps

def affine_warp(im, theta):
    num_batch = im.shape[0]
    height = im.shape[2]
    width = im.shape[3]

    y_t, x_t = torch.meshgrid(torch.arange(0,height), torch.arange(0,width))
    y_t = y_t.float()
    x_t = x_t.float()
    x_t_flat = x_t.reshape(1,-1)
    y_t_flat = y_t.reshape(1,-1)
    ones = torch.ones_like(x_t_flat)
    
    # print(x_t_flat.device,y_t_flat.device,ones.device)
    grid = torch.cat((x_t_flat, y_t_flat, ones),dim = 0).to(device)
    grid = grid.unsqueeze(0)
    grid = grid.repeat(num_batch,1,1)

    T_g = torch.matmul(theta, grid)
    x_s = T_g[:,0,:].reshape(num_batch, height, width)
    y_s = T_g[:,1,:].reshape(num_batch, height, width)
    return interpolate(im, x_s, y_s)

def repeat(x,n_repeats):
    rep = torch.ones(1,n_repeats).float()
    x = torch.matmul(x.reshape(-1,1).float(),rep)    
    return x.flatten()

def interpolate(im,x,y):
    im = F.pad(im,[1,1,1,1],"reflect")  
    num_batch = im.shape[0]
    height = im.shape[2]
    width = im.shape[3]
    channels = im.shape[1]

    out_height = x.shape[1]
    out_width = x.shape[2]
                
    x = x.flatten()
    y = x.flatten()        
    x = x+1
    y = y+1
                
    max_x = width - 1
    max_y = height - 1
        
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1
                
    x0 = x0.clamp(0, max_x)
    x1 = x1.clamp(0, max_x)
    y0 = y0.clamp(0, max_y)
    y1 = y1.clamp(0, max_y)
      
    base = repeat(torch.arange(num_batch)*width*height, out_height*out_width).to(device)

    base_y0 = base + y0*width
    base_y1 = base + y1*width

    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1
                
    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = im.permute(0,2,3,1).reshape(-1,channels)
    # im_flat = tf.cast(im_flat, 'float32')

    Ia = im_flat[idx_a.long()]
    Ib = im_flat[idx_b.long()]
    Ic = im_flat[idx_c.long()]
    Id = im_flat[idx_d.long()]

    
    # and finally calculate interpolated values
    x1_f = x1.float()
    y1_f = y1.float()
    
    dx = x1_f - x
    dy = y1_f - y
                
    wa = torch.unsqueeze(dx * dy, 1)
    wb = torch.unsqueeze(dx * (1-dy), 1)
    wc = torch.unsqueeze((1-dx) * dy, 1)
    wd = torch.unsqueeze((1-dx) * (1-dy), 1)
    
         
    output = wa*Ia + wb*Ib + wc*Ic + wd*Id

    output = output.reshape(-1,out_height,out_width,channels).permute(0,3,1,2)
    return output

def fast_gaussian_torch(img_width, img_height, center, var_x, var_y, L=49):
    tmp = torch.zeros((img_height,img_width)).cuda()
    
    if var_x<10 and var_y<10:
        L=11

    patch = make_gaussian_map_torch(L, L, (L//2, L//2), var_x, var_y, 0)
    xx = int(center[0])
    yy = int(center[1])
    start_H = yy-patch.shape[0]//2
    end_H = clamp_min_max(yy-patch.shape[0]//2+patch.shape[0], min=0, max=img_height-1)
    start_W = xx-patch.shape[1]//2
    end_W = clamp_min_max(xx-patch.shape[1]//2+patch.shape[1], min=0, max=img_width-1)
    flag = 4
    if start_H <0 and start_W<0: 
        flag = 0
    if start_H <0 and start_W>0: 
        flag = 1
    if start_H >0 and start_W<0: 
        flag = 2
    if start_H >0 and start_W>0: 
        flag = 3
    
    start_H = clamp_min_max(start_H, min=0, max=img_height) 
    start_W = clamp_min_max(xx-patch.shape[1]//2, min=0, max=img_width) 
    patch = crop_patch(patch, end_H-start_H, end_W-start_W, flag)
    if tmp[start_H: end_H , start_W: end_W].shape == patch.shape:
        tmp[start_H: end_H , start_W: end_W] = patch

    return tmp

def make_gaussian_map(img_width, img_height, center, var_x, var_y, theta):
    xv, yv = np.meshgrid(np.array(range(img_width)), np.array(range(img_height)),
                         sparse=False, indexing='xy')
    a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
    b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
    c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)

    return np.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                    2 * b * (xv - center[0]) * (yv - center[1]) +
                    c * (yv - center[1]) * (yv - center[1])))

def make_gaussian_map_torch(img_width, img_height, center, var_x, var_y, theta):
    yv, xv = torch.meshgrid(torch.arange(0,img_height), torch.arange(0,img_width))

    yv = yv.cuda()
    xv = xv.cuda()
    a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
    b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
    c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)
    return torch.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                    2 * b * (xv - center[0]) * (yv - center[1]) +
                    c * (yv - center[1]) * (yv - center[1])))


def make_limb_masks(limbs, joints, img_width, img_height, sigma_perp_root ):
    n_limbs = len(limbs)
    mask = torch.zeros((img_height, img_width, n_limbs)).cuda()

    # Gaussian sigma perpendicular to the limb axis.
    sigma_perp = np.array(sigma_perp_root) ** 2

    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            p[j, :] = [joints[limbs[i][j], 0], joints[limbs[i][j], 1]]

        if n_joints_for_limb == 4:
            p_top = np.mean(p[0:2, :], axis=0)
            p_bot = np.mean(p[2:4, :], axis=0)
            p = np.vstack((p_top, p_bot))

        center = np.mean(p, axis=0)
        sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 1.5])
        theta = np.arctan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])
        mask_i = make_gaussian_map_torch(img_width, img_height, center, sigma_parallel, sigma_perp[i], theta)
        mask[:, :, i] = mask_i / (torch.max(mask_i) + 1e-6)
    return mask.permute(2,0,1)

def make_cluster_kp(cluster, joints, img_width, img_height, var_root):
    n_cluster = len(cluster)
    pose = torch.zeros((img_height, img_width, n_cluster)).cuda()
    
    var = np.array(var_root)**2
    for i in range(n_cluster):
        n_joints_for_cluster = len(cluster[i])
        kp_canvas = torch.zeros((img_height,img_width)).cuda()
        for j in range(n_joints_for_cluster):
            tmp =fast_gaussian_torch(img_width,img_height,joints[cluster[i][j]],var[i],var[i])
            kp_canvas += tmp
        pose[:,:,i] = kp_canvas
    return pose.permute(2,0,1)

def get_limb_transforms(limbs, joints1, joints2):
    n_limbs = len(limbs)

    Ms = np.zeros((2, 3, n_limbs))
    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p1 = np.zeros((n_joints_for_limb, 2))
        p2 = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            p1[j, :] = [joints1[limbs[i][j], 0], joints1[limbs[i][j], 1]]
            p2[j, :] = [joints2[limbs[i][j], 0], joints2[limbs[i][j], 1]]

        tform = transformations.make_similarity(p2, p1, False)
        Ms[:, :, i] = np.array([[tform[1], -tform[3], tform[0]], [tform[3], tform[1], tform[2]]])
    return Ms

def batchify_mask_prior(joints, img_width, img_height, sigma_perp_root = [35, 25, 25, 20, 25, 20, 10, 10]):
    limbs = [[0,8,9],[1,2,5],[2,3],[3,4],[5,6],[6,7],range(101,122),range(80,101)]
    
    batch_size = joints.shape[0]
    result = []
    for i in range(batch_size):
        limb_masks = make_limb_masks(limbs,joints[i],img_width,img_height, sigma_perp_root)
        bg_mask = (1.0 - torch.max(limb_masks, dim=0)[0]).unsqueeze(0)
        mask_prior = torch.log(torch.cat((bg_mask, limb_masks), dim=0) + 1e-10)
        result.append(mask_prior)
    
    return torch.stack(result, dim=0)


def batchify_cluster_kp(joints, img_width, img_height, var_root = [5,5,5,5,5,5,5,2,1,1,1,1,1]):
    cluster = [[1],[2],[3],[4],[5],[6],[7],range(10,27),[8,9,78,79]+list(range(28,36))+list(range(47,57)),range(37,46),range(58,78),range(101,122),range(80,101)]
    batch_size = joints.shape[0]
    result = []
    for i in range(batch_size):
        result.append(make_cluster_kp(cluster,joints[i],img_width,img_height, var_root))
    
    return torch.stack(result, dim=0)

def clamp_min_max(x, min=0, max=99999):
    if x<min:
        return min
    elif x>max:
        return max
    return x
    
def crop_patch(patch, valid_H, valid_W, type):
    H, W = patch.shape
    if type == 0:#左上
        return patch[H-valid_H:, W-valid_W:]
    if type == 1:#右上
        return patch[H-valid_H:, :valid_W]
    if type == 2:#左下
        return patch[:valid_H, W-valid_W:]
    if type == 3:#右下
        return patch[:valid_H, :valid_W]
    return patch

def pose137_to_pose122(x):
    return np.concatenate([x[:2, 0:8],  # upper_body
								x[:2, 15:17],   # eyes
								x[:2, 25:]], axis=1)   # face, hand_l and hand_r


def scale_resize(curshape, myshape=(1080, 1920, 3), mean_height=0.0):

	if curshape == myshape:
		return None

	x_mult = myshape[0] / float(curshape[0])
	y_mult = myshape[1] / float(curshape[1])

	if x_mult == y_mult:
		# just need to scale
		return x_mult, (0.0, 0.0)
	elif y_mult > x_mult:
		### scale x and center y
		y_new = x_mult * float(curshape[1])
		translate_y = (myshape[1] - y_new) / 2.0
		return x_mult, (translate_y, 0.0)
	### x_mult > y_mult
	### already in landscape mode scale y, center x (rows)

	x_new = y_mult * float(curshape[0])
	translate_x = (myshape[0] - x_new) / 2.0

	return y_mult, (0.0, translate_x)

def fix_scale_coords(points, scale, translate):
	points = np.array(points).transpose(1,0)

	points[0::3] = scale * points[0::3] + translate[0]
	points[1::3] = scale * points[1::3] + translate[1]

	return points.transpose(1,0)
    
if __name__=="__main__":


    x,y = torch.meshgrid(torch.range(0,5),torch.range(0,5))
    print(y)
    a = torch.randn((5,3,64,64))
    b = torch.randn((5,2,3,11)) 
    # c = affine_warp(a, b)

    mask  = torch.randn((5,11,64,64))

    a = make_warped_stack(mask,a,b)
    print(a.shape)
    repeat((torch.arange(2)*9).type(torch.FloatTensor), 5)