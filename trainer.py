from model import WarpModel, VGGLoss, Discriminator
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
import os, pdb
from dataset import WarpDataset
import threading
from d_loss import *
from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Trainer():
    def __init__(self, opt_parser):
        print('Run on device {}'.format(device))
        self.opt_parser = opt_parser

        self.G = WarpModel(n_joints = 13, n_limbs = 8)
        self.D = Discriminator()
        self.G.to(device)
        self.D.to(device)
        
        self.optimizer = torch.optim.Adam(self.G.parameters(), lr=opt_parser.lr, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(params = self.D.parameters(), lr=opt_parser.lr/2, betas=(0.5, 0.999))
        self.criterionDreal = LossDSCreal()
        self.criterionDfake = LossDSCfake()

        # optimizer

        if (opt_parser.load_ckpt_path != ''):
            ckpt = torch.load(opt_parser.load_ckpt_path)
            try:
                self.G.load_state_dict(ckpt['G'])
                self.D.load_state_dict(ckpt['D'])

            except:
                tmp = nn.DataParallel(self.G)
                tmp.load_state_dict(ckpt['G'])
                self.G.load_state_dict(tmp.module.state_dict())
                del tmp

                tmp = nn.DataParallel(self.D)
                tmp.load_state_dict(ckpt['D'])
                self.D.load_state_dict(tmp.module.state_dict())

                del tmp

            self.optimizerD.load_state_dict(ckpt['optD'])
            self.optimizer.load_state_dict(ckpt['opt'])
            print("load success")

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs in G mode!")
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

  

            
        self.dataset = WarpDataset(opt_parser.config_path,mode = "train")
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=opt_parser.batch_size,
                                                        shuffle=True,
                                                        num_workers=opt_parser.num_workers)
        self.dataset_val = WarpDataset(opt_parser.config_path,mode = "val")
        self.dataloader_val = torch.utils.data.DataLoader(self.dataset_val,
                                                        batch_size=opt_parser.batch_size*2,
                                                        shuffle=False,
                                                        num_workers=opt_parser.num_workers)

        # criterion
        self.criterionL1 = nn.L1Loss()
        self.criterionVGG = VGGLoss()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs in VGG model!")
            self.criterionVGG = nn.DataParallel(self.criterionVGG)
        self.criterionVGG.to(device)

        # writer
        self.writer = SummaryWriter(log_dir=os.path.join(opt_parser.log_dir, opt_parser.name))
        self.count = 0


    def __train_with_D__(self, epoch):
        st_epoch = time.time()

        self.G.train()
       

        g_time = 0.0
        for i, batch in enumerate(self.dataloader):
            if(i >= len(self.dataloader)-2):
                break
            st_batch = time.time()

            src_in = batch["src_in"].float().to(device)
            trans_in = batch["trans_in"].float().to(device)
            target = batch["target"].float().to(device)
            kp_src = batch["kp_src"].float().to(device)
            kp_tgt = batch["kp_tgt"].float().to(device)

            
            scale = self.dataset.scale
            src_mask_prior = batchify_mask_prior(kp_src,int(self.dataset.img_W/scale), int(self.dataset.img_H/scale), self.dataset.cfg.HYPERPARAM.mask_sigma_perp)
            pose_src = batchify_cluster_kp(kp_src,int(self.dataset.img_W/scale), int(self.dataset.img_H/scale), self.dataset.cfg.HYPERPARAM.kp_var_root)
            pose_target = batchify_cluster_kp(kp_tgt,int(self.dataset.img_W/scale), int(self.dataset.img_H/scale), self.dataset.cfg.HYPERPARAM.kp_var_root)
            
              
            # print(pose_src.shape)
            # print(pose_target.shape)
            # print(src_mask_prior.shape)
            # print(src_in.shape)
            g_out = self.G(src_in, pose_src, pose_target, src_mask_prior, trans_in)
            loss_l1 = 10 * self.criterionL1(g_out, target)
            loss_vgg, loss_style = self.criterionVGG(g_out, target, style=True)

            loss_vgg, loss_style = torch.mean(loss_vgg), torch.mean(loss_style)

            real_score = self.D(g_out)

            lossG = 0.5 * self.criterionDreal(real_score)
            loss = loss_l1  + loss_vgg + loss_style + lossG  

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #update D:
            g_out.detach_()
            self.optimizerD.zero_grad()
            real_score = self.D(target)
            fake_score = self.D(g_out)
            lossDfake = self.criterionDfake(fake_score)
            lossDreal = self.criterionDreal(real_score)
            lossD =  (lossDfake + lossDreal)/2 
            lossD.backward()
            self.optimizerD.step()

            # log
            self.writer.add_scalar('loss/Loss', loss.cpu().detach().numpy(), self.count)
            self.writer.add_scalar('loss/L1', loss_l1.cpu().detach().numpy(), self.count)
            self.writer.add_scalar('loss/Vgg', loss_vgg.cpu().detach().numpy(), self.count)
            self.writer.add_scalar('loss/lossG', lossG.cpu().detach().numpy(), self.count)
            self.writer.add_scalar('loss/style', loss_style.cpu().detach().numpy(), self.count)
            self.writer.add_scalar('loss/realscore', real_score.mean().cpu().detach().numpy(), self.count)
            self.writer.add_scalar('loss/fakescore', fake_score.mean().cpu().detach().numpy(), self.count)
            self.count += 1


            print("Epoch {}, Batch {}/{}, loss {:.4f}, l1 {:.4f}, vggloss {:.4f}, styleloss {:.4f} time {:.4f}".format(
                epoch, i, len(self.dataset) // self.opt_parser.batch_size,
                loss.cpu().detach().numpy(),
                loss_l1.cpu().detach().numpy(),
                loss_vgg.cpu().detach().numpy(),
                loss_style.cpu().detach().numpy(),
                          time.time() - st_batch))

            g_time += time.time() - st_batch


        print('Epoch time usage:', time.time() - st_epoch, 'I/O time usage:', time.time() - st_epoch - g_time, '\n=========================')

        if(epoch % self.opt_parser.ckpt_epoch_freq == 0):
            self.__save_model__('{:02d}'.format(epoch), epoch)
 

    def __save_model__(self, save_type, epoch):
        try:
            os.makedirs(os.path.join(self.opt_parser.ckpt_dir, self.opt_parser.name))
        except:
            pass

        torch.save({
        'G': self.G.state_dict(),
        'D': self.D.state_dict(),
        'opt': self.optimizer.state_dict(),
        'optD': self.optimizerD.state_dict(),
        'epoch': epoch
        }, os.path.join(self.opt_parser.ckpt_dir, self.opt_parser.name, 'ckpt_{}.pth'.format(save_type)))
        


    def run(self):
        for epoch in range(self.opt_parser.nepoch):
            self.__train_with_D__(epoch)
            print("epoch:", epoch)
            with torch.no_grad():
                if epoch%2 == 0:
                    self.__val_pass__(epoch,infer = True)
                else:
                    self.__val_pass__(epoch,infer = False)
        self.__save_model__('last', epoch)



    def __val_pass__(self, epoch, infer = False):
        self.G.eval()
        results = []
        epoch_loss_l1 = 0
        epoch_loss_vgg = 0
        epoch_loss = 0
        epoch_loss_style = 0
        for batch in self.dataloader_val:
            src_in = batch["src_in"].float().to(device)
            trans_in = batch["trans_in"].float().to(device)
            target = batch["target"].float().to(device)
            kp_src = batch["kp_src"].float().to(device)
            kp_tgt = batch["kp_tgt"].float().to(device)
            
            scale = self.dataset.scale
            src_mask_prior = batchify_mask_prior(kp_src,int(self.dataset.img_W/scale), int(self.dataset.img_H/scale), self.dataset.cfg.HYPERPARAM.mask_sigma_perp)
            pose_src = batchify_cluster_kp(kp_src,int(self.dataset.img_W/scale), int(self.dataset.img_H/scale), self.dataset.cfg.HYPERPARAM.kp_var_root)
            pose_target = batchify_cluster_kp(kp_tgt,int(self.dataset.img_W/scale), int(self.dataset.img_H/scale), self.dataset.cfg.HYPERPARAM.kp_var_root)
            
       

            g_out = self.G(src_in, pose_src, pose_target, src_mask_prior, trans_in)

            loss_l1 = 10 * self.criterionL1(g_out, target)
            epoch_loss_l1 += loss_l1
            loss_vgg, loss_style = self.criterionVGG(g_out, target, style=True)

            loss_vgg, loss_style = torch.mean(loss_vgg), torch.mean(loss_style)
            epoch_loss_vgg += loss_vgg
            epoch_loss_style += loss_style
            loss = loss_l1  + loss_vgg + loss_style
            epoch_loss +=loss

            if not infer:
                continue
            tmp = g_out.unsqueeze(0).flip([2]).cpu()
            results.append(tmp)
            # log
        self.writer.add_scalar('val/loss', epoch_loss.cpu().detach().numpy()/256, epoch)
        self.writer.add_scalar('val/l1', 10*epoch_loss_l1.cpu().detach().numpy()/256, epoch)
        self.writer.add_scalar('val/vgg', epoch_loss_vgg.cpu().detach().numpy()/256, epoch)
        self.writer.add_scalar('val/style', epoch_loss_style.cpu().detach().numpy()/256, epoch)   
   
        
        if not infer:
            return

        video = (torch.cat(results, dim=0).view(-1,64,3,int(self.dataset.img_H/scale),int(self.dataset.img_W/scale))+1.0)/2.0
        w_thread = threading.Thread(target=write_video, args=(self.writer, video, epoch))
        w_thread.start()

def write_video(writer, video, epoch):
    writer.add_video(tag="val_video", vid_tensor= video, global_step=epoch, fps=15)
    print("epoch%s save success"%str(epoch))

        




    





