import numpy as np
import os, argparse

from torch.utils.data import Dataset, DataLoader
from model import WarpModel
import torch.nn as nn
import time
from util import *
import glob2 as gb
import cv2
from configs import get_cfg_defaults
import torch

batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_root_node(npy_base):
    cnt = 0
    sum_result = np.zeros(2)
    for npy_file in os.listdir(npy_base):
        kp_137 = np.load(os.path.join(npy_base,npy_file))
        sum_result += kp_137[:2,1]
        cnt +=1
        print(cnt)
        if cnt == 200:
            break
    return sum_result/cnt


class NPZ_infer(Dataset):
    def __init__(self,kpt_path, cfg):
        self.cfg = cfg
        np_kpts = np.load(kpt_path)['poses_pred_batch'][0]  # frames, 2, 119
        self.img_extension = cfg.PATH.img_extension
        self.img_base = cfg.PATH.img_base
        self.W_bias = cfg.TRAIN.CROP.W_bias
        self.H_bias = cfg.TRAIN.CROP.H_bias

        self.img_H = cfg.HYPERPARAM.img_H
        self.img_W = cfg.HYPERPARAM.img_W
        self.scale = cfg.HYPERPARAM.scale
        self.bias = np.array([[self.W_bias],[self.H_bias]])
        self.kp_path = sorted(gb.glob(os.path.join(cfg.PATH.kp_base, '*.npy')))

        self.root_node_mean = np.array(cfg.INFER.root_node)   # root for oliver


        np_kpts = np.insert(np_kpts, 1, np.zeros(2), axis=2)
        np_kpts = (np_kpts / cfg.INFER.scale) + self.root_node_mean

        np_kpts = np_kpts - self.bias

        self.np_kpts = np_kpts.transpose(0,2,1)
        self.np_kpts /= self.scale
        self.limbs = [[0,8,9],[1,2,5],[2,3],[3,4],[5,6],[6,7],range(101,122),range(80,101)]
        
        self.source_dict = self.process_source()

    def __len__(self):
        return self.np_kpts.shape[0]


    def __getitem__(self, idx):
        kp_tgt = self.np_kpts[idx]
        kp_src = self.source_dict["kp"]
        src_in = self.source_dict["img"]
        
        
        trans_in = get_limb_transforms(self.limbs, kp_src, kp_tgt)

        sample = {
            'src_in': src_in,
            'kp_src': kp_src,
            'kp_tgt': kp_tgt,
            'trans_in': trans_in,
        }
        return sample

    def process_source(self):
        kp_path = self.cfg.INFER.src_kp_path
        kp = np.load(kp_path)
        kp = pose137_to_pose122(kp).transpose(1,0)
        path = kp_path.split("/")[-1]
        filename, _ = os.path.splitext(path)  
        img_path = os.path.join(self.img_base,filename+self.img_extension)
        img = cv2.imread(img_path).transpose(2,0,1)
        img = img[:,self.H_bias:self.H_bias+self.img_H,self.W_bias:self.W_bias + self.img_W]/255.0*2.0 - 1.0

        scale = self.scale
        if scale !=1.0:
            img = img.transpose(1,2,0)
            img = cv2.resize(img,(int(self.img_H/scale),int(self.img_W/scale)))
            img = img.transpose(2,0,1)
            kp[:,0] -= self.W_bias
            kp[:,1] -= self.H_bias
            kp/=scale
        else:
            kp[:,0] -= self.W_bias
            kp[:,1] -= self.H_bias
        return  {'img': img, 'kp' : kp}


def infer_only(cfg, dataloader):
    G = WarpModel(n_joints = 13, n_limbs = 8)
    ckpt = torch.load(cfg.INFER.ckpt_path)

    tmp = nn.DataParallel(G)
    tmp.load_state_dict(ckpt['G'])
    G.load_state_dict(tmp.module.state_dict())
    del tmp
    G = nn.DataParallel(G)
    G.to(device)
    G.eval()

    results = []
    with torch.no_grad():
        for batch in dataloader:
            src_in = batch["src_in"].float().to(device)
            trans_in = batch["trans_in"].float().to(device)
            kp_src = batch["kp_src"].float().to(device)
            kp_tgt = batch["kp_tgt"].float().to(device)

            
            scale = cfg.HYPERPARAM.scale
            src_mask_prior = batchify_mask_prior(kp_src,int(cfg.HYPERPARAM.img_W/scale), int(cfg.HYPERPARAM.img_H/scale), cfg.HYPERPARAM.mask_sigma_perp)
            pose_src = batchify_cluster_kp(kp_src,int(cfg.HYPERPARAM.img_W/scale), int(cfg.HYPERPARAM.img_H/scale), cfg.HYPERPARAM.kp_var_root)
            pose_target = batchify_cluster_kp(kp_tgt,int(cfg.HYPERPARAM.img_W/scale), int(cfg.HYPERPARAM.img_H/scale), cfg.HYPERPARAM.kp_var_root)
            g_out = G(src_in, pose_src, pose_target, src_mask_prior, trans_in)

            print(g_out.shape)
            results.append(g_out.cpu())
            del g_out
        
        video = ((torch.cat(results, dim=0)+1.0)/2.0)*255
        print("video",video.shape)
        return video.permute(0,2,3,1)

def img2vid(output_path, audio_path,name):
    vid_tic = time.time()
    output_dir = os.path.join(output_path, name+".mp4")
    input_audio = ffmpeg.input(audio_path)
    img_dir = os.path.join(output_path, name)
    
    input_video = (
        ffmpeg
        .input('%s/*.jpg' % img_dir, pattern_type='glob', framerate=15)
    )

    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_dir).run(quiet=False)
    vid_toc = (time.time() - vid_tic)
    print(vid_toc)

def save_img(cfg, output_path, npz_path, wav_path, name, to_video = False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_dir = os.path.join(output_path, name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    dataset = NPZ_infer(npz_path,cfg)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)    

    imgs = infer_only(cfg, dataloader)
    imgs = imgs.cpu().numpy()
    print(imgs.shape)
    for idx in range(imgs.shape[0]):
        img_path = os.path.join(img_dir, '%03d.jpg' % idx)
        cv2.imwrite(img_path, imgs[idx])
    print("done")
    if to_video:
        img2vid(output_path, wav_path, name)





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", default = "configs/yaml/oliver.yaml", help="checkpoint path", type=str)
    parser.add_argument("--name", default = 'test',help="experiment name", type=str)
    parser.add_argument("--output_path", default = './results',help="output path", type=str)
    parser.add_argument("--npz_path", default ='{your root}/target_pose/epoch0-DEMO-step1.npz', help="target pose npz path", type=str)
    parser.add_argument("--wav_path", default ='{your root}/target_pose/epoch0-DEMO-step1.mp4',help="target audio path", type=str)

    args = parser.parse_args()

    cfg_path = args.cfg_path
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg = cfg.POSE2IMAGE

    exp_name = args.name
    npz_path = args.npz_path
    wav_path = args.wav_path
    output_path = args.output_path

    save_img(cfg, output_path, npz_path, wav_path, exp_name+"_rgb")






  