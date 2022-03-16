import os
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
import glob2 as gb
from util import *
from torch.multiprocessing import Manager
from configs import get_cfg_defaults


class WarpDataset(Dataset):
    """Speech2Gesture dataset."""
    def __init__(self, config_path, mode):

        cfg = get_cfg_defaults()
        cfg.merge_from_file(config_path)
        cfg = cfg.POSE2IMAGE

        all_kp_path = sorted(gb.glob(os.path.join(cfg.PATH.kp_base, '*.npy')))[::]
        self.img_extension = cfg.PATH.img_extension
        self.img_base = cfg.PATH.img_base

        self.mode = mode

        self.W_bias = cfg.TRAIN.CROP.W_bias
        self.H_bias = cfg.TRAIN.CROP.H_bias

        self.img_H = cfg.HYPERPARAM.img_H
        self.img_W = cfg.HYPERPARAM.img_W
        self.scale = cfg.HYPERPARAM.scale

        self.limbs = [[0,8,9],[1,2,5],[2,3],[3,4],[5,6],[6,7],range(101,122),range(80,101)]

        if mode == "train":
            self.kp_path = all_kp_path[256:]
            self.len = len(self.kp_path) 
        elif mode == "val":
            self.kp_path = all_kp_path[:256] 
            self.len = len(self.kp_path)
        if cfg.TRAIN.CACHING:
            self.cache_dict = Manager().dict()
        self.cfg = cfg

    def __len__(self):
        return self.len

    
    def get_data(self, idx):
        kp_path = self.kp_path[idx]
        kp = np.load(kp_path)
        if kp.shape[1]==137:
            kp = pose137_to_pose122(kp).transpose(1,0)
        else:
            kp = kp
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
        
    def lookup_cache(self, idx):
        if self.cfg.TRAIN.CACHING:
            if idx in self.cache_dict:
                return self.cache_dict[idx]
            sample = self.get_data(idx)
            self.cache_dict[idx] = sample
            return sample
        else:
            return self.get_data(idx)

    def __getitem__(self, idx):
        if self.mode == "train": 
            while True:
                unpair_idx = torch.randint(0, self.len,[1])[0]
                if unpair_idx != idx:
                    break 
            src = self.lookup_cache(idx)
            tgt = self.lookup_cache(unpair_idx)
            
            src_in = src["img"]
            target = tgt["img"]
            kp_src = src["kp"]
            kp_tgt = tgt["kp"]
            
            trans_in = get_limb_transforms(self.limbs, kp_src, kp_tgt)
        
        if self.mode == "val":
            unpair_idx = 0
            src = self.lookup_cache(unpair_idx)
            tgt = self.lookup_cache(idx)
            
            src_in = src["img"]
            target = tgt["img"]
            kp_src = src["kp"]
            kp_tgt = tgt["kp"]
            trans_in = get_limb_transforms(self.limbs, kp_src, kp_tgt)

        sample = {
            'src_in': src_in,
            'kp_src': kp_src,
            'kp_tgt': kp_tgt,
            'trans_in': trans_in,
            'target': target
        }
        return sample
       


if __name__ == "__main__":
    batch_size = 8
    dataset = WarpDataset("configs/yaml/oliver.yaml",mode = "train")
    print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=batch_size,
            shuffle=False, num_workers=8)


    for data in dataloader:
        for key in data.keys():
            print(key, data[key].shape)
        break
    
   

       

    