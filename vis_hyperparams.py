import os
import numpy as np
import cv2
import glob2 as gb
from util import make_cluster_kp, make_limb_masks, pose137_to_pose122

       
def display(cfg):
    img_base = cfg.PATH.img_base
    kp_base = cfg.PATH.kp_base
    all_img_path = sorted(gb.glob(os.path.join(img_base, "*"+cfg.PATH.img_extension)))
    for i in range(len(all_img_path)):
        img_path = all_img_path[i]
        path = img_path.split("/")[-1]
        filename, file_extension = os.path.splitext(path)  
        key_name = os.path.join(kp_base,filename+'.npy')
        kp_122 = pose137_to_pose122(np.load(key_name)).transpose(1,0)

        W_bias = cfg.TRAIN.CROP.W_bias
        H_bias = cfg.TRAIN.CROP.H_bias
        img = cv2.imread(img_path)[H_bias:H_bias+cfg.HYPERPARAM.img_H,W_bias:W_bias+cfg.HYPERPARAM.img_W,:]
        kp_122[:,0] -= W_bias
        kp_122[:,1] -= H_bias

        vis_gaussian_map(img, kp_122, cfg, flag = 0)    # limbs masks
        vis_gaussian_map(img, kp_122, cfg, flag = 1)    # joint cluster 

        break 

def vis_gaussian_map(img, kp, cfg, flag = False):
    kp_var_root = cfg.HYPERPARAM.kp_var_root
    mask_sigma_perp = cfg.HYPERPARAM.mask_sigma_perp
    cluster = [[1],[2],[3],[4],[5],[6],[7],range(10,27),[78,79]+list(range(28,36))+list(range(47,57)),range(37,46),range(58,78),range(101,122),range(80,101)]
    limbs = [[0,8,9],[1,2,5],[2,3],[3,4],[5,6],[6,7],range(101,122),range(80,101)]
    # joint cluster 
    if flag:
        kp_cluster_canvas = make_cluster_kp(cluster,kp, img.shape[1], img.shape[0], kp_var_root).cpu().numpy()
        for channel in range(len(cluster)):
            kp_canvas = kp_cluster_canvas[channel,:,:]
            kp_canvas = np.expand_dims(kp_canvas,-1).repeat(3,axis=2)
            # tmp = np.concatenate([img/255.0,kp_canvas],axis=1)
            tmp =img/255.0+kp_canvas
            cv2.imshow("joint cluster",tmp)
            cv2.waitKey()
    # limbs masks
    else:
        print(kp.shape)
        limb_masks = make_limb_masks(limbs, kp,img.shape[1], img.shape[0], mask_sigma_perp).cpu().numpy()
        print(limb_masks.shape)
        for channel in range(len(limbs)):
            canvas = limb_masks[channel,:,:][..., None]
            canvas = canvas.repeat(3,axis=-1)
            tmp =img/255.0 + canvas
            cv2.imshow("limbs masks",tmp)
            cv2.waitKey()



if __name__ == "__main__":
    from configs import get_cfg_defaults
    from main import *
    opt_parser = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt_parser.config_path)

    # img_base = "source/Oliver/imgs"
    # kp_base = "source/Oliver/keypoints"
    # cfg.POSE2IMAGE.PATH.img_base = img_base
    # cfg.POSE2IMAGE.PATH.kp_base = kp_base
    display(cfg.POSE2IMAGE)
    