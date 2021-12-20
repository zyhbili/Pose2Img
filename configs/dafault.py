from yacs.config import CfgNode as CN


_C = CN()


_C.POSE2IMAGE = CN()
_C.POSE2IMAGE.TRAIN = CN()
_C.POSE2IMAGE.TRAIN.CACHING = False
_C.POSE2IMAGE.TRAIN.NUM_EPOCHS = 100
_C.POSE2IMAGE.TRAIN.BATCH_SIZE = 4

_C.POSE2IMAGE.TRAIN.CROP = CN()
_C.POSE2IMAGE.TRAIN.CROP.H_bias =0
_C.POSE2IMAGE.TRAIN.CROP.W_bias =0


_C.POSE2IMAGE.HYPERPARAM = CN()
_C.POSE2IMAGE.HYPERPARAM.kp_var_root = []
_C.POSE2IMAGE.HYPERPARAM.mask_sigma_perp = []
_C.POSE2IMAGE.HYPERPARAM.img_H =0
_C.POSE2IMAGE.HYPERPARAM.img_W =0
_C.POSE2IMAGE.HYPERPARAM.scale =1.0     #downsample the source img and keypoints


_C.POSE2IMAGE.PATH = CN()
_C.POSE2IMAGE.PATH.img_base = ""
_C.POSE2IMAGE.PATH.kp_base = ""
_C.POSE2IMAGE.PATH.img_extension = ""

_C.POSE2IMAGE.INFER = CN()
_C.POSE2IMAGE.INFER.root_node = []    # compute_root_node in inference.py
_C.POSE2IMAGE.INFER.ckpt_path = ""
_C.POSE2IMAGE.INFER.scale = 1.0
_C.POSE2IMAGE.INFER.src_img_path = ""
_C.POSE2IMAGE.INFER.src_kp_path = ""







def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  return _C.clone()
