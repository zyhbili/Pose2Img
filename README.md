# Pose2Img
Upper body image synthesis from skeleton(Keypoints).
Sub module in the ICCV-2021 paper "Speech Drives Templates: Co-Speech Gesture Synthesis with Learned Templates".
[[arxiv](https://arxiv.org/abs/2108.08020) / [github](https://github.com/ShenhanQian/SpeechDrivesTemplates)]

This is a modified implementation of [Synthesizing Images of Humans in Unseen Poses](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1978.pdf). 


## Setup

To install dependencies, run

``` bash
pip install -r requirements.txt
```

To run this module, you need two NVIDIA gpus with at least 11 GB respectively.
Our code is tested on Ubuntu 18.04LTS with Python3.6.


## Train on the custom dataset

For your own dataset, you need to modify custom config.yaml.

``` bash
python main.py \
    --name test \
    --config_path {config_path}.yaml \
    --batch_size 1 \
```

- The raw keypoints for each frame is of shape (3, 137)

## Demo

Generate a realistic video for **Oliver** from {keypoints}.npz. You need to do some minor modifications to $configs/yaml/oliver.yaml$.

``` bash
python inference.py \
    --cfg_path configs/yaml/oliver.yaml \
    --name test \
    --npz_path target_pose/oliver_pose.npz \
    --wav_path target_pose/oliver_audio.wav
```

- In the result directory, you can find `jpg` files which correspond to the npz.


## Demo Dataset and Checkpoint
- Coming soon



## Citation
If you find this code useful for your research, please use the following BibTeX entry.
```
@inproceedings{qian2021speech,
  title={Speech Drives Templates: Co-Speech Gesture Synthesis with Learned Templates},
  author={Qian, Shenhan and Tu, Zhi and Zhi, YiHao and Liu, Wen and Gao, Shenghua},
  journal={International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

