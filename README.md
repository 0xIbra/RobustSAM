# RobustSAM: Segment Anything Robustly on Degraded Images (CVPR 2024 Highlight)

Official repository for RobustSAM: Segment Anything Robustly on Degraded Images



[Project Page](https://robustsam.github.io/) | [Paper]() | [Video]() | [Dataset]()



## Updates
- Feb 2024: ✨ RobustSAM was accepted into CVPR 2024!


## Setup
1) Create a conda environment and activate it.
```
conda create --name robustsam python=3.10 -y
conda activate robustsam
```
2) Clone and enter into repo directory.
```
git clone https://github.com/robustsam/RobustSAM
cd RobustSAM
```
3) Use command below to check your CUDA version.
```
nvidia-smi
```
4) Replace the CUDA version with yours in command below.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[$YOUR_CUDA_VERSION]
# For example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 # cu117 = CUDA_version_11.7
```
4) Install remaining dependencies
```
pip install -r requirements.txt
```

## Demo


## Reference
If you find this work useful, please consider citing us!
```python
@inproceedings{chen2024robustsam,
  title={RobustSAM: Segment Anything Robustly on Degraded Images},
  author={Chen, Wei-Ting and Vong, Yu-Jiet and Kuo, Sy-Yen and Ma, Sizhou and Wang, Jian},
  journal={CVPR},
  year={2024}
}
```


## Acknowledgements
We thank the authors of [SAM](https://github.com/facebookresearch/segment-anything) from which our repo is based off of.

