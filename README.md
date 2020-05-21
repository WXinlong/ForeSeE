# Task-Aware Monocular Depth Estimation for 3D Object Detection

This project hosts the code for implementing the ForeSeE algorithm for depth estimation.


> [**Task-Aware Monocular Depth Estimation for 3D Object Detection**](https://arxiv.org/abs/1909.07701),     
> Xinlong Wang, Wei Yin, Tao Kong, Yuning Jiang, Lei Li, Chunhua Shen    
> *AAAI, 2020*


## Installation

This implementation is based on [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction). Please refer to [INSTALL.md](INSTALL.md) for installation.

## Dataset

Please refer to [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) for details.
The annotation files of [KITTI Object subset](https://github.com/WXinlong/ForeSeE/tree/master/datasets/KITTI_object/annotations) used in our work are provided.

## Models
Download the trained model from this [link](https://cloudstor.aarnet.edu.au/plus/s/M3LFxiDPZkMKrtw) and put it under experiments/foresee/.

## Testing

      cd experiments/foresee
      sh test.sh
 
## Training

      cd experiments/foresee
      sh train.sh
  
## Citations

Please consider citing our papers in your publications if the project helps your research. BibTeX reference is as follows.

```
@InProceedings{wang2020foresee, 
  title={Task-Aware Monocular Depth Estimation for 3D Object Detection}, 
  author = {Wang, Xinlong and Yin, Wei and Kong, Tao and Jiang, Yuning, and Li, Lei and Shen, Chunhua},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2020}
}
```
