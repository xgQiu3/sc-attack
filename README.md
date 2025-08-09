# Shape-Consistent Attack

<p align="center"><img width="100%" src="teaser.png" /></p>

## Get Started

### Setup
The code is tested with Python3.7, Pytorch == 1.8.0 and CUDA == 11.1
```
conda env create -f environment.yml
conda activate shape_consis
```
### Dataset
Download the dataset [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and [ShapeNetPart](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip).


### Pretrained Models
You can download the following [victim models](https://drive.google.com/file/d/1L25i0l6L_b1Vw504WQR8-Z0oh2FJA0G9/view?usp=sharing):

[PointNet](https://github.com/charlesq34/pointnet), 
[PointNet++](https://github.com/charlesq34/pointnet2), 
[DGCNN](https://github.com/WangYueFt/dgcnn), 
[PointCNN](https://github.com/yangyanli/PointCNN), 
[PointConv](https://github.com/DylanWusee/pointconv_pytorch), 
[RS-CNN](https://github.com/Yochengliu/Relation-Shape-CNN), 
[PCT](https://github.com/Strawberry-Eat-Mango/PCT_Pytorch), 
[PAConv](https://github.com/CVMI-Lab/PAConv), 
[SimpleView](https://github.com/princeton-vl/SimpleView)
[CurveNet](https://github.com/tiangexiang/CurveNet)

and the [transformer](https://cloud.tsinghua.edu.cn/f/9be5d9dcbaeb48adb360/?dl=1).


## Attack
To launch the white-box attack, simply run:
```
CUDA_VISIBLE_DEVICES='x' python main.py
```

As for the black-box attack, run:
```
CUDA_VISIBLE_DEVICES='x' python main.py --surrogate_model_1 your/white_box/model --target_model your/black_box/model
```
