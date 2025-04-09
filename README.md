# SANVOT

This probject hosts the code for implementing our T-CYB paper “Siamese Adaptive Network-Based Accurate and Robust Visual Object Tracking Algorithm for
Quadrupedal Robots” for visual tracking on a Unitree quadruped robot.

# Abstract

Real-time accurate visual object tracking (VOT) for quadrupedal robots is a great challenge when the scale or aspect ratio of moving objects vary. To overcome this challenge, existing methods apply anchor-based schemes that search a handcrafted space to locate moving objects. However, their performances are limited given complicated environments, especially when the speed of quadrupedal robots is relatively high. In this work, a newly designed VOT algorithm for a quadrupedal robot based on a Siamese network is introduced. First, a one-stage detector for locating moving objects is designed and applied. Then, position information of moving objects is fed into a newly designed Siamese adaptive network to estimate their scale and aspect ratio. For regressing bounding boxes of a target object, a box adaptive head with an asymmetric convolution (ACM) layer is newly proposed. The proposed approach is successfully used on a quadrupedal robot, which can accurately track a specific moving object in real-world  complicated scenes.

The code based on the [PySOT](https://github.com/STVIR/pysot) and [SiamBAN](https://github.com/hqucv/siamban). Thanks to their great works.

## Network Structure

### Structure of the SANVOT:

<img src="graph/pic1.jpg" width="720" height="480"/>

### Detection Results on the OTB100 Dataset:

<img src="graph/pic2.jpg" width="720" height="480"/>

### Detection Results in the real-world scenarios:

<img src="graph/pic3.jpg" width="720" height="360"/>

### Detection Video in the real-world scenarios:

[![Experimental results of SANVOT](https://i.ytimg.com/vi/JpYMJhBC1B0/maxresdefault.jpg)](https://www.youtube.com/watch?v=JpYMJhBC1B0 "Experimental results of SANVOT")

## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using SANVOT

### Add SANVOT to your PYTHONPATH

```bash
export PYTHONPATH=/path/to/sanvot:$PYTHONPATH
```

### Webcam demo

```bash
python tools/demo.py \
    --config experiments/sanvot_r50_l234/config.yaml \
    --snapshot experiments/sanvot_r50_l234/model.pth
    # --video demo/bag.avi # (in case you don't have webcam)
```

### Download testing datasets

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [here](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [here](https://pan.baidu.com/s/1et_3n25ACXIkH063CCPOQQ), extraction code: `8fju`. If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker

```bash
cd experiments/sanvot_r50_l234
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
```

The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker

assume still in experiments/sanvot_r50_l234

``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

### Eval on untriee quadrupedal robot with a RealSense D455 Depth Camera

assume in tools

``` bash
python demo.py 	 
```
## If you are interested in this paper, please cite
```BibTeX
@article{cao2025siamese,
  title={Siamese Adaptive Network-Based Accurate and Robust Visual Object Tracking Algorithm for Quadrupedal Robots},
  author={Cao, Zhengcai and Li, Junnian and Shao, Shibo and Zhang, Dong and Zhou, MengChu},
  journal={IEEE Transactions on Cybernetics},
  year={2025},
  volume={55},
  number={3},
  pages={1264--1276},
  publisher={IEEE}
}
```
