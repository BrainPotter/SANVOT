# SANVOT

This probject hosts the code for implementing the SANVOT algorithm for visual tracking on a Unitree quadruped robot.

The code based on the [PySOT](https://github.com/STVIR/pysot) and [SiamBAN](https://github.com/hqucv/siamban).

## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using SANVOT

### Add SANVOT to your PYTHONPATH

```bash
export PYTHONPATH=/path/to/sanvot:$PYTHONPATH
```

### Download models

Download models in [Model Zoo](MODEL_ZOO.md) and put the `model.pth` in the correct directory in experiments

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

###  Training :wrench:

See [TRAIN.md](TRAIN.md) for detailed instruction.

### License

This project is released under the [Apache 2.0 license](LICENSE). 
