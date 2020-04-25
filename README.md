# pure pytorch yoloV3

 The project is a pure PyTorch Implement of YOLOv3 with support to train your own dataset.
 
 At the same time, it is easy to use, and you just need to install some packages in requirements.

## Overview

[YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

### Highlight
- There is no need for darknet & yoloV3 cfg parser to implement yolov3.
- Besides python packages, there are few dependencies.
- It is Easy to custom backbone network.
- Support Linux & Windows

## TODO

- [x] darknet53 + yoloV3
- [x] train on your datasets
- [x] eval
- [x] detect.py
- [ ] support video detect
- [ ] add more backbone net

## Installation
### Requirements

- python 3
- pytorch >= 1.0
- torchvision
- imaug
- opencv-python
- matplotlib, tqdm, collections

### Git

```bash
git clone https://github.com/SoulTop/PyTorch-yolov3.git
cd PyTorch-yolov3
```

## Detect Images

1. download the pyotch yolov3 weights `(yolov3_weights_pytorch.pth)` into weights folder:

<div align='center'>

[Google Driver](https://drive.google.com/open?id=1Bm_CLv9hP3mMQ5cyerKRjvt7_t1duvjI) | [Baidu Driver](https://pan.baidu.com/s/1gx-XRUE1NTfIMKkQ1L0awQ)

</div>

2. run `detect.py`

```bash
python detct.py --input_folder=test/images/ --output_folder=test/output/
                --weights_path=weights/official_yolov3_weights_pytorch.pth
                --class_path=data/coco.names
                --conf_thres=0.5 --nms_thres=0.45 --img_size=416
                --batch_size=1 --mGpus=True
```

3. The result images will be saved in `test/output` folder

## Train

### Train on COCO

#### Prepare COCO Dataset

Linux:
```
cd data/
bash get_coco_dataset.sh
```
Windows:  download by yourself

#### Prepare pre-trained model
1. If you want to load pre-trained backbone weights, download it from: [Google Drive](https://drive.google.com/open?id=1VYwHUznM3jLD7ftmOSCHnpkVpBJcFIOA) | [Baidu Drive](https://pan.baidu.com/s/1axXjz6ct9Rn9GtDTust6DA)

2. Move the weights model to `weights/` folder.

#### Modify training parameters
1. Reviewing `TRAINING_PARAMS` in `uitils/config.py`.
2. Adjust parameters which you want to change, such as `"data"` item.

#### Strat Training 
```
python train.py --epochs=100 --batch_size=8
                --img_size=416 
                --backbone=darknet53
                --pretrained=“weights/darknet53_weights_pytorch.pth”
```
In addition to the parameters displayed above, there are:
```bash
- mGpus = True,       # to use multi GPUs
- multiscale_training = True,
- freeze_backbone = False,      # freeze_backbone paramters
- checkpoint_dir="checkpoints"  # The path to save the checkpoints models.

... can check it in train.py
```


### Train on Custom Dataset
#### Make Custom Datasets
1. Put your images in `data/custom/images/`;
2. Put your labels in `data/custom/images/`;
3. Add your class names to `data/custom/classes.names`
4. Generate `train.txt, valid.txt`.

#### Modify cfg

1. Open the `utils/config.py`.
2. Modify :
```py
1. the amount of classses

    "yolo": {
        "classes": 1,   # amount of classes
    }


2. datasets configuration

    "data": {
        "train": "data/custom/train.txt",
        "valid": "data/custom/valid.txt",
        "names": "data/custom/classes.names",
        "backup": "backup/",
        "eval": "coco",
    },
```
#### Train

```bash
$ python train.py --epochs=100 --batch_size=8
                --img_size=416 
                --backbone=darknet53
                --pretrained=“weights/darknet53_weights_pytorch.pth”
```