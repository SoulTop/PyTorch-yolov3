TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet53",
        "backbone_pretrained": None
        # "backbone_pretrained": "../weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "lr": {
        "backbone_lr": 1e-4,
        "other_lr": 1e-3,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 20,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "data": {
        "train": "data/custom/train.txt",
        "valid": "data/custom/valid.txt",
        "names": "data/custom/classes.names",
        "backup": "backup/",
        "eval": "coco",
    },
    # "data": {
    #     "train": "data/coco/trainvalno5k.txt",
    #     "valid": "data/coco/5k.txt",
    #     "names": "data/coco.names",
    #     "backup": "backup/",
    #     "eval": "coco",
    #  },
    "parallels": [0,1,2,3],                         #  config GPU device
    "working_dir": None,                            #  replace with your working dir
    "pretrain_snapshot": None,                      #  load checkpoint
}

TESTING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet53",
        "backbone_pretrained": None,
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "confidence_threshold": 0.5,
    "pretrain_snapshot": "../weights/official_yolov3_weights_pytorch.pth",
}
