# coding='utf-8'
from __future__ import division

from utils.datasets import *
from models.yoloV3 import *
from models.yoloLoss import *
from utils.utils import *
from utils.config import *

import os
import argparse
import tqdm
import random
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader



def evaluate(model, config, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = COCODataset(list_path='../'+config['data']['valid'], img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLayer(config["yolo"]["anchors"][i],
                                    num_classes=config['yolo']['classes'],
                                     img_size=img_size))

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, samples in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        images, targets = samples["image"].to(device), samples["label"].to(device)
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(images.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            output_list = []
            for i in range (3):
                o, _ = yolo_losses[i](outputs[i], img_size=img_size)
                output_list.append(o.cpu())
            detections = torch.cat(output_list, 1)
            outputs = non_max_suppression(detections, num_classes=config['yolo']['classes'],conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="../checkpoints/darknet53/size416x416/20200417161243/model.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TRAINING_PARAMS
    class_names = load_classes('../'+config['data']["names"])
    num_classes = class_names.__len__()
    config['yolo']['classes'] = num_classes

    # Initiate model
    model = YoloNet(config).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        config,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
