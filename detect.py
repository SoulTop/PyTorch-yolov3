# coding='utf-8'
import os
import argparse
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


from torch.utils.data import DataLoader
from utils.datasets import *
from models.yoloV3 import *
from models.yoloLoss import *
from utils.utils import *
from utils.config import *

def detect():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_folder, exist_ok=True)

    config = TESTING_PARAMS
    net = YoloNet(config)
    if args.mGpus:
        net = nn.DataParallel(net)
    net = net.to(device)

    # Restore pretrain model
    if args.weights_path.endswith(".pth"):
        print("load pretrained_Model from {}".format(args.weights_path))
        state_dict = torch.load(args.weights_path)
        net.load_state_dict(state_dict)
    elif args.weights_path.endswith(".weights"):
        print("load pretrained_Model from {}".format(args.weights_path))
        state_dict = torch.load(args.weights_path)
        net.load_darknet_weights(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    net.eval()

    dataloader = DataLoader(
            ImageFolder(args.input_folder, img_size=args.img_size),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.n_cpu,
        )

    classes = load_classes(args.class_path)

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLayer(config["yolo"]["anchors"][i],
                                    classes.__len__(),
                                     args.img_size))

    imgs = []
    img_detections = []

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = input_imgs.to(device)
        with torch.no_grad():
            detections = net(input_imgs)        # 特征图(1*3*13*13) 每个像素点的三种anchors下每个class的得分, 1*(3*85)*13*13
            output_list = []
            for i in range(3):
                o, _ = yolo_losses[i](detections[i], img_size=args.img_size)
                output_list.append(o)
            detections = torch.cat(output_list, 1)      # center_x, center_y, w, h
            detections = non_max_suppression(detections, classes.__len__(), args.conf_thres, args.nms_thres)    # x1,y1,x2,y2

        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        if detections is not None:
            # rescale boxes to original image size
            detections = rescale_boxes(detections, args.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2-x1
                box_h = y2-y1

                # Create a Rectangle patch
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color='white',
                    verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
                # Save generated image with detections

            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig(args.output_folder+'/{}.png'.format(img_i), bbox_inches='tight', pad_inches=0.0)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="test/images/", help="path to testimage folder")
    parser.add_argument("--output_folder", type=str, default="test/output/", help="path to outputimg folder")
    parser.add_argument("--weights_path", type=str, default="weights/official_yolov3_weights_pytorch.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--mGpus", type=bool, default=True, help="Use muti gpus")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    args = parser.parse_args()
    print(args)

    detect()
