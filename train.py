import os
import sys
import time
import argparse

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluate.eval import evaluate
from utils import config
from utils.datasets import COCODataset
from models.yoloV3 import YoloNet
from models.yoloLoss import *

def _get_optimizer(config, net):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, net.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    else:
        print("freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
        ]

    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        print("Using Adam optimizer.")
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        # Default to sgd
        print("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))
    return optimizer

def _save_checkpoint(state_dict, sub_working_dir):
    # global best_eval_result
    checkpoint_path = os.path.join(sub_working_dir, "model.pth")
    torch.save(state_dict, checkpoint_path)
    print("Model checkpoint saved to %s" % checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension (w*h)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--mGpus", type=bool, default=True, help="multi GPUs for training")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    parser.add_argument("--backbone", type=str, default='darknet53', help="Check bone model")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="if specified starts from checkpoint model", required=True)
    parser.add_argument("--freeze_backbone", type=bool, default=False, help="freeze backbone wegiths to finetune")
    parser.add_argument("--optim_type", type=str, default="adam", help="type of optimizer")
    parser.add_argument("--weight_decay", type=float, default=4e-5, help="weight_decay")
    parser.add_argument("--decay_gamma", type=float, default=0.1, help="decay_gamma")
    parser.add_argument("--decay_step", type=int, default=20, help="decay lr in every epochs")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="A dir to save checkpoint")

    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    args = parser.parse_args()
    print(args)

    config = config.TRAINING_PARAMS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = config['data']['train']
    valid_path = config['data']['valid']
    classes = load_classes(config['data']['names'])
    if classes is None:
        raise Exception("Wrong classes file, please check the classesname path in utils/config.py!")
    num_classes = classes.__len__()
    config['yolo']['classes'] = num_classes

    print('train path is: {}\nvalid path is: {}\nnumber of classes is: {}\n'.format(train_path, valid_path, num_classes))

    writer = SummaryWriter('logs')

    '''
    get args to modified the config
    '''
    config["img_size"] = args.img_size
    config["model_params"]["backbone_pretrained"] = args.pretrained_weights if args.pretrained_weights != "None" else None

    config["lr"]["freeze_backbone"] = args.freeze_backbone if args.freeze_backbone is not None else False
    config["lr"]["decay_gamma"] = args.decay_gamma if args.decay_gamma is not None else 0.1
    config["lr"]["decay_step"] = args.decay_step if args.decay_step is not None else 20
    config["optimizer"]["type"] = args.optim_type if args.optim_type is not None else 'sgd'
    config["optimizer"]["weight_decay"] = args.weight_decay if args.weight_decay is not None else '4e-5'

    # Create sub_working_dir
    sub_working_dir = '{}/{}/size{}x{}/{}'.format(
        args.checkpoint_dir, config['model_params']['backbone_name'],
        args.img_size, args.img_size,
        time.strftime("%Y%m%d%H%M%S", time.localtime()))

    os.makedirs("output", exist_ok=True)
    os.makedirs(sub_working_dir, exist_ok=True)

    '''
    # end there!
    '''
    # load and initialize network
    net = YoloNet(config)
    net.apply(weights_init_normal)
    if args.mGpus:
        nn.DataParallel(net)
    net.to(device)
    net.train()

    # Restore pretrain model
    if config["model_params"]["backbone_pretrained"]:
        if config["model_params"]["backbone_pretrained"](".pth"):
            net.load_state_dict(torch.load(config["model_params"]["backbone_pretrained"]))
        else:
            net.load_darknet_weights(config["model_params"]["backbone_pretrained"])

    datasets = COCODataset(train_path, args.img_size)
    dataloader = DataLoader(datasets,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.n_cpu,
                            pin_memory=True,
                            collate_fn=datasets.collate_fn)


    # Optimizer and learning rate
    # optimizer = _get_optimizer(config, net)
    # lr_scheduler = optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=config["lr"]["decay_step"],
    #     gamma=config["lr"]["decay_gamma"])

    optimizer = optim.Adam(net.parameters())

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLayer(config["yolo"]["anchors"][i], num_classes, args.img_size))

    # Start the training loop
    print("Start training...")
    batches_done = 0
    for epoch in range(args.epochs):
        print("starting epoch ", epoch)
        for step, samples in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + step
            images, targets = samples["image"].to(device), samples["label"].to(device)
            start_time = time.time()

            outputs = net(images)
            loss = 0

            metricValues = []
            for i in range(3):
                _, _loss_item, metric_Value = yolo_losses[i](outputs[i], targets, args.img_size)
                loss += _loss_item
                metricValues.append(metric_Value)

            loss.backward()

            # Accumulates gradient before each step
            if batches_done % args.gradient_accumulations:
                optimizer.step()
                optimizer.zero_grad()

            for i, layers in enumerate(metricValues):
                for key, val in layers.items():
                    if key != "grid_size":
                        writer.add_scalar(key + '_' + str(i), val, batches_done)
            writer.add_scalar('total_loss', loss.item(), batches_done)

            if step > 0 and step % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("lr",
                                  lr,
                                  config["global_step"])

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model=net,
                config=config,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=args.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            for key, val in evaluation_metrics:
                metricValues.append(metric_Value)
                writer.add_scalar('val/'+key, val, epoch)

        if epoch % args.checkpoint_interval == 0:
            torch.save(net.state_dict(), args.checkpoint_dir + "/yolov3_ckpt_%d.pth" % epoch)

        print("One Batch`s time is :", time.time()-start_time, "And Total_loss is:", loss.item())
        print(f"---- mAP {AP.mean()}")
        print('--'*10)

        # lr_scheduler.step()

    _save_checkpoint(net.state_dict(), sub_working_dir)
    print("Bye~")
