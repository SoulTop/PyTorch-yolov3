import os
import glob
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from utils.dataset_transforms import *
from torch.utils.data import Dataset

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))      # get file list
        self.img_size = img_size
        self.transforms = ImageTrans2Tensor(self.img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Load img
        img = cv2.imread(img_path)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(img)

        return img_path, img

    def __len__(self):
        return len(self.files)

class COCODataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=False, multiscale=False):
        self.img_files = []
        self.label_files = []

        for path in open(list_path, 'r'):
            label_path = path.replace('images', 'labels')\
                .replace('.png', '.txt')\
                .replace('.jpg', '.txt')\
                .strip()
            if os.path.isfile(label_path):
                self.img_files.append(path)
                self.label_files.append(label_path)

        self.img_size = img_size  # (w, h)
        self.max_objects = 100
        self.multiscale = multiscale

        #  transforms and augmentation
        self.transforms = Compose()

        if augment:
            self.transforms.add(ImageBaseAug())

        self.transforms.add(ResizeImage(self.img_size))
        self.transforms.add(ToTensor())

    def __getitem__(self, index):
        # ----------
        # Image
        # ----------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))

        # Image ori size type is w, h
        ori_h, ori_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ----------
        # Label
        # ----------
        label_path = self.label_files[index % len(self.label_files)].rstrip()

        if os.path.exists(label_path):
            label = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        else:
            label = np.zeros((1,5), np.float32)

        if self.transforms is not None:
            img, label = self.transforms(img, label)


        return str([ori_w, ori_h]), img_path, img, label

    def collate_fn(self, batch):
        ori_size, img_paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        imgs = torch.stack([img for img in imgs])

        sample = {'image': imgs, 'label': targets, "ori_size" : ori_size, "img_path": img_paths}

        return sample

    def __len__(self):
        return len(self.img_files)



