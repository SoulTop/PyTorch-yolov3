import torch
import imgaug.augmenters as iaa
import cv2
import numpy as np

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img, label=None):
        if label is None:
            raise Exception("There is lack of label please check it!")
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

    def add(self, transform):
        self.transforms.append(transform)

class ToTensor(object):
    def __call__(self, img, label):
        image = img

        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)

        target = torch.zeros((len(label), 6))
        target[:, 1:] = label

        return torch.from_numpy(image), target


class ResizeImage(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        self.new_size = new_size #  (w, h)
        self.interpolation = interpolation

    def __call__(self, img, label):
        image = img
        image = cv2.resize(image, (self.new_size,self.new_size), interpolation=self.interpolation)
        return image, label

class ImageBaseAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                # horisontal_flip
                iaa.Fliplr(0.3),
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                # Add gaussian noise to some images.
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # Add a value of -5 to 5 to each pixel.
                sometimes(iaa.Add((-5, 5), per_channel=0.5)),
                # Change brightness of images (80-120% of original value).
                sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                # Improve or worsen the contrast of images.
                sometimes(iaa.LinearContrast((0.5, 2.0), per_channel=0.5)),
            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def __call__(self, img, label):
        seq_det = self.seq.to_deterministic()
        image, label = img, label
        image = seq_det.augment_images([image])[0]
        return image, label



class ImageTrans2Tensor(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        self.img_size = new_size
        self.interpolation = interpolation
    def __call__(self, img):
        image = img
        image = cv2.resize(image, (self.img_size,self.img_size), interpolation=self.interpolation)
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        return torch.from_numpy(image)