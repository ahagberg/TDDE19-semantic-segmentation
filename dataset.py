
import os
import random
import itertools
import cv2
import numpy as np

from fs_utils import get_img_annot_pairs_from_paths
from img_utils import colorize_annot

from keras.utils import Sequence

class DataSequence(Sequence):

    def __init__(self, dataset, img_path, annot_path, class_weights=None):
        self.dataset = dataset
        self.img_annot = get_img_annot_pairs_from_paths(img_path, annot_path)
        self.class_weights = class_weights

    def __len__(self):
        return int(np.ceil(len(self.img_annot) / float(self.dataset.batch_size)))

    def __getitem__(self, i):
        batch = self.img_annot[i*self.dataset.batch_size : (i+1) * self.dataset.batch_size]
        X = []
        Y = []
        for imgpath, annotpath in batch:
            img = cv2.imread(imgpath, 1)
            img = self.dataset.resize_image(img)
            img = self.dataset.normalize_image(img)
            annot = cv2.imread(annotpath, 1)
            annot = self.dataset.process_annot(annot)
            X.append(img)
            Y.append(annot)
        if self.class_weights is None:
            return np.array(X), np.array(Y)
        else:
            return np.array(X), np.array(Y) * np.array(self.class_weights)

class Dataset():

    def __init__(self, name, batch_size, n_classes, width, height, 
                mean=0.0, std=1.0, class_weights=None):
        self.name = name
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.width = width
        self.height = height
        self.mean = mean
        self.std = std

        self.train = DataSequence(dataset=self,
            img_path=os.path.join('./datasets', self.name, "train"),
            annot_path=os.path.join('./datasets', self.name, "trainannot"), 
            class_weights=class_weights)

        self.val = DataSequence(dataset=self,
            img_path=os.path.join('./datasets', self.name, "val"),
            annot_path=os.path.join('./datasets', self.name, "valannot"))

        self.test = DataSequence(dataset=self,
            img_path=os.path.join('./datasets', self.name, "test"),
            annot_path=os.path.join('./datasets', self.name, "testannot"))

    def resize_image(self, img):
        # Resize image
        img = cv2.resize(img, (self.width , self.height))
        # Convert to float
        img = img.astype(np.float32)
        img /= 255
        return img

    def normalize_image(self, img):
        return (img - self.mean) / self.std

    def un_normalize_image(self, img):
        return img*self.std + self.mean

    def process_annot(self, annot):
        # The last channel is a mask, showing class membership
        labels = np.zeros((self.height, self.width, self.n_classes))
        # Resize without interpolation
        annot = cv2.resize(annot, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        # Grab first channel
        annot = annot[:, : , 0]
        # Fill in mask for each label
        for c in range(self.n_classes):
            labels[:, :, c] = (annot == c).astype(np.int)
        return labels

class CamVid(Dataset):
    def __init__(self, batch_size, normalize=False, class_weights=None):
        mean = [0.4315961, 0.4240506, 0.4107813] if normalize else 0.0
        std = [0.29778674, 0.3015305, 0.29543346] if normalize else 1.0
        super().__init__("CamVid", batch_size, 12, 224, 224, mean, std, class_weights)
        # classes=['sky', 'house', 'poles', 'road', 'sidewalk', 'trees', 'signs', 'fence', 'car', 'people', 'bicycle', 'none']

class Kitti(Dataset):
    def __init__(self, batch_size, normalize=False, class_weights=None):
        mean = [0.40353236, 0.41684213, 0.39147156] if normalize else 0.0
        std = [0.33218956, 0.3194578, 0.30435866] if normalize else 1.0
        super().__init__("kitti", batch_size, 29, 224, 224, mean, std, class_weights)
        #super().__init__("kitti", batch_size, 29, 1242, 375)

if __name__ == '__main__':

    dataset = CamVid(batch_size=3, normalize=True)

    for imgbatch, annotbatch in dataset.test:
        for img, annot in zip(imgbatch, annotbatch):
            cv2.imshow("img", img)
            cv2.imshow("annot", colorize_annot(annot))
            cv2.imshow("un normalized", dataset.un_normalize_image(img))
            cv2.waitKey(0)
