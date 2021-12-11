
from dataset import CamVid, Kitti
import argparse
import numpy as np
import cv2
from img_utils import colorize_annot
import matplotlib.colors


parser = argparse.ArgumentParser(description='Calculate mean and std of train data.')
parser.add_argument('--dataset', default='camvid',
                    help='The dataset to use.')
parser.add_argument('--batch', default=3, type=int,
                    help='Batch size.')
args = parser.parse_args()
print('\nCurrent arguments -> ', args, '\n')

if args.dataset == 'camvid':
    dataset = CamVid(batch_size=args.batch, normalize=False)
elif args.dataset == 'kitti':
    dataset = Kitti(batch_size=args.batch, normalize=False)
else:
    print('Dataset not supported: %s' % args.dataset)
    exit()

N = 0
cummean = 0
cumstd = 0

for imgbatch, annotbatch in dataset.train:
    cummean += np.mean(imgbatch, axis=(0,1,2))
    cumstd += np.std(imgbatch, axis=(0,1,2))
    N += 1

mean = cummean / N
std = cumstd / N

print("Mean:", ", ".join([str(e) for e in mean]))
print("Std:", ", ".join([str(e) for e in std]))