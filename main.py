
###########################################
###       Parse program arguments      ####
###########################################
import argparse

booltype = lambda s: s in ["yes", "1", "true", "True"]

parser = argparse.ArgumentParser(description='Train and evaluate segmentation networks.')
parser.add_argument('--net', default='segnet',
                    help='Network architecture.')
parser.add_argument('--dataset', default='camvid',
                    help='Dataset to train and test on.')
parser.add_argument('--normalize', default=True, type=booltype,
                    help='Should the dataset be normalized based on mean and std of train set.')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate.')
parser.add_argument('--batch', default=3, type=int,
                    help='Batch size.')
parser.add_argument('--epochs', default=1, type=int,
                    help='Number of epochs.')
parser.add_argument('--optimizer', default='adam',
                    help='The optimizer to use.')
parser.add_argument('--plot', default=False, type=booltype,
                    help='Plot the model.')
parser.add_argument('--crf_iter', default=0, type=int,
                    help='Number of iterations of CRF')
parser.add_argument('--eval', default=False, type=booltype,
                    help='Eval results.')
parser.add_argument('--weighting', default=False, type=booltype,
                    help='Whether to use class weighting during training')
parser.add_argument('--patience', default=10, type=int,
                    help='The patience used for early stopping')
parser.add_argument('--decay', default='',
                    help='The decay type.')
parser.add_argument('--decayFactor', default=0.25, type=float,
                    help='The factor for which we decay every decay interval of epochs.')
parser.add_argument('--decayInterval', default=10, type=int,
                    help='At what epoch interval we decay by the decay factor')
args = parser.parse_args()
print('\nCurrent arguments -> ', args, '\n')

###########################################
###               Imports              ####
###########################################

import numpy as np
from keras.optimizers import Adam
from dataset import CamVid, Kitti
from segnet0 import Segnet
from segnet import Segnet as SegnetOld
from unet import Unet
from fcn import FCN
from decay import StepDecay
from img_utils import colorize_annot
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from sklearn.utils import class_weight
import keras
import cv2
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pylab
import crf
from meanIoU import mean_iou

import tensorflow as tf

import scikitplot as skplt
from keras import optimizers

from plot_over_time import plot_over_time
from evaluate_metrics import evaluate_metrics


np.random.seed(1337)

class_weights = None


###########################################
###    Create class weighting vecor    ####
###########################################
if args.weighting:
    try:
        class_weights = np.load(args.dataset + "_weighting.npy")
    except FileNotFoundError:
        print("Generating weighting vector...")
        if args.dataset == 'camvid':
            dataset = CamVid(batch_size=args.batch, class_weights=class_weights)
        elif args.dataset == 'kitti':
            dataset = Kitti(batch_size=args.batch, class_weights=class_weights)
        else:
            print('Dataset not supported: %s' % args.dataset)
            exit()
        for i in range(len(dataset.train)):
            batch = dataset.train.__getitem__(i)[1]
            if i == 0:
                train_data = batch
            else:
                print(train_data.shape)
                train_data = np.concatenate((train_data, batch))
        c = np.argmax(train_data, axis=3).flatten()
        print(set(list(c)))
        classes = [c for c in range(dataset.n_classes)]
        print(classes)
        nonexistent = []
        for class_ in classes:
            if class_ not in c:
                 nonexistent.append(class_)
        print(nonexistent)
        for i in nonexistent:
            print(i)
            classes.remove(i)
        class_weights = class_weight.compute_class_weight('balanced',
                classes, c)
        print(nonexistent)

        for i in nonexistent:
            class_weights = np.insert(class_weights, i, 1)
        print(class_weights)
        print(class_weights.shape)
        np.save(args.dataset + "_weighting.npy", class_weights)



###########################################
###      Create dataset generators      ####
###########################################
if args.dataset == 'camvid':
    dataset = CamVid(batch_size=args.batch, normalize=args.normalize, class_weights=class_weights)
elif args.dataset == 'kitti':
    dataset = Kitti(batch_size=args.batch, normalize=args.normalize, class_weights=class_weights)
else:
    print('Dataset not supported: %s' % args.dataset)
    exit()

###########################################
###     Select network architecture    ####
###########################################
if args.net == 'segnet':
    net = Segnet(dataset=dataset)
elif args.net == 'segnetOld':
    net = SegnetOld(dataset=dataset)
elif args.net == 'unet':
    net = Unet(dataset=dataset)
elif args.net == 'fcn':
    net = FCN(dataset=dataset)
else:
    print('Network architecture not implemented: %s' % args.net)
    exit()

###########################################
###     Select network architecture    ####
###########################################
decay = None
if args.decay == 'step':
    decay = StepDecay(initAlpha=args.lr, factor=args.decayFactor, dropEvery=args.decayInterval)
elif args.decay == '':
    pass
else:
    print('Decay type not implemented: %s' % args.decay)
    exit()


if decay is not None:
    decay.plot(args.epochs)

###########################################
###     Load or create and train net   ####
###########################################
try:
    if net.load_model():
        result = None
    else:
        print('\nNo model on disk, will create a new one!\n')

        # Create model and compile it
        net.create_model()

        net.model.summary()

        if args.optimizer == 'adam':
            optimizer = Adam(lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = optimizers.SGD(lr=args.lr, momentum=0.9)
        else:
            print('Optimizer not supported: %s' % args.optimizer)
            exit()
        print("compiling model")
        net.model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy,
            metrics=[categorical_accuracy, mean_iou]
        )
        print("model compiled")

        callbacks = []

        # Early Stopping
        es = EarlyStopping(monitor='val_loss',
                mode='min', verbose=1, patience=args.patience,
                min_delta=0.01,restore_best_weights=True)
        callbacks.append(es)

        if decay is not None:
            callbacks.append(decay)

        # Train the model
        print("Training model: \n")
        result = net.model.fit_generator(
            dataset.train,
            epochs=args.epochs,
            validation_data=dataset.val,
            validation_steps=1,
            callbacks=callbacks,
        )


        # Save trained model for next time
        net.save_model()

    if args.plot:
        plot_model(net.model, show_shapes=True)

    ###########################################
    ###          Evaluate accuracy!        ####
    ###########################################

    if result:
        plot_over_time(result, args.net)

    if args.eval:
        evaluate_metrics(net, args.crf_iter)

except KeyboardInterrupt:
    exit()

imgs = []
annots = []
preds = []

for imgbatch, annotbatch in dataset.test:
    if len(imgs):
        cv2.imshow("Batch",
            np.vstack((
                np.hstack(imgs),
                np.hstack(preds),
                np.hstack(annots),
            ))
        )
        imgs = []
        annots = []
        preds = []

    for img, annot in zip(imgbatch, annotbatch):

        prediction = net.model.predict(
            img[np.newaxis,...]
        )[0]
        if args.crf_iter > 0:
            crf_pred = crf.dense_crf(img, prediction, args.crf_iter)
            cv2.imshow("prediction crf", colorize_annot(crf_pred))

        un_normalized_img = dataset.un_normalize_image(img)
        colorized_annot = colorize_annot(annot)
        colorized_pred = colorize_annot(prediction)

        cv2.imshow("image", img)
        cv2.imshow("un normalized", un_normalized_img)
        cv2.imshow("annotation", colorized_annot)
        cv2.imshow("prediction", colorized_pred)

        imgs.append(un_normalized_img)
        annots.append(colorized_annot)
        preds.append(colorized_pred)

        # Terminates program if escape key is pressed
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            exit()
