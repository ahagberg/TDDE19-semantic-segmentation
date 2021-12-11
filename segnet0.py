

# python main.py --net segnet --dataset camvid --lr 0.0001 --batch 10 --epochs 60 --plot yes --crf_iter 0 --eval yes --patience 5 --decay step --decayFactor 0.7 --decayInterval 3
# loss: 0.6166 - categorical_accuracy: 0.9485 - val_loss: 0.8123 - val_categorical_accuracy: 0.8750

# python main.py --net segnet --dataset camvid --lr 0.0001 --batch 10 --epochs 60 --plot yes --crf_iter 0 --eval yes --patience 5 --decay step --decayFactor 0.7 --decayInterval 3 --weighting yes
# loss: 0.5420 - categorical_accuracy: 0.9554 - val_loss: 0.7981 - val_categorical_accuracy: 0.8800
# 76.6%

# python main.py --net segnet --dataset camvid --lr 0.0001 --batch 10 --epochs 30 --plot yes --crf_iter 0 --eval yes --patience 4 --decay step --decayFactor 0.25 --decayInterval 3 --weighting yes
# 77.5%

import keras, os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, BatchNormalization, Activation, UpSampling2D, Layer, Input
import numpy as np
from net import Net
from keras.models import Model

from custom_layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

class Segnet(Net):

    def __init__(self, name='segnet', dataset=None):
        super().__init__(name, dataset)

    def create_model(self):
        input = Input(shape = (self.dataset.height, self.dataset.width, 3))
        conv1 = Conv2D(filters=64,kernel_size=(3,3),padding="same")(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv2 = Conv2D(filters=64,kernel_size=(3,3),padding="same")(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        pool1, indices1 = MaxPoolingWithArgmax2D(pool_size=(2,2),strides=(2,2))(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(3,3), padding="same")(pool1)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv4 = Conv2D(filters=128, kernel_size=(3,3), padding="same")(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        pool2, indices2 = MaxPoolingWithArgmax2D(pool_size=(2,2),strides=(2,2))(conv4)

        conv5 = Conv2D(filters=256, kernel_size=(3,3), padding="same")(pool2)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        conv6 = Conv2D(filters=256, kernel_size=(3,3), padding="same")(conv5)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation("relu")(conv6)
        conv7 = Conv2D(filters=256, kernel_size=(3,3), padding="same")(conv6)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation("relu")(conv7)
        pool3, indices3 = MaxPoolingWithArgmax2D(pool_size=(2,2),strides=(2,2))(conv7)

        conv8 = Conv2D(filters=512, kernel_size=(3,3), padding="same")(pool3)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation("relu")(conv8)
        conv9 = Conv2D(filters=512, kernel_size=(3,3), padding="same")(conv8)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation("relu")(conv9)
        conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same")(conv9)
        conv10 = BatchNormalization()(conv10)
        conv10 = Activation("relu")(conv10)
        pool4, indices4 = MaxPoolingWithArgmax2D(pool_size=(2,2),strides=(2,2))(conv10)

        conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same")(pool4)
        conv11 = BatchNormalization()(conv11)
        conv11 = Activation("relu")(conv11)
        conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same")(conv11)
        conv12 = BatchNormalization()(conv12)
        conv12 = Activation("relu")(conv12)
        conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same")(conv12)
        conv13 = BatchNormalization()(conv13)
        conv13 = Activation("relu")(conv13)
        pool5, indices5 = MaxPoolingWithArgmax2D(pool_size=(2,2),strides=(2,2))(conv13)

        up1 = MaxUnpooling2D(size=(2,2))([pool5,indices5])
        conv14 = Conv2D(filters=512, kernel_size=(3,3), padding="same", data_format="channels_last")(up1)
        conv14 = BatchNormalization()(conv14)
        conv14 = Activation("relu")(conv14)
        conv15 = Conv2D(filters=512, kernel_size=(3,3), padding="same", data_format="channels_last")(conv14)
        conv15 = BatchNormalization()(conv15)
        conv15 = Activation("relu")(conv15)
        conv16 = Conv2D(filters=512, kernel_size=(3,3), padding="same", data_format="channels_last")(conv15)
        conv16 = BatchNormalization()(conv16)
        conv16 = Activation("relu")(conv16)

        up2 = MaxUnpooling2D(size=(2,2))([conv16,indices4])
        conv17 = Conv2D(filters=512, kernel_size=(3,3), padding="same", data_format="channels_last")(up2)
        conv17 = BatchNormalization()(conv17)
        conv17 = Activation("relu")(conv17)
        conv18 = Conv2D(filters=512, kernel_size=(3,3), padding="same", data_format="channels_last")(conv17)
        conv18 = BatchNormalization()(conv18)
        conv18 = Activation("relu")(conv18)
        conv19 = Conv2D(filters=256, kernel_size=(3,3), padding="same", data_format="channels_last")(conv18)
        conv19 = BatchNormalization()(conv19)
        conv19 = Activation("relu")(conv19)

        up3 = MaxUnpooling2D(size=(2,2))([conv19,indices3])
        conv20 = Conv2D(filters=256, kernel_size=(3,3), padding="same", data_format="channels_last")(up3)
        conv20 = BatchNormalization()(conv20)
        conv20 = Activation("relu")(conv20)
        conv21 = Conv2D(filters=256, kernel_size=(3,3), padding="same", data_format="channels_last")(conv20)
        conv21 = BatchNormalization()(conv21)
        conv21 = Activation("relu")(conv21)
        conv22 = Conv2D(filters=128, kernel_size=(3,3), padding="same", data_format="channels_last")(conv21)
        conv22 = BatchNormalization()(conv22)
        conv22 = Activation("relu")(conv22)

        up4 = MaxUnpooling2D(size=(2,2))([conv22,indices2])
        conv23 = Conv2D(filters=128, kernel_size=(3,3), padding="same", data_format="channels_last")(up4)
        conv23 = BatchNormalization()(conv23)
        conv23 = Activation("relu")(conv23)
        conv24 = Conv2D(filters=64, kernel_size=(3,3), padding="same", data_format="channels_last")(conv23)
        conv24 = BatchNormalization()(conv24)
        conv24 = Activation("relu")(conv24)

        up5 = MaxUnpooling2D(size=(2,2))([conv24,indices1])
        conv25 = Conv2D(filters=64, kernel_size=(3,3), padding="same", data_format="channels_last")(up5)
        conv25 = BatchNormalization()(conv25)
        conv25 = Activation("relu")(conv25)
        conv26 = Conv2D(filters=self.dataset.n_classes, kernel_size=(1,1), padding="valid", data_format="channels_last")(conv25)
        conv26 = BatchNormalization()(conv26)
        conv26 = Activation("softmax")(conv26)

        self.model = Model(inputs=input, outputs=conv26, name=self.name)
