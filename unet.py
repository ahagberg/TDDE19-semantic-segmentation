import keras, os
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, BatchNormalization, Activation, UpSampling2D, Layer, Input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras.layers.merge import Concatenate
from net import Net

class Unet(Net):

    def __init__(self, name='unet', dataset=None):
        super().__init__(name, dataset)
        self.model = Model(inputs=[], outputs=[])

    # Block in down part of model
    def down_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1, bn=True):
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        if bn: x = BatchNormalization()(c)
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        if bn: x = BatchNormalization()(c)
        p = MaxPool2D((2, 2), (2, 2))(c)
        return c, p

    # Block in up part of model
    def up_block(self, x, skip, filters, kernel_size=(3, 3), padding="same", strides=1, bn=True):
        us = UpSampling2D((2, 2))(x)
        concat = Concatenate()([us, skip])
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
        if bn: x = BatchNormalization()(c)
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        if bn: x = BatchNormalization()(c)
        return c

    # Block furthest down in model
    def bottleneck(self, x, filters, kernel_size=(3, 3), padding="same", strides=1, bn=True):
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        if bn: x = BatchNormalization()(c)
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        if bn: x = BatchNormalization()(c)
        return c

    # Create model using blocks in the class
    def create_model(self):
        print("creating Unet model!")
        n = 16 #Nr. of filters
        #f = [16, 32, 64, 128, 256]
        f = [1*n, 2*n, 4*n, 8*n, 16*n]
        inputs = Input((self.dataset.height, self.dataset.width, 3))

        #Encoder
        c1, p1 = self.down_block(inputs, f[0])
        c2, p2 = self.down_block(p1, f[1])
        c3, p3 = self.down_block(p2, f[2])
        c4, p4 = self.down_block(p3, f[3])

        #Bottleneck
        bn = self.bottleneck(p4, f[4])

        #Decoder
        u1 = self.up_block(bn, c4, f[3])
        u2 = self.up_block(u1, c3, f[2])
        u3 = self.up_block(u2, c2, f[1])
        u4 = self.up_block(u3, c1, f[0])

        outputs = keras.layers.Conv2D(filters=self.dataset.n_classes, kernel_size=(1, 1), padding="valid", activation="softmax")(u4)

        model = keras.models.Model(inputs, outputs)

        self.model = model
