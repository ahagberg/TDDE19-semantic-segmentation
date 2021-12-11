import keras, os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, BatchNormalization, Activation, UpSampling2D, Layer, Input
import numpy as np
from net import Net

class Segnet(Net):

    def __init__(self, name='segnet-old', dataset=None):
        super().__init__(name, dataset)

    def add_conv_batch_relu(self, input_shape=None, filters=None, kernel_size=(3,3), padding="same"):
        if input_shape is None:
            self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding=padding))
        else:
            self.model.add(Conv2D(input_shape=input_shape, filters=filters, kernel_size=kernel_size, padding=padding))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))

    def add_maxpool(self, pool_size=(2,2), strides=(2,2)):
        self.model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    def add_encoder(self, layers=None, filters=None, input_shape=None):
        for i in range(layers):
            self.add_conv_batch_relu(filters=filters, input_shape=input_shape if i == 0 else None)
        self.add_maxpool()

    def create_model(self):
        self.model = Sequential()

        self.add_encoder(layers=2, filters= 64, input_shape=(self.dataset.height,self.dataset.width,3))
        self.add_encoder(layers=2, filters=128)
        self.add_encoder(layers=3, filters=256)
        self.add_encoder(layers=3, filters=512)
        self.add_encoder(layers=3, filters=512)

        self.model.add(UpSampling2D(size=(2,2)))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))

        self.model.add(UpSampling2D(size=(2,2)))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))

        self.model.add(UpSampling2D(size=(2,2)))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))

        self.model.add(UpSampling2D(size=(2,2)))
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))

        self.model.add(UpSampling2D(size=(2,2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters=12, kernel_size=(1,1), padding="valid", data_format="channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("softmax"))
