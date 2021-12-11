import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Dropout, MaxPool2D, add
from keras.layers.convolutional import Conv2D, Deconv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.utils import plot_model
from net import Net

class FCN(Net):

    def __init__(self, name='fcn', dataset=None):
        super().__init__(name, dataset)

    def create_model(self):
        rate = 0.2
        fcn_8 = Sequential()
        fcn_8.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (self.dataset.height, self.dataset.width, 3)))
        fcn_8.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(MaxPool2D(pool_size=2, name ="Pool1"))

        fcn_8.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(MaxPool2D(pool_size=2, name ="Pool2"))

        fcn_8.add(Conv2D(256, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(Conv2D(256, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(Conv2D(256, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(MaxPool2D(pool_size=2, name ="Pool3"))

        skip_con2 = Conv2D(self.dataset.n_classes, 1, padding = 'same', activation = None, name = 'Predict3')

        fcn_8.add(Conv2D(512, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(Conv2D(512, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(Conv2D(512, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(MaxPool2D(pool_size=2, name ="Pool4"))

        skip_con1 = Conv2D(self.dataset.n_classes, 1, padding = 'same', activation = None, name = 'Predict2')

        fcn_8.add(Conv2D(512, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(Conv2D(512, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(Conv2D(512, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(Dropout(rate))
        fcn_8.add(MaxPool2D(pool_size=2, name ="Pool5"))

        fcn_8.add(Conv2D(4096, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(Conv2D(4096, kernel_size = 3, padding = 'same', activation = 'relu'))
        fcn_8.add(Dropout(rate))

        fcn_8.add(Conv2D(self.dataset.n_classes, 1, padding = 'same', activation = 'relu', name = 'Predict1'))
        fcn_8.add(Deconv2D(self.dataset.n_classes, 4, strides = 2, padding = 'same', activation = None, name = 'Deconv1'))

        summed = add(inputs = [skip_con1(fcn_8.layers[14].output), fcn_8.layers[-1].output])

        deconv2 = Deconv2D(self.dataset.n_classes, kernel_size = 4, strides = 2, padding = 'same', activation = None, name = 'Deconv2')(summed)

        summed2 = add(inputs = [skip_con2(fcn_8.layers[10].output), deconv2])

        output = Deconv2D(self.dataset.n_classes, kernel_size = 16, strides = 8, padding = 'same', activation = 'softmax', name = 'Deconv3')(summed2)

        self.model = Model(fcn_8.input, output)
