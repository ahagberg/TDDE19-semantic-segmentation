
from keras.models import Sequential
from keras.models import load_model
from custom_layers import custom_objects
from meanIoU import mean_iou

class Net:

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.model = Sequential()

    def load_model(self):
        try:
            file = '%s-%s.h5' % (self.name, self.dataset.name)
            self.model = load_model(file,
                custom_objects=custom_objects)
            print('\nLoaded model from disk: %s\n' % file)
            return True
        except Exception as e:
            print("Could not load model due to reason: %s" % str(e))
            return False

    def save_model(self):
        file = '%s-%s.h5' % (self.name, self.dataset.name)
        self.model.save(file, include_optimizer=False)
        print('Saved model to: %s' % file)
