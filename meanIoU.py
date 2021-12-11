import keras.backend as K
import tensorflow as tf
import numpy as np
from dataset import CamVid


def mean_iou(y_true, y_pred):
    y_pred = K.one_hot(K.argmax(y_pred), y_pred.shape[-1])
    y_true = K.sign(y_true)
    intersection = K.sum(y_true * y_pred, axis=[0,1,2])
    union = K.sum(y_true,axis=[0,1,2]) + K.sum(y_pred,axis=[0,1,2])
    union = union - intersection
    iou = (intersection + 1) / (union + 1)
    return K.mean(iou)

def conf_mean_iou(confusion):
    """
    Return the per class Intersection over Union (I/U) from confusion matrix.
    Args:
        confusion: the confusion matrix between ground truth and predictions
    Returns:
        a vector representing the per class I/U
    Reference:
        https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = (confusion * np.eye(len(confusion))).sum(axis=-1)
    preds = confusion.sum(axis=0)
    trues = confusion.sum(axis=-1)
    union = trues + preds - intersection
    return np.mean((intersection + 1) / (union + 1))


if __name__ == '__main__':
    dataset = CamVid(1)
    batch = dataset.train.__getitem__(1)[1]
    t = mean_iou(batch, batch)
    print(K.eval(t))
