
import scikitplot as skplt
import crf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, \
                            accuracy_score
from meanIoU import mean_iou, conf_mean_iou
import keras.backend as K
import tensorflow as tf

def evaluate_metrics(net, crf_iter):

    # Predict on all test data
    Y_pred = net.model.predict_generator(net.dataset.test, verbose=1)
    Y_true = np.concatenate([annot for img, annot in net.dataset.test])
    images = np.concatenate([img for img, annot in net.dataset.test])
    labels = list(range(net.dataset.n_classes))
    if crf_iter > 0:
        # TODO: CRF post processing
        for i in range(Y_pred.shape[0]):
            Y_pred[i] = crf.dense_crf(images[i], Y_pred[i], crf_iter)
        pass
    y_pred = np.argmax(Y_pred, axis=3).flatten()
    y_true = np.argmax(Y_true, axis=3).flatten()

    

    # Confusion matrix
    print('Generating Confusion Matrix...')
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt("confusion_%s_%s.csv" % (net.name, net.dataset.name), conf, fmt='%i', delimiter=",")
    skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
    plt.show()

    # Classification report
    print('Classification Report')
    print(classification_report(y_true, y_pred, labels=labels))

    # Accuracy score
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy on test data: ' + str(round(accuracy*100, 1)) + '%')
    m_iou = conf_mean_iou(conf)
    print('Mean intersecion over union on test data: ' + str(round(m_iou*100, 1)))
