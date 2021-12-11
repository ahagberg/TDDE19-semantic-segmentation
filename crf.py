import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 100
Bi_RGB_STD = 3


def dense_crf(img, output_probs, iterations):
    img = (255 * img).astype(np.uint8)
    output_probs = np.transpose(output_probs, (2, 0, 1))
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(1, 3)
    d.addPairwiseBilateral(Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(iterations)
    Q = np.array(Q).reshape((c, h, w))
    return np.transpose(Q, (1, 2, 0))