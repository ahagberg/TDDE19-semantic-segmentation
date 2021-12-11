
import cv2
import numpy as np
import colorsys

def colorize_annot(annot):
    labels = np.argmax(annot, -1)
    res = np.zeros((labels.shape[0], labels.shape[1], 3))
    n = annot.shape[2]
    for label in range(n):
        res[labels==label] = pretty_color(label, n)
    return res

def pretty_color(i, n):
    rgb = colorsys.hsv_to_rgb(((i*2)/(n-1)) % n, 0.8, 1.0)
    return (rgb[2], rgb[1], rgb[0])
