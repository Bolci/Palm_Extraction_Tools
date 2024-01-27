from osgeo import gdal
import numpy as np
import mmcv
from copy import copy
import cv2


def get_rsi_image(image_path):
    data = gdal.Open(image_path)
    band = data.ReadAsArray().astype(np.float32)
    # print(band)
    return band.transpose((1, 2, 0))


def load_rgb_image(image_path):
    img = mmcv.imread(image_path)
    return img


def get_printable_rsi(img):
    return img[:, :, (4, 2, 1)].astype(np.uint8)


def merge_img_mask(img, mask, alpha):
    img = copy(img)
    k, j, i = np.where(mask == 255)

    for id_x in range(len(k)):
        img[k[id_x], j[id_x], :] = np.array([255.0, 0.0, 0.0])
    return np.asarray(img).astype(np.uint8)


def colour_img_new(img):
    img = copy(img)
    img = img.astype(np.uint8)

    unique_objects = np.unique(img)
    unique_objects = unique_objects[unique_objects != 0]

    img_colours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    new_img = np.zeros(img_colours.shape).astype(np.uint8)

    for x in unique_objects:
        j, i = np.where(img == x)
        peaks_coordinates = np.concatenate((j.reshape((1,) + j.shape), i.reshape((1,) + i.shape)), axis=0).T

        colours = []
        for _ in range(3):
            colours.append(np.random.randint(0, 255))

        for coordinates in peaks_coordinates:
            new_img[coordinates[0], coordinates[1], :] = colours

    return np.asarray(new_img)


