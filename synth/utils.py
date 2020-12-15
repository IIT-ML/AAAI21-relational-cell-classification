#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate the synthetic data

import pickle
from collections import Counter
import numpy as np
import cv2


GRID_SIZE = 25
CIRC_SIZE = 27
RADIUS = 10

Imag_SIZE = GRID_SIZE * CIRC_SIZE
VARIANCE = 0.05


def get_neibs_cds(img, l, u, dist=27, rads=27):

    candids = [(u-27, l-27), (u-27, l), (u-27, l+27), (u, l+27),
               (u+27, l+27), (u+27, l), (u+27, l-27), (u, l-27)]
    rst = []

    # Append neighbors
    for row, col in candids:
        rst.append(img[row:row+rads, col:col+rads])

    # Append itself
    rst.append(img[u:u+rads, l:l+rads])
    rst = np.array(rst)

    return rst


def load_pickle(indices, image_data):
    """"
    0: Empty
    1: Active
    2: Inactive
    """
    size = 13

    # image_data = "./data/images.pkl"
    with open(image_data, "rb") as f:
        images = pickle.load(f)

    x = []
    y = []
    n = []
    cds = []

    for idx in indices:

        D_dict = images[idx]

        img = D_dict['image']
        label = D_dict['label']

        row, col = label.shape
        length, width = img.shape

        img = np.expand_dims(img, axis=-1)

        img_r, img_c = 40, 40

        for g_r in range(1,  row-1):
            img_c = 40
            for g_c in range(1, col-1):

                # Check whether it's empty
                if label[g_r][g_c] == 0.0:
                    pass
                else:
                    l = img_c - size
                    u = img_r - size
                    r = img_c + size + 1
                    d = img_r + size + 1

                    pt = img[u:d, l:r]
                    nb = get_neibs_cds(img, l, u)
                    lb = label[g_r][g_c]

                    x.append(pt)
                    y.append(lb)
                    n.append(nb)
                    cds.append((img_r, img_c))

                img_c += 27
            img_r += 27

    x = np.array(x)
    y = np.array(y)
    n = np.array(n)

    return x, y, n, cds


def load_dict(dir_1, dir_2, dir_3, dir_4, image_data):

    record = {}
    record['folder'] = {}
    record['images'] = {}

    record['folder']['D1'] = dir_1
    record['folder']['D2'] = dir_2
    record['folder']['D3'] = dir_3
    record['folder']['D4'] = dir_4

    record['images']['D1'] = {}
    record['images']['D2'] = {}
    record['images']['D3'] = {}
    record['images']['D4'] = {}

    def fill_dict(name, type):

        x, y, n, cds = load_pickle(record['folder'][name], image_data)

        idx = 0

        for node, label, neib in zip(x, y, n):

            record['images'][name][idx] = {}
            record['images'][name][idx]['node'] = node

            if label == 1.0:

                record['images'][name][idx]['labl'] = 1

            elif label == 2.0:

                record['images'][name][idx]['labl'] = 0

            else:

                raise ValueError("AAA")

            record['images'][name][idx]['neib'] = neib
            idx += 1

    fill_dict('D1', 'pickle')
    fill_dict('D2', 'pickle')
    fill_dict('D3', 'pickle')
    fill_dict('D4', 'pickle')

    return record


def load_generate(image_data, indice_dict):
    """
    main entry to generate, load and pickle the data
    seed_num: seed number
    num: number of labeled images
    """
    with open(image_data, "rb") as f:
        images = pickle.load(f)

    D1 = indice_dict['D1']
    D2 = indice_dict['D2']
    D3 = indice_dict['D3']
    D4 = indice_dict['D4']

    data = load_dict(D1, D2, D3, D4, image_data)

    return data


if __name__ == "__main__":

    pass
