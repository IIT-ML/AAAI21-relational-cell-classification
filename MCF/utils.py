#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Load and generate the data

import os
import glob
import pickle
import numpy as np
import matplotlib.image as mpimg


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


def load_pickle(png_files):

    size = 13
    row, col = 512, 640

    folder_prefix = "./MCF-7/"

    label_details = "./data/record.pkl"
    with open(label_details, 'rb') as handle:
        record = pickle.load(handle)

    x = []
    y = []
    n = []

    for png in png_files:

        png_full_path = folder_prefix + png
        img = mpimg.imread(png_full_path)
        img = np.array(img)

        details = record[png]

        for idx, detail in details.items():

            yy = detail['y']
            xx = detail['x']

            yy, xx = int(yy), int(xx)

            u, d = yy-size, yy+size+1
            l, r = xx-size, xx+size+1

            if u < 0:
                d = d + abs(u)
                u = 0

            if d > row:
                u = u - abs(d-row)
                d = row

            if l < 0:
                r = r + abs(l)
                l = 0

            if r > col:
                l = l - abs(r-col)
                r = col

            if l < 27 or u < 27 or d > (row-27) or r > (col-27):
                continue

            patch = img[u:d, l:r]

            r, c, d = patch.shape

            if r != 27 or c != 27:
                raise("error")

            neibs = get_neibs_cds(img, l, u)
            x.append(patch)
            y.append(detail['label_name'])
            n.append(neibs)

    x = np.array(x)
    y = np.array(y)
    n = np.array(n)

    return x, y, n


def load_dict(dir_1, dir_2, dir_3, dir_4):

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

    def fill_dict(name):

        x, y, n = load_pickle(record['folder'][name])

        idx = 0

        for node, label, neib in zip(x, y, n):

            record['images'][name][idx] = {}
            record['images'][name][idx]['node'] = node
            record['images'][name][idx]['labl'] = label
            record['images'][name][idx]['neib'] = neib
            idx += 1

    fill_dict('D1')
    fill_dict('D2')
    fill_dict('D3')
    fill_dict('D4')

    return record


def load_generate(indice_dict):
    """
    main entry to generate, load and pickle the data 
    indice_dict: splitting file
    """
    # D1 for dev, D2 for train, D3 for unsupervised train, D4 for test
    D1 = indice_dict['D1']
    D2 = indice_dict['D2']
    D3 = indice_dict['D3']
    D4 = indice_dict['D4']

    data = load_dict(D1, D2, D3, D4)

    return data
