#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Load and generate the data

import os
import pickle

import numpy as np


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


def load_pickle(folders, aug_list=['org', '090', '180', '270', 'ver', 'hor']):
    """"
    Others: class 0
    Epithelial: class 1
    Fibroblast: class 2
    Inflammatory: class 3
    """
    size = 13
    types = ['others', 'epithelial', 'fibroblast', 'inflammatory']
    class_dict = {key: i for i, key in enumerate(types)}

    folder_prefix = "./CRCHistoPhenotypes_2016_04_28/Classification/"

    x = []
    y = []
    cds = []
    n = []

    for f in folders:

        folder_path = folder_prefix + f + '/'
        pkl_file = folder_path + f + '.pkl'

        with open(pkl_file, 'rb') as handle:
            D_dict = pickle.load(handle)

        for aug in aug_list:

            img = D_dict[aug]['image']

            for tp in types:

                mat = D_dict[aug][tp]

                for xx, yy in mat:

                    yy, xx = int(yy), int(xx)

                    u, d = yy-size, yy+size+1
                    l, r = xx-size, xx+size+1

                    if u < 0:
                        d = d + abs(u)
                        u = 0

                    if d > 500:
                        u = u - abs(d-500)
                        d = 500

                    if l < 0:
                        r = r + abs(l)
                        l = 0

                    if r > 500:
                        l = l - abs(r-500)
                        r = 500

                    if l < 27 or u < 27 or d > 473 or r > 473:

                        continue

                    patch = img[u:d, l:r]

                    row, col, depth = patch.shape

                    if row != 27 or col != 27:

                        raise("error")

                    neibs = get_neibs_cds(img, l, u)

                    x.append(patch)
                    y.append(class_dict[tp])
                    n.append(neibs)
                    cds.append(((d-u)//2, (r-l)//2))

    x = np.array(x)
    y = np.array(y)
    n = np.array(n)

    return x, y, n, cds


def load_dict(dir_1, dir_2, dir_3, dir_4, seednum, aug_list):

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

    def fill_dict(name, auglist):

        x, y, n, cds = load_pickle(record['folder'][name], auglist)

        idx = 0

        for node, label, neib in zip(x, y, n):

            record['images'][name][idx] = {}
            record['images'][name][idx]['node'] = node
            record['images'][name][idx]['labl'] = label
            record['images'][name][idx]['neib'] = neib
            idx += 1

    fill_dict('D1', aug_list)
    fill_dict('D2', aug_list)
    fill_dict('D3', aug_list)
    fill_dict('D4', aug_list)

#     pickle_name = './pickle/data_' + str(seednum) + '.pkl'

#     with open(pickle_name, 'wb') as handle:
#         pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return record


def load_generate(seed_num, indice_dict, aug_list):
    """
    main entry to generate, load and pickle the data 
    seed_num: seed number
    num: number of labeled images 
    """
    D1 = indice_dict['D1']
    D2 = indice_dict['D2']
    D3 = indice_dict['D3']
    D4 = indice_dict['D4']

    data = load_dict(D1, D2, D3, D4, seed_num, aug_list)

    return data
