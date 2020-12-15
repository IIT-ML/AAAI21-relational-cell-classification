import cv2
import pickle
import argparse
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.naive_bayes import GaussianNB

GRID_SIZE = 25
CIRC_SIZE = 27
RADIUS = 10

Imag_SIZE = GRID_SIZE * CIRC_SIZE


def sample_image(rst_label, rst_intst):
    """
    Sample one image given the label and intensity. Draw the circle on the patch.
    """
    row, col = rst_intst.shape

    image = np.zeros((Imag_SIZE, Imag_SIZE))

    s_r, s_c = 13, 13

    for r in range(row):

        s_c = 13

        for c in range(col):

            intst = rst_intst[r][c]
            label = rst_label[r][c]

            if label != 0:
                radius = RADIUS
                cv2.circle(image, (s_c, s_r), radius, (intst), -1)

            s_c += 27
        s_r += 27

    return image


def k_nearest_neighbor_distance(r, c, matrix, model, distance=1):
    """
    KNN classifier based on the neighbor given the distance. Didn't use softmax to get the prob.
    """
    target_intst = matrix[r][c]
    t_prob = model.predict_proba([[target_intst]])[0]

    row, col = matrix.shape

    left = max(0, c - distance)
    right = min(col - 1, c + distance)

    top = max(0, r - distance)
    down = min(row - 1, r + distance)

    sub_matrix = matrix[top:down + 1, left:right + 1]
    sub_r, sub_c = sub_matrix.shape

    score = [-t_prob[0], -t_prob[1]]

    count = 0

    for r_ in range(sub_r):
        for c_ in range(sub_c):

            if sub_matrix[r_][c_] == 0.0:
                continue

            else:
                prob = model.predict_proba([[sub_matrix[r_][c_]]])[0]
                score[0] += prob[0]
                score[1] += prob[1]

                count += 1

    if count == 0:
        print("No neighbor")

    if score[0] > score[1]:
        return 1

    else:
        return 2


def generation_sample(il, ih, al, ah, rate1=0.5, rate2=0.5):
    """
    Generate one sample in this function. 
    2 is inactive, 1 is active. 
    """

    x = []
    y = []

    indices = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            indices.append((r, c))

    np.random.shuffle(indices)

    grid_label = np.zeros((GRID_SIZE, GRID_SIZE))
    grid_intst = np.zeros((GRID_SIZE, GRID_SIZE))

    non_e_coords = []

    # Step 1
    for r, c in indices:

        # Empty or not
        roll_1 = np.random.rand(1)[0]
        if roll_1 < rate1:
            # Inactive or active
            roll_2 = np.random.rand(1)[0]

            if roll_2 < rate2:
                # Inactive
                low, high = il, ih
                label = 2
            else:
                # Active
                low, high = al, ah
                label = 1

            intensity = np.random.uniform(low, high)
            x.append(intensity)
            y.append(label)

            grid_intst[r][c] = intensity
            grid_label[r][c] = label

            # Non-empty
            non_e_coords.append((r, c))
        else:
            pass

    return grid_label, grid_intst, non_e_coords, x, y


def function(ipt, lambda_=0.2):

    return np.log2(1/ipt)/10 + lambda_


def generator_syn(il, ih, al, ah, count=40, lambda_=0.25):
    """
    Function to generate synthetic images
    # 0.3, 0.7, 0.6, 1.0
    """

    X_train = []
    y_train = []
    i = 0

    rst = {}

    while i < count:

        label, intst, non_e_coords, x, y = generation_sample(il, ih, al, ah)

        rst[i] = {}
        rst[i]['coord'] = non_e_coords
        rst[i]['intst'] = intst
        rst[i]['label'] = label

        X_train.extend(x)
        y_train.extend(y)

        i += 1

    X_train = np.array(X_train).reshape(-1, 1)

    model = GaussianNB()
    model.fit(X_train, y_train)

    intensity_list = []
    count_list = []

    for i in range(count):

        count = 0

        record = rst[i]
        non_e_coords = record['coord']
        intst_matrix = record['intst']
        label_matrix = record['label']

        for r, c in non_e_coords:

            intst_value = intst_matrix[r][c]
            probs = model.predict_proba([[intst_value]])[0]
            criterion = max(probs)
            epsilon = function(criterion, lambda_)

            roll = np.random.rand(1)[0]
            if roll > criterion - epsilon:

                count += 1
                intensity_list.append(intst_value)

                # Query the neighbors
                label = k_nearest_neighbor_distance(r, c, intst_matrix, model)
                label_matrix[r, c] = label

        image = sample_image(label_matrix, intst_matrix)
        rst[i]['label'] = label_matrix
        rst[i]['image'] = image

        count_list.append(count)

    return rst, intensity_list, count_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parameters of simulation script")

    parser.add_argument(
        "--num_images",
        default=40,
        type=int,
        help="the number of images to generate"
    )

    parser.add_argument(
        "--images",
        default="./data/images.pkl",
        help="the path to save the synthetic images"
    )

    parser.add_argument(
        "--il",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--ih",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--al",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--ah",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lmd",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--splitting",
        default="./data/splitting.pkl",
        help="the output of splitting as a pickle file"
    )

    args = parser.parse_args()

    rst, intensity_list, count_list = generator_syn(
        args.il, args.ih, args.al, args.ah, count=args.num_images, lambda_=args.lmd)

    # since data is equally distributed, just randomly splitting.
    num_list = [i for i in range(args.num_images)]

    indices = {}
    supe = num_list[:20]

    for i in range(10):

        temp = num_list[:20]

        indices[i] = {}
        indices[i]['D3'] = num_list[20:]

        idx = i * 2
        indices[i]['D1'] = supe[idx:idx+2]

        d4_idx = (idx + 2) % 20
        indices[i]['D4'] = supe[d4_idx:d4_idx+2]

        temp.remove(indices[i]['D1'][0])
        temp.remove(indices[i]['D1'][1])
        temp.remove(indices[i]['D4'][0])
        temp.remove(indices[i]['D4'][1])

        indices[i]['D2'] = []
        for rest in temp:
            indices[i]['D2'].extend([rest])

    with open(args.images, "wb") as handle:
        pickle.dump(rst, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.splitting, 'wb') as handle:
        pickle.dump(indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
