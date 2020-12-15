import os
import pickle
import operator
import argparse
import numpy as np
from scipy.stats import entropy


def load_pickle_overall(pickle_path):
    """"
    pickle_path: pkl file path, which contains label and coordinates 
    """

    size = 13
    row, col = 512, 640

    with open(pickle_path, 'rb') as handle:
        record = pickle.load(handle)

    label_details = {}
    y = []

    error_count = 0

    for img_name, data in record.items():

        value = [0, 0]
        label_details[img_name] = value

        for idx, detail in data.items():

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
                error_count += 1
                continue

            label = detail["label_name"]
            label_details[img_name][label] += 1

            y.append(label)

    return label_details, y


def normalize(probs, num_labels=2):

    value = probs.copy()

    if sum(value) == 0.:

        return [0.1 for i in range(num_labels)]

    else:

        value = [i/sum(value) for i in value]

    for idx, v in enumerate(value):

        if v == 0.0:

            value[idx] = 1e-20

            imax = np.argmax(value)

            value[imax] -= 1e-20

    return value


def klgreedy(image2count, target_probs, random_seed=42, num_split=10):

    np.random.seed(random_seed)

    keys = list(image2count.keys())
    np.random.shuffle(keys)

    rst = {}

    for i in range(num_split):

        rst[i] = {}
        rst[i]['folder'] = [keys[i]]
        rst[i]['count'] = image2count[keys[i]]
        rst[i]['ratio'] = normalize(rst[i]['count'])

    if len(image2count) % num_split == 0:
        num_iterate = len(image2count) // num_split - 1
    else:
        num_iterate = len(image2count) // num_split

    start = num_split

    for idx in range(num_iterate):

        images = keys[start:start+num_split]

        scores = {key: float("inf") for key, value in rst.items()}

        # Add image one by one into n-folds.
        for img in images:

            for key, value in scores.items():

                new_count = [i+j for i,
                             j in zip(image2count[img], rst[key]['count'])]
                converted_score = normalize(new_count)

                kl_score = entropy(target_probs, converted_score)
                scores[key] = kl_score

            # Find the key which has the smallest kl score
            smallest_k = min(scores.items(), key=operator.itemgetter(1))[0]

            rst[smallest_k]['folder'].append(img)

            temp = rst[smallest_k]['count']
            rst[smallest_k]['count'] = [
                i+j for i, j in zip(image2count[img], temp)]
            rst[smallest_k]['ratio'] = normalize(rst[smallest_k]['count'])

            del scores[smallest_k]

            for key, value in scores.items():
                scores[key] = float("inf")

        start += num_split

    return rst


def get_target_distribution(data):
    """
    Get the class distribution given the dataset
    """
    target = []

    for key, value in data.items():
        target.append(value)

    target = np.array(target)
    target = np.mean(target, axis=0)
    target = target/np.sum(target)

    return target


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parameters of simulation script")

    parser.add_argument(
        "--num",
        default=20,
        type=int,
        help="the number of runs for KL-greedy"
    )

    parser.add_argument(
        "--input",
        default="./data/record.pkl",
        help="the input of images"
    )

    parser.add_argument(
        "--output",
        default="./data/splitting.pkl",
        help="the output of splitting as a pickle file"
    )

    args = parser.parse_args()

    # Use KL-Greedy for 2-folds first
    label_details, y = load_pickle_overall(args.input)

    # Calculate the targt distribution
    target = get_target_distribution(label_details)

    samples = {}
    score_map = {}

    for key in range(args.num):

        scores = []

        current_record = klgreedy(label_details, target, key, num_split=2)

        for i in range(2):
            scores.append(entropy(target, current_record[i]['ratio']))

        samples[key] = current_record
        score_map[key] = scores

    key_list = []
    max_list = []

    for key, value in score_map.items():

        key_list.append(key)
        max_list.append(np.max(value))

    idx = np.argmin(max_list)

    unsu = samples[idx][0]['folder']
    supe = samples[idx][1]['folder']

    # Use KL-Greedy for 10-folds
    supevised_data = {}

    for img_name, data in label_details.items():
        if img_name in supe:
            supevised_data[img_name] = data

    target = get_target_distribution(supevised_data)

    samples = {}
    score_map = {}

    for key in range(args.num):

        scores = []
        current_record = klgreedy(supevised_data, target, key, num_split=10)

        for i in range(10):
            scores.append(entropy(target, current_record[i]['ratio']))

        samples[key] = current_record
        score_map[key] = scores

    key_list = []
    max_list = []

    for key, value in score_map.items():

        key_list.append(key)
        max_list.append(np.max(value))

    idx = np.argmin(max_list)

    indices = {}

    for i in range(10):

        idx_list = list(range(10))

        indices[i] = {}
        indices[i]['D4'] = samples[idx][i]['folder']

        d1_idx = (i + 1) % 10
        indices[i]['D1'] = samples[idx][d1_idx]['folder']

        indices[i]['D3'] = unsu

        idx_list.remove(i)
        idx_list.remove(d1_idx)

        indices[i]['D2'] = []

        for rest in idx_list:
            indices[i]['D2'].extend(samples[idx][rest]['folder'])

    with open(args.output, 'wb') as handle:
        pickle.dump(indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
