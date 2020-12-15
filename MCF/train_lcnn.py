import os
import sys
import pickle
import numpy as np

from pytorch_func import ImageData, CNN, get_accuracy, AutoEncoder, AECNN, Combine, ImageDataNeib, get_F1, train_epoch, evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.utils import class_weight
import matplotlib.image as mpimg

import warnings
warnings.filterwarnings("ignore")

EPOCH_NUM = 4000


class LARGECNN(nn.Module):
    def __init__(self):

        super(LARGECNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, 3)
        self.conv2 = nn.Conv2d(12, 24, 3)
        self.conv3 = nn.Conv2d(24, 48, 3)

        self.fc1 = nn.Linear(48 * 8 * 8, 160)
        self.fc2 = nn.Linear(160, 40)
        self.fc3 = nn.Linear(40, 2)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1
        for s in size:
            num_features *= s

        return num_features


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

            patch = img[u-27:d+27, l-27:r+27]

            x.append(patch)
            y.append(detail['label_name'])

    x = np.array(x)
    y = np.array(y)

    return x, y


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

        x, y = load_pickle(record['folder'][name])

        idx = 0

        for node, label in zip(x, y):

            record['images'][name][idx] = {}
            record['images'][name][idx]['node'] = node
            record['images'][name][idx]['labl'] = label
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


def train(record, cls_weight):

    dev_loss_list = []

    D1 = record['images']['D1']
    D2 = record['images']['D2']
    D3 = record['images']['D3']
    D4 = record['images']['D4']

    EPOCH = EPOCH_NUM
    BATCH_SIZE = 32

    train_data = ImageData(D2)
    train_dataset_loader = DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    valid_data = ImageData(D1)
    valid_dataset_loader = DataLoader(
        dataset=valid_data, batch_size=1, shuffle=False)

    test_data = ImageData(D4)
    test_dataset_loader = DataLoader(
        dataset=test_data, batch_size=1, shuffle=False)

    best_dev_acc = 0.0
    best_tes_acc = 0.0
    best_dev_loss = float('inf')

    model = LARGECNN().cuda()
    loss_function = nn.CrossEntropyLoss(weight=cls_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    no_up = 0
    for i in range(EPOCH):

        if i == 0 or (i+1) % 100 == 0:
            print('epoch: %d start!' % i)
            print('now best dev acc and loss:', str(
                best_dev_acc)[:5], str(best_dev_loss)[:5])
            print('corresponding test acc:', str(best_tes_acc)[:5])
            print()

        train_epoch(model, train_dataset_loader, loss_function, optimizer, i)

        dev_acc, dev_loss, (_, _) = evaluate(
            model, valid_dataset_loader, loss_function, 'dev')
        test_acc, test_loss, (test_truth, test_pred) = evaluate(
            model, test_dataset_loader, loss_function, 'test')

        dev_loss_list.append(dev_loss)

        if dev_loss < best_dev_loss:

            best_dev_acc = dev_acc
            best_tes_acc = test_acc
            best_dev_loss = dev_loss

            best_truth = test_truth
            best_predi = test_pred

            if not os.path.exists('best_models/largecnn/'):
                os.makedirs('best_models/largecnn/')

            os.system('rm best_models/largecnn/best_model_minibatch_acc_*.model')
            torch.save(model.state_dict(
            ), 'best_models/largecnn/best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
            no_up = 0

        else:
            no_up += 1
            if no_up >= 500:
                break

    print("Ending")
    print('now best dev acc:', best_dev_acc)
    print('now best dev loss:', best_dev_loss)
    print('now corresponding test acc:', best_tes_acc)
    print('now corresponding test F1:', get_F1(best_truth, best_predi)
          [0], get_F1(best_truth, best_predi)[1])
    print('*'*50)
    print()

    macro_f1, weighted_f1 = get_F1(best_truth, best_predi)

    return best_tes_acc, dev_loss_list, macro_f1, weighted_f1


def experiment(record, class_weights, trials=5):

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for i in range(trials):

        accuracy, _, macroF1, weightedF1 = train(record, class_weights)

        accuracy_list.append(accuracy)
        macroF1_list.append(macroF1)
        weigtF1_list.append(weightedF1)

    return np.average(accuracy_list), np.average(macroF1_list), np.average(weigtF1_list)


if __name__ == "__main__":

    folds = list(range(10))

    data_path = "./data/indices_55_splitting_KL.pkl"

    with open(data_path, 'rb') as handle:
        indices = pickle.load(handle)

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for fold_num in folds:

        record = load_generate(indices[fold_num])

        D2 = record['images']['D2']

        y_D2 = []

        for key, value in sorted(D2.items()):

            y_D2.append(value['labl'])

        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y_D2),
                                                          y_D2)
        class_weights = torch.tensor(class_weights).float().cuda()

        accuracy, macroF1, weightedF1 = experiment(
            record, class_weights, trials=1)

        accuracy_list.append(accuracy)
        macroF1_list.append(macroF1)
        weigtF1_list.append(weightedF1)

    rst = [accuracy_list, macroF1_list, weigtF1_list]
    rst = np.array(rst)

    np.savetxt("larger_cnn.csv", rst, delimiter=",")
