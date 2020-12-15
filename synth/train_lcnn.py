import os
import sys
import pickle

from collections import Counter

import numpy as np

from pytorch_func import ImageData, CNN, get_accuracy, AutoEncoder, AECNN, Combine, ImageDataNeib, get_F1, train_epoch, evaluate

from sklearn.svm import SVC
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")

EPOCH_NUM = 200

idx_file = "./data/indices_55_splitting.pkl"
image_file = "./data/images_informative_v4_3.pkl"


class LARGECNN(nn.Module):
    def __init__(self):

        super(LARGECNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 12, 3)
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


def load_pickle(indices, image_data):
    """"
    0: Empty
    1: Active
    2: Inactive
    """
    size = 13

    with open(image_data, "rb") as f:
        images = pickle.load(f)

    x = []
    y = []

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

                    pt = img[u-27:d+27, l-27:r+27]
                    lb = label[g_r][g_c]

                    x.append(pt)
                    y.append(lb)

                img_c += 27
            img_r += 27

    x = np.array(x)
    y = np.array(y)

    return x, y


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

        x, y = load_pickle(record['folder'][name], image_data)

        idx = 0

        for node, label in zip(x, y):

            record['images'][name][idx] = {}
            record['images'][name][idx]['node'] = node

            if label == 1.0:

                record['images'][name][idx]['labl'] = 1

            elif label == 2.0:

                record['images'][name][idx]['labl'] = 0

            else:

                raise ValueError("AAA")

            idx += 1

    fill_dict('D1', 'pickle')
    fill_dict('D2', 'pickle')
    fill_dict('D3', 'pickle')
    fill_dict('D4', 'pickle')

#     pickle_name = './data/data_' + str(seednum) + '.pkl'
#     with open(pickle_name, 'wb') as handle:
#         pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)

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


def train(record, cls_weight):

    dev_loss_list = []

    D1 = record['images']['D1']
    D2 = record['images']['D2']
    D3 = record['images']['D3']
    D4 = record['images']['D4']

    EPOCH = EPOCH_NUM
    BATCH_SIZE = 64

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
    loss_function = nn.CrossEntropyLoss()
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

    with open(idx_file, 'rb') as handle:
        indices = pickle.load(handle)

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for fold_num in folds:

        record = record = load_generate(image_file, indices[fold_num])

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

    np.savetxt("large_cnn.csv", rst, delimiter=",")
