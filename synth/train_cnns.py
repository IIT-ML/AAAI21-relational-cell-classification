import os
import sys
import pickle
import numpy as np

from utils import load_generate
from pytorch_func import ImageData, CNN, get_accuracy, AutoEncoder, AECNN, Combine, ImageDataNeib, get_F1, train_epoch, evaluate

from sklearn.svm import SVC
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

EPOCH_NUM = 200

idx_file = "./data/indices_55_splitting.pkl"
image_file = "./data/images_informative_v4_3.pkl"

ae_model_name = 'ae_model.model'


def experiment(record, class_weights, function, trials=5):

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for i in range(trials):

        accuracy, _, macroF1, weightedF1 = function(record, class_weights)

        accuracy_list.append(accuracy)
        macroF1_list.append(macroF1)
        weigtF1_list.append(weightedF1)

    return np.average(accuracy_list), np.average(macroF1_list), np.average(weigtF1_list)


def train_cnn(record, cls_weight):

    dev_loss_list = []

    D1 = record['images']['D1']
    D2 = record['images']['D2']
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

    model = CNN().cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    no_up = 0
    for i in range(EPOCH):

        if i == 0 or (i+1) % 50 == 0:
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

            if not os.path.exists('./best_models/cnn/'):
                os.makedirs('./best_models/cnn/')

            os.system('rm best_models/cnn/best_model_minibatch_acc_*.model')
            torch.save(model.state_dict(
            ), 'best_models/cnn/best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
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


def entry_cnn(data_path):

    folds = list(range(10))
    seed_num = 42

    with open(data_path, 'rb') as handle:
        indices = pickle.load(handle)

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for fold_num in folds:

        record = load_generate(image_file, indices[fold_num])

        D2 = record['images']['D2']
        y_D2 = []

        for key, value in sorted(D2.items()):
            y_D2.append(value['labl'])

        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(y_D2), y_D2)
        class_weights = torch.tensor(class_weights).float().cuda()

        accuracy, macroF1, weightedF1 = experiment(
            record, class_weights, train_cnn, trials=1)

        accuracy_list.append(accuracy)
        macroF1_list.append(macroF1)
        weigtF1_list.append(weightedF1)

    rst = [accuracy_list, macroF1_list, weigtF1_list]
    rst = np.array(rst)

    np.savetxt("cnn.csv", rst, delimiter=",")


def train_aecnn(record, cls_weight):

    dev_loss_list = []

    D1 = record['images']['D1']
    D2 = record['images']['D2']
    D4 = record['images']['D4']

    EPOCH = EPOCH_NUM
    BATCH_SIZE = 64

    path = "./best_models/ae/" + ae_model_name

    ae_model = AutoEncoder()
    ae_model.load_state_dict(torch.load(path))

    train_data = ImageData(D2)
    train_dataset_loader = DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    valid_data = ImageData(D1)
    valid_dataset_loader = DataLoader(
        dataset=valid_data, batch_size=1, shuffle=True)

    test_data = ImageData(D4)
    test_dataset_loader = DataLoader(
        dataset=test_data, batch_size=1, shuffle=True)

    best_dev_acc = 0.0
    best_tes_acc = 0.0
    best_dev_loss = float('inf')

    model = AECNN(ae_model).cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    no_up = 0
    for i in range(EPOCH):

        if i == 0 or (i+1) % 50 == 0:
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

            if not os.path.exists('./best_models/aecnn/'):
                os.makedirs('./best_models/aecnn/')

            os.system('rm best_models/aecnn/best_model_minibatch_acc_*.model')
            torch.save(model.state_dict(
            ), 'best_models/aecnn/best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
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


def entry_aecnn(data_path):

    seed_num = 42
    folds = list(range(10))

    with open(data_path, 'rb') as handle:
        indices = pickle.load(handle)

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for fold_num in folds:

        record = load_generate(image_file, indices[fold_num])

        D2 = record['images']['D2']
        y_D2 = []

        for key, value in sorted(D2.items()):
            y_D2.append(value['labl'])

        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(y_D2), y_D2)
        class_weights = torch.tensor(class_weights).float().cuda()

        accuracy, macroF1, weightedF1 = experiment(
            record, class_weights, train_aecnn, trials=1)

        accuracy_list.append(accuracy)
        macroF1_list.append(macroF1)
        weigtF1_list.append(weightedF1)

    rst = [accuracy_list, macroF1_list, weigtF1_list]
    rst = np.array(rst)

    np.savetxt("aecnn.csv", rst, delimiter=",")


def train_lstm(record, cls_weight):

    dev_loss_list = []

    D1 = record['images']['D1']
    D2 = record['images']['D2']
    D4 = record['images']['D4']

    path = "./best_models/ae/" + ae_model_name
    ae_model = AutoEncoder()
    ae_model.load_state_dict(torch.load(path))

    EPOCH = EPOCH_NUM
    BATCH_SIZE = 64

    train_data = ImageDataNeib(D2)
    train_dataset_loader = DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    valid_data = ImageDataNeib(D1)
    valid_dataset_loader = DataLoader(
        dataset=valid_data, batch_size=1, shuffle=True)

    test_data = ImageDataNeib(D4)
    test_dataset_loader = DataLoader(
        dataset=test_data, batch_size=1, shuffle=True)

    best_dev_acc = 0.0
    best_tes_acc = 0.0
    best_dev_loss = float('inf')

    model = Combine(ae_model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    no_up = 0
    for i in range(400):

        if i == 0 or (i+1) % 50 == 0:
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

            if not os.path.exists('./best_models/lstm/'):
                os.makedirs('./best_models/lstm/')

            os.system('rm best_models/lstm/best_model_minibatch_acc_*.model')
            torch.save(model.state_dict(
            ), 'best_models/lstm/best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
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


def entry_lstm(data_path):

    folds = list(range(10))
    seed_num = 42

    with open(data_path, 'rb') as handle:
        indices = pickle.load(handle)

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for fold_num in folds:

        print("This is fold number: ", (fold_num+1))

        record = load_generate(image_file, indices[fold_num])

        D2 = record['images']['D2']

        y_D2 = []

        for key, value in sorted(D2.items()):

            y_D2.append(value['labl'])

        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(y_D2), y_D2)
        class_weights = torch.tensor(class_weights).float().cuda()

        accuracy, macroF1, weightedF1 = experiment(
            record, class_weights, train_lstm, trials=1)

        accuracy_list.append(accuracy)
        macroF1_list.append(macroF1)
        weigtF1_list.append(weightedF1)

    rst = [accuracy_list, macroF1_list, weigtF1_list]
    rst = np.array(rst)

    np.savetxt("cnnlstm.csv", rst, delimiter=",")


if __name__ == "__main__":

    # ===================Train CNN========================
    print("CNN")
    entry_cnn(idx_file)

    # ===================Train AECNN========================
    print("AE-CNN")
    entry_aecnn(idx_file)

    # ===================Train LSTM========================
    print("AE-CNN-LSTM")
    entry_lstm(idx_file)
