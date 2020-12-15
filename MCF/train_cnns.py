import os
import sys
import pickle

from collections import Counter

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

EPOCH_NUM = 4000
BATCH_SIZE = 32


def ae_experiment(record):

    D3 = record['images']['D3']

    num_epochs = 4000
    model = AutoEncoder().cuda()

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    img_data = ImageData(D3)
    img_dataset_loader = torch.utils.data.DataLoader(
        dataset=img_data, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(num_epochs):

        total_loss = 0

        for data in img_dataset_loader:

            img, _ = data
            img = Variable(img).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
        # ===================log========================
        if (epoch+1) % 100 == 0:
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch+1, num_epochs, total_loss))

    if not os.path.exists('best_models/ae/'):
        os.makedirs('best_models/ae/')

    torch.save(model.state_dict(), 'best_models/ae/ae_model.model')


def entry_ae(data_path):

    Kfold_num = 0

    with open(data_path, 'rb') as handle:
        indices = pickle.load(handle)

    record = load_generate(indices[Kfold_num])

    ae_experiment(record)


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
    D3 = record['images']['D3']
    D4 = record['images']['D4']

    EPOCH = EPOCH_NUM

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

            if not os.path.exists('best_models/cnn/'):
                os.makedirs('best_models/cnn/')

            os.system('rm best_models/cnn/best_model_minibatch_acc_*.model')
            torch.save(model.state_dict(
            ), 'best_models/cnn/best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
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


def entry_cnn(data_path):

    folds = list(range(10))
    seed_num = 42
    aug_list = ['org']

    with open(data_path, 'rb') as handle:
        indices = pickle.load(handle)

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for fold_num in folds:

        print("This is fold number: ", (fold_num+1))

        record = load_generate(indices[fold_num])

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
    D3 = record['images']['D3']
    D4 = record['images']['D4']

    EPOCH = EPOCH_NUM

    path = 'best_models/ae/ae_model.model'

    ae_model = AutoEncoder()
    ae_model.load_state_dict(torch.load(path))

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

    model = AECNN(ae_model).cuda()
    loss_function = nn.CrossEntropyLoss(weight=cls_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    no_up = 0
    for i in range(EPOCH):

        if i == 1 or (i+1) % 100 == 0:
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

            if not os.path.exists('best_models/aecnn/'):
                os.makedirs('best_models/aecnn/')

            os.system('rm best_models/aecnn/best_model_minibatch_acc_*.model')
            torch.save(model.state_dict(
            ), 'best_models/aecnn/best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
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


def entry_aecnn(data_path):

    folds = list(range(10))
    seed_num = 42
    aug_list = ['org']

    with open(data_path, 'rb') as handle:
        indices = pickle.load(handle)

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for fold_num in folds:

        print("This is fold number: ", (fold_num+1))

        record = load_generate(indices[fold_num])

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
    D3 = record['images']['D3']
    D4 = record['images']['D4']

    path = 'best_models/ae/ae_model.model'
    ae_model = AutoEncoder()
    ae_model.load_state_dict(torch.load(path))

    EPOCH = EPOCH_NUM
    BATCH_SIZE = 32

    train_data = ImageDataNeib(D2)
    train_dataset_loader = DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    valid_data = ImageDataNeib(D1)
    valid_dataset_loader = DataLoader(
        dataset=valid_data, batch_size=1, shuffle=False)

    test_data = ImageDataNeib(D4)
    test_dataset_loader = DataLoader(
        dataset=test_data, batch_size=1, shuffle=False)

    best_dev_acc = 0.0
    best_tes_acc = 0.0
    best_dev_loss = float('inf')

    model = Combine(ae_model)

    loss_function = nn.CrossEntropyLoss(weight=cls_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    no_up = 0
    for i in range(EPOCH):

        if i == 1 or (i+1) % 200 == 0:
            print('epoch: %d start!' % i)
            print('now best dev acc and loss:', str(
                best_dev_acc)[:5], str(best_dev_loss)[:5])
            print('corresponding test acc:', str(best_tes_acc)[:5])
            print()

        train_epoch(model, train_dataset_loader, loss_function, optimizer, i)

        # print('now best dev acc:',best_dev_acc)
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

            if not os.path.exists('best_models/lstm/'):
                os.makedirs('best_models/lstm/')

            os.system('rm best_models/lstm/best_model_minibatch_acc_*.model')
            torch.save(model.state_dict(
            ), 'best_models/lstm/best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
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


def entry_lstm(data_path):

    folds = list(range(10))

    seed_num = 42
    aug_list = ['org']

    with open(data_path, 'rb') as handle:
        indices = pickle.load(handle)

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for fold_num in folds:

        print("This is fold number: ", (fold_num+1))

        record = load_generate(indices[fold_num])

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

    np.savetxt("cnn_lstm.csv", rst, delimiter=",")


if __name__ == "__main__":

    data_path = "./data/indices_55_splitting_KL.pkl"

    # ===================Train auto-encoder========================
    entry_ae(data_path)

    # ===================Train CNN========================
    entry_cnn(data_path)

    # ===================Train AECNN========================
    entry_aecnn(data_path)

    # ===================Train LSTM========================
    entry_lstm(data_path)
