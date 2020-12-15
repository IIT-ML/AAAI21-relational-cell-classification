import os
import sys
import pickle

import numpy as np

from collections import Counter

from utils import load_generate
from pytorch_func import ImageData, CNN, get_accuracy, AutoEncoder, AECNN, Combine, ImageDataNeib, get_F1

from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import warnings
warnings.filterwarnings("ignore")

idx_file = "./data/indices_55_splitting.pkl"
image_file = "./data/images_informative_v4_3.pkl"


def load_data(Kfold_num):

    seed_num = 42

    with open(idx_file, 'rb') as handle:
        indices = pickle.load(handle)

    record = load_generate(image_file, indices[Kfold_num])

    return record


class ImageDataNodeNeibLabel(Dataset):
    def __init__(self, data_dict):

        self.data_dict = data_dict
        self.to_pillow = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.to_transf = transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __getitem__(self, index):

        current = self.data_dict[index]

        label = current['labl']
        image = current['node']
        neigb = current['neib']

        final_neibs = []

        image = self.to_tensor(image)
        image = image.type(torch.FloatTensor)
        image = self.to_transf(image)

        for n in neigb:

            # n = self.to_pillow(n)
            n = self.to_tensor(n)
            n = n.type(torch.FloatTensor)
            n = self.to_transf(n)

            final_neibs.append(n)

        return (image, final_neibs, label)

    def __len__(self):

        # of how many examples(images?) you have
        return len(self.data_dict.items())


class EncoderAE(nn.Module):
    def __init__(self, original_model):
        super(EncoderAE, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:1])

    def forward(self, x):
        x = self.features(x)
        return x


def train_ae(record):

    D3 = record['images']['D3']

    num_epochs = 600
    learning_rate = 1e-4

    model = AutoEncoder().cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)

    img_data = ImageData(D3)
    img_dataset_loader = torch.utils.data.DataLoader(dataset=img_data, batch_size=128,
                                                     shuffle=True)
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

    if not os.path.exists('./best_models/ae/'):
        os.makedirs('./best_models/ae/')

    torch.save(model.state_dict(), './best_models/ae/ae_model.model')

    return model


def svm_experiment(record, fold_num, tuned_parameters):

    print("This is %d-fold experiment" % fold_num)

    D1 = record['images']['D1']
    D2 = record['images']['D2']
    D4 = record['images']['D4']

    def transform(D):

        x = []
        y = []

        for key, value in sorted(D.items()):

            data = value['node']
            # data = transforms.ToPILImage()(data)
            data = transforms.ToTensor()(data)
            data = data.type(torch.FloatTensor)
            data = transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(data)
            data = data.numpy()

            x.append(data.flatten())
            y.append(value['labl'])

        return x, y

    x1, y1 = transform(D1)
    x2, y2 = transform(D2)

    x_test, y_test = transform(D4)

    print("Distribution of test:")
    print(Counter(y_test))
    s = Counter(y_test)[0] + Counter(y_test)[1] + \
        Counter(y_test)[2] + Counter(y_test)[3]
    m = max(Counter(y_test)[0], Counter(y_test)[1],
            Counter(y_test)[2], Counter(y_test)[3])
    print("Majority ratio is", m/s)
    print()

    best_accuracy = 0.0

    for para in ParameterGrid(tuned_parameters):

        clf = SVC(class_weight="balanced", **para)
        clf.fit(x2, y2)
        y_pred_valid = clf.predict(x1)

        current_accuracy = accuracy_score(y1, y_pred_valid)
        if current_accuracy > best_accuracy:

            best_accuracy = current_accuracy
            optimal_para = para

    print("This is SVM classifier with parameters:")
    print("\t", optimal_para)

    clf = SVC(class_weight="balanced", **optimal_para)
    clf.fit(x2, y2)
    y_pred = clf.predict(x_test)

    print(classification_report(y_test, y_pred))
    print()
    print(accuracy_score(y_test, y_pred))
    print()
    print('now corresponding test F1:', get_F1(
        y_test, y_pred)[0], get_F1(y_test, y_pred)[1])
    print()

    macro_f1, weighted_f1 = get_F1(y_test, y_pred)

    return accuracy_score(y_test, y_pred), macro_f1, weighted_f1


def svm_embed_experiment_1(record, fold_num, tuned_parameters):

    print("This is %d-fold flatten experiment" % fold_num)

    x1, n1, y1 = record['valid']
    x2, n2, y2 = record['train']
    x4, n4, y_test = record['test']

    def convert(x_array, n_array):

        flat = []
        aver = []

        for x, n in zip(x_array, n_array):

            flat.append(np.concatenate((x, n.flatten())))

            aver.append(np.concatenate((x, np.mean(n, axis=0))))

        return np.array(flat), np.array(aver)

    x1, _ = convert(x1, n1)
    x2, _ = convert(x2, n2)
    x_test, _ = convert(x4, n4)

    best_accuracy = 0.0

    for para in ParameterGrid(tuned_parameters):

        clf = SVC(class_weight="balanced", **para)
        clf.fit(x2, y2)
        y_pred_valid = clf.predict(x1)

        current_accuracy = accuracy_score(y1, y_pred_valid)
        if current_accuracy > best_accuracy:

            best_accuracy = current_accuracy
            optimal_para = para

    print("This is SVM classifier with parameters:")
    print("\t", optimal_para)

    clf = SVC(class_weight="balanced", **optimal_para)
    clf.fit(x2, y2)
    y_pred = clf.predict(x_test)

    print(classification_report(y_test, y_pred))
    print()
    print(accuracy_score(y_test, y_pred))
    print()
    print('now corresponding test F1:', get_F1(
        y_test, y_pred)[0], get_F1(y_test, y_pred)[1])
    print()

    macro_f1, weighted_f1 = get_F1(y_test, y_pred)

    return accuracy_score(y_test, y_pred), macro_f1, weighted_f1


def svm_embed_experiment_2(record, fold_num, tuned_parameters):

    print("This is %d-fold average experiment" % fold_num)

    x1, n1, y1 = record['valid']
    x2, n2, y2 = record['train']
    x4, n4, y_test = record['test']

    def convert(x_array, n_array):

        flat = []
        aver = []

        for x, n in zip(x_array, n_array):

            flat.append(np.concatenate((x, n.flatten())))
            aver.append(np.concatenate((x, np.mean(n, axis=0))))

        # Change the order which is different from another
        return np.array(aver), np.array(flat)

    x1, _ = convert(x1, n1)
    x2, _ = convert(x2, n2)
    x_test, _ = convert(x4, n4)

    best_accuracy = 0.0

    for para in ParameterGrid(tuned_parameters):

        clf = SVC(class_weight="balanced", **para)
        clf.fit(x2, y2)
        y_pred_valid = clf.predict(x1)

        current_accuracy = accuracy_score(y1, y_pred_valid)
        if current_accuracy > best_accuracy:

            best_accuracy = current_accuracy
            optimal_para = para

    print("This is SVM classifier with parameters:")
    print("\t", optimal_para)

    clf = SVC(class_weight="balanced", **optimal_para)
    clf.fit(x2, y2)
    y_pred = clf.predict(x_test)

    print(classification_report(y_test, y_pred))
    print()
    print(accuracy_score(y_test, y_pred))
    print()
    print('now corresponding test F1:', get_F1(
        y_test, y_pred)[0], get_F1(y_test, y_pred)[1])
    print()

    macro_f1, weighted_f1 = get_F1(y_test, y_pred)

    return accuracy_score(y_test, y_pred), macro_f1, weighted_f1


def input2embed(encoder, dataset_loader):

    x = []
    e = []
    y = []

    for data in dataset_loader:

        _, neigb, label = data
        temp = []

        for n in neigb:

            n = Variable(n).cuda()
            output = encoder(n)
            output = output.view(-1).cpu().detach().numpy()

            temp.append(output)

        x.append(temp[-1])
        e.append(temp[:-1])
        y.append(label)

    x = np.array(x)
    e = np.array(e)
    y = np.array(y)

    return x, e, y


if __name__ == "__main__":

    seed_num = 42

    KFolds = range(10)
    tuned_parameters = {'kernel': ['rbf'],
                        'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}

    accuracy_list = []
    macroF1_list = []
    weigtF1_list = []

    for fold_num in KFolds:

        data = load_data(fold_num)
        accuracy, macroF1, weightedF1 = svm_experiment(
            data, fold_num, tuned_parameters)

        accuracy_list.append(accuracy)
        macroF1_list.append(macroF1)
        weigtF1_list.append(weightedF1)

    rst = [accuracy_list, macroF1_list, weigtF1_list]
    rst = np.array(rst)

    np.savetxt("svm.csv", rst, delimiter=",")

    ################################################################

    acc_fla_list = []
    mf1_fla_list = []
    wf1_fla_list = []

    acc_ave_list = []
    mf1_ave_list = []
    wf1_ave_list = []

    ae_data = load_data(0)
    ae_model = train_ae(ae_data)

    encoderAE = EncoderAE(ae_model).cuda()
    encoderAE.eval()

    for fold_num in KFolds:

        data = load_data(fold_num)

        D1 = data['images']['D1']
        D2 = data['images']['D2']
        D3 = data['images']['D3']
        D4 = data['images']['D4']

        d1_loader = torch.utils.data.DataLoader(
            dataset=ImageDataNodeNeibLabel(D1), batch_size=1, shuffle=False)
        d2_loader = torch.utils.data.DataLoader(
            dataset=ImageDataNodeNeibLabel(D2), batch_size=1, shuffle=False)
        d4_loader = torch.utils.data.DataLoader(
            dataset=ImageDataNodeNeibLabel(D4), batch_size=1, shuffle=False)

        record = {}
        record['valid'] = input2embed(encoderAE, d1_loader)
        record['train'] = input2embed(encoderAE, d2_loader)
        record['test'] = input2embed(encoderAE, d4_loader)

        accuracy, macroF1, weightedF1 = svm_embed_experiment_1(
            record, fold_num, tuned_parameters)
        acc_fla_list.append(accuracy)
        mf1_fla_list.append(macroF1)
        wf1_fla_list.append(weightedF1)

        accuracy, macroF1, weightedF1 = svm_embed_experiment_2(
            record, fold_num, tuned_parameters)
        acc_ave_list.append(accuracy)
        mf1_ave_list.append(macroF1)
        wf1_ave_list.append(weightedF1)

    rst = [acc_fla_list, mf1_fla_list, wf1_fla_list]
    rst = np.array(rst)
    np.savetxt("svm_flatten_neib.csv", rst, delimiter=",")

    rst = [acc_ave_list, mf1_ave_list, wf1_ave_list]
    rst = np.array(rst)
    np.savetxt("svm_average_neib.csv", rst, delimiter=",")
