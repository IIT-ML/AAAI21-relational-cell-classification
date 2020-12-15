#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Different classes and function used for training and evalution.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.autograd import Variable

from sklearn.metrics import f1_score


def get_accuracy(truth, pred):

    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right/len(truth)


def get_F1(truth, pred):

    return f1_score(truth, pred, average='macro'), f1_score(truth, pred, average='weighted')


class ImageData(Dataset):
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

        image = self.to_pillow(image)
        image = self.to_tensor(image)
        image = self.to_transf(image)

        return (image, np.int64(label))

    def __len__(self):

        # of how many examples(images?) you have
        return len(self.data_dict.items())


class ImageDataNeib(Dataset):
    def __init__(self, data_dict):

        self.data_dict = data_dict
        self.to_pillow = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.to_transf = transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __getitem__(self, index):

        current = self.data_dict[index]

        label = current['labl']
        neigb = current['neib']

        final_neibs = []

        for n in neigb:

            n = self.to_pillow(n)
            n = self.to_tensor(n)
            n = self.to_transf(n)
            final_neibs.append(n)

        final_neibs = torch.stack(final_neibs)

        return (final_neibs, np.int64(label))

    def __len__(self):

        # of how many examples(images?) you have
        return len(self.data_dict.items())


class CNN(nn.Module):
    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, 4)
        self.conv2 = nn.Conv2d(12, 24, 3)

        self.fc1 = nn.Linear(24 * 5 * 5, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 4)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

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


class AutoEncoder(nn.Module):
    def __init__(self):

        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(12, 24, 3),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 8, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 5, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AECNN(nn.Module):
    def __init__(self, original_model):

        super(AECNN, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:1])
        self.fc1 = nn.Linear(24 * 5 * 5, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.features(x)
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


class CNNFC(nn.Module):
    def __init__(self, original_model):

        super(CNNFC, self).__init__()

        self.features = nn.Sequential(*list(original_model.children())[:1])
        self.fc1 = nn.Linear(24 * 5 * 5, 160)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.features(x)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Combine(nn.Module):
    """
    Docstring
    """

    def __init__(self, original_model):
        """
        Docstring
        """
        super(Combine, self).__init__()

        self.num_layer = 1

        self.cnn = CNNFC(original_model)
        self.rnn = nn.LSTM(
            input_size=160,
            hidden_size=100,
            num_layers=1,
            batch_first=True)

        self.linear = nn.Linear(100, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Docstring
        """
        batch_size, timesteps, C, H, W = x.size()

        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)

        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out = self.dropout(r_out[:, -1, :])
        r_out = self.linear(r_out)

        # log_probs = F.log_softmax(r_out2, dim=-1)

        return r_out

    def init_hidden(self):

        return (autograd.Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_dim)).cuda())


def train_epoch(model, train_iter, loss_function, optimizer, i):

    model.train()

    device = torch.device("cuda")
    model.to(device)

    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for batch in train_iter:

        data, label = batch

        label = Variable(label).cuda()
        data = Variable(data).cuda()

        truth_res += list(label.data.cpu().numpy().tolist())
        model.batch_size = len(label.data)

        pred = model(data)

        pred_label = pred.data.cpu().max(1)[1].numpy().tolist()
        pred_res += pred_label

        model.zero_grad()

        loss = loss_function(pred, label)
        avg_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss /= len(train_iter)


def evaluate(model, eval_iter, loss_function,  name='dev'):

    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for batch in eval_iter:

        data, label = batch

        label = Variable(label).cuda()
        data = Variable(data).cuda()

        truth_res += list(label.data.cpu().numpy().tolist())
        model.batch_size = len(label.data)

        pred = model(data)

        pred_label = pred.data.cpu().max(1)[1].numpy().tolist()
        pred_res += pred_label

        loss = loss_function(pred, label)
        avg_loss += loss.item()

    avg_loss /= len(eval_iter)
    acc = get_accuracy(truth_res, pred_res)

    return acc, avg_loss, (truth_res, pred_res)
