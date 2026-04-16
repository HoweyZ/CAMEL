import os

import numpy as np
import torch
import pandas as pd
import math
import time

def write_into_xls(excel_name, mat, columns=None):
    file_extension = os.path.splitext(excel_name)[1]

    if file_extension != ".xls" and file_extension != ".xlsx":
        raise ValueError('excel_name is not right in write_into_xls')

    folder_name = os.path.dirname(excel_name)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)

    if isinstance(mat, np.ndarray) and mat.ndim > 2:
        mat = mat.reshape(-1, mat.shape[-1])
        mat = mat[:1000]
    if columns is not None:
        dataframe = pd.DataFrame(mat, columns=columns)
    else:
        dataframe = pd.DataFrame(mat)
    # print(dataframe)
    # print(excel_name)
    dataframe.to_excel(excel_name, index=False)

def plot_mat(mat, str_cat='series_2D', str0='tmp', save_folder='./results'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.switch_backend('agg')
    if not isinstance(mat, np.ndarray):
        mat = mat.detach().cpu().numpy()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # fig, axs = plt.subplots(1, 1)
    # plt.imshow(mat, cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)  # viridis  hot
    # plt.colorbar()

    plt.figure(figsize=(8, 8))
    sns.heatmap(mat, annot=False, cmap='coolwarm', square=True, cbar=True)
    plt.xticks([])  # 去除x轴刻度
    plt.yticks([])  # 去除y轴刻度
    timestamp = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    plt.savefig(os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.pdf'))
    plt.show()
    # save to excel
    excel_name = os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.xlsx')
    write_into_xls(excel_name, mat)
    # save to npy
    np.save(os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.npy'), mat)

def create_sin_pos_embed(max_len, d_model):
    pe = torch.zeros(max_len, d_model).float()

    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float()
                * -(math.log(10000.0) / d_model)).exp()

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)

    #  [1, max_len, d_model]
    return pe

def hier_half_token_weight(token_weight, ratio=2):
    if token_weight is None:
        return None
    # temp_token_weight_time: [b, token_num]
    B, N = token_weight.shape
    if N % ratio != 0:
        tmp = ratio - N % ratio
        token_weight = torch.cat([token_weight, token_weight[:, -tmp:]], dim=-1)
    token_weight = token_weight.reshape(B, -1, ratio).sum(dim=-1)
    return token_weight

def forward_fill(x, mask):
    b, l, n = x.size()
    # x = x.clone()
    mask = mask.clone()

    padding_positions = (mask == 1).nonzero(as_tuple=True)

    for batch_index, length_index, feature_index in zip(*padding_positions):
        # search backwards
        for prev_length_index in range(length_index - 1, -1, -1):
            if mask[batch_index, prev_length_index, feature_index] == 0:
                x[batch_index, length_index, feature_index] = x[batch_index, prev_length_index, feature_index]
                mask[batch_index, length_index, feature_index] = 0
                break

    padding_positions = (mask == 1).nonzero(as_tuple=True)

    for batch_index, length_index, feature_index in zip(*padding_positions):
        # search forwards
        for prev_length_index in range(length_index + 1, l, 1):
            if mask[batch_index, prev_length_index, feature_index] == 0:
                x[batch_index, length_index, feature_index] = x[batch_index, prev_length_index, feature_index]
                mask[batch_index, length_index, feature_index] = 0
                break

    return x, mask


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    import matplotlib.pyplot as plt

    plt.switch_backend('agg')
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = x.transpose(-1, -2) / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = I.unsqueeze(0)

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z
