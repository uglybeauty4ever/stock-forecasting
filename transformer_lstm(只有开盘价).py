import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tqdm
from torch.autograd import Variable
import argparse
import math
import torch.nn.functional as F

torch.random.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser("Transformer-LSTM")
parser.add_argument("-data_path", type=str, default="D:\\Users\\86188\\Desktop\\v2\\merged_file.csv",
                    help="dataset path")

args = parser.parse_args()
time_step = 10  # 根据10天数据预测第11天


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:len(pe[:, 1::2][0])])

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=1, num_layers=2, hidden_size=64, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.hidden_size = hidden_size

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model=feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(feature_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(1):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class AttnDecoder(nn.Module):
    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=1)
        self.tilde = nn.Linear(in_features=code_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(in_features=code_hidden_size + hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, h, y_seq):
        h_ = h.transpose(0, 1)
        batch_size = h.size(0)
        if y_seq.size(0) != batch_size:
            y_seq = y_seq[:batch_size, :]

        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)

        for t in range(self.T):
            if t < h_.size(0):
                x = torch.cat((d, h_[t, :, :].unsqueeze(0)), 2)
                attn_weights = self.attn3(self.tanh(self.attn2(self.attn1(x))))
                attn_weights = F.softmax(attn_weights, dim=0)

                h1 = attn_weights * h_[t, :, :].unsqueeze(0)
                _, states = self.lstm(y_seq[:, t].unsqueeze(0), (h1, s))

                d = states[0]
                s = states[1]
        y_res = self.fc2(self.fc1(torch.cat((d.squeeze(0), h_[-1, :, :]), dim=1)))
        return y_res

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        return Variable(zero_tensor)


class StockDataset(Dataset):
    def __init__(self, file_path, T=time_step, train_flag=True):
        data_pd = pd.read_csv(file_path)
        feature_data = data_pd[['open']]  # 只保留开盘价

        self.train_flag = train_flag
        self.data_train_ratio = 0.9
        self.T = T

        data_all = np.array(feature_data, dtype=np.float32)
        data_all = (data_all - np.mean(data_all, axis=0)) / np.std(data_all, axis=0)

        if train_flag:
            self.data_len = int(self.data_train_ratio * len(data_all))
            self.data = data_all[:self.data_len]
        else:
            self.data_len = int((1 - self.data_train_ratio) * len(data_all))
            self.data = data_all[-self.data_len:]

    def __len__(self):
        return self.data_len - self.T

    def __getitem__(self, idx):
        input_sequence = self.data[idx:idx + self.T]
        target_value = self.data[idx + self.T, 0]  # 使用'open'列作为标签
        return input_sequence, target_value


def l2_loss(pred, label):
    loss = torch.nn.functional.mse_loss(pred, label, reduction='mean')
    return loss


def train_once(encoder, decoder, dataloader, encoder_optim, decoder_optim):
    encoder.train()
    decoder.train()
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    for idx, (data, label) in enumerate(loader):
        data_x = data.transpose(0, 1).float()
        label = label.float()

        code_hidden = encoder(data_x)
        code_hidden = code_hidden.transpose(0, 1)
        output = decoder(code_hidden, data.float())

        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        loss = l2_loss(output.squeeze(1), label)
        loss.backward()
        encoder_optim.step()
        decoder_optim.step()
        loss_epoch += loss.detach().item()
    loss_epoch /= len(loader)
    return loss_epoch


def eval_once(encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    preds = []
    labels = []
    for idx, (data, label) in enumerate(loader):
        data_x = data.transpose(0, 1).float()
        label = label.float()
        data_y = data.float().transpose(0, 1)

        code_hidden = encoder(data_x)
        output = decoder(code_hidden, data_y).squeeze(1)

        min_len = min(output.size(0), label.size(0))
        output = output[:min_len]
        label = label[:min_len]

        loss = l2_loss(output, label)
        loss_epoch += loss.detach().item()

        preds += output.detach().tolist()
        labels += label.detach().tolist()

    if len(preds) > 1 and len(labels) > 1:
        preds = torch.Tensor(preds)
        labels = torch.Tensor(labels)
        pred_ = preds[1:] > preds[:-1]
        label_ = labels[1:] > labels[:-1]
        accuracy = (label_ == pred_).sum().item() / len(pred_)
    else:
        accuracy = 0

    loss_epoch /= len(loader)
    return loss_epoch, accuracy


def eval_plot(encoder, decoder, dataloader):
    dataloader.shuffle = False
    preds = []
    labels = []
    encoder.eval()
    decoder.eval()
    loader = tqdm.tqdm(dataloader)
    for idx, (data, label) in enumerate(loader):
        data_x = data.transpose(0, 1).float()
        data_y = data.float()

        if data_y.size(1) > time_step:
            data_y = data_y[:, :time_step, :]
        elif data_y.size(1) < time_step:
            padding = torch.zeros((data_y.size(0), time_step - data_y.size(1), data_y.size(2)))
            data_y = torch.cat((data_y, padding), dim=1)

        code_hidden = encoder(data_x)

        if code_hidden.size(0) != data_y.size(0):
            if code_hidden.size(0) > data_y.size(0):
                code_hidden = code_hidden[:data_y.size(0)]
            else:
                data_y = data_y[:code_hidden.size(0)]

        output = decoder(code_hidden, data_y)
        preds += output.detach().tolist()
        labels += label.detach().tolist()

    min_len = min(len(preds), len(labels))
    preds = preds[:min_len]
    labels = labels[:min_len]

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig, ax = plt.subplots(figsize=(12, 6))
    data_x = list(range(len(preds)))
    ax.plot(data_x, preds, label='预测值', color='red')
    ax.plot(data_x, labels, label='实际值', color='blue')

    # 添加轴标签和标题
    ax.set_xlabel('时间步')
    ax.set_ylabel('标准化后的开盘价')
    ax.set_title('预测开盘价 vs 实际开盘价（特征向量只有开盘价时）')

    # 添加图例
    plt.legend()

    # 添加网格线以增加可读性
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图片
    plt.savefig('特征向量只有开盘价.png')

    # 显示图片
    plt.show()


def main():
    dataset_train = StockDataset(file_path=args.data_path)
    dataset_val = StockDataset(file_path=args.data_path, train_flag=False)

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False, drop_last=False)

    encoder = TransAm(feature_size=1)  # 设置特征维度为 1
    decoder = AttnDecoder(code_hidden_size=64, hidden_size=64, time_step=time_step)

    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=0.001)

    total_epoch = 400
    for epoch_idx in range(total_epoch):
        train_loss = train_once(encoder, decoder, train_loader, encoder_optim, decoder_optim)
        print("stage: train, epoch:{:5d}, loss:{}".format(epoch_idx, train_loss))

        if epoch_idx % 399 == 0:
            eval_loss, accuracy = eval_once(encoder, decoder, val_loader)
            print("####stage: test, epoch:{:5d}, loss:{}, accuracy:{}".format(epoch_idx, eval_loss, accuracy))
            eval_plot(encoder, decoder, val_loader)


if __name__ == "__main__":
    main()