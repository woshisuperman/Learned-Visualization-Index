# -*- coding: utf-8 -*-

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from functools import reduce
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model2 import Net2

from config import opt

class Data(Dataset):
    def __init__(self, x, y):
        self.data = list(zip(x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.data[idx]

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        x = x1 * 1000 + x2
        return torch.mean(torch.pow((x - y), 2))

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def normalize(a):  # 标准化
    mean = np.array(a).mean()  # 计算平均数
    deviation = np.array(a).std()  # 计算标准差
    a = ((np.array(a) - mean) / deviation).tolist()
    return a, mean, deviation

def antinormal(b, mean, deviation):  # 反标准化
    b = ((np.array(b) * deviation) + mean).tolist()
    return b

def collate_fn(batch):
    '''
    batch 解包
    '''
    data, label = zip(*batch)
    return data, label

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_data():
    '''
    generate npy dataset for loading in training process
    '''

    all_x = np.load(opt.bin_file,allow_pickle=True)
    all_v = np.load(opt.data_file,allow_pickle=True)

    #all_v = (np.array(all_v)/max_number).tolist()
    return all_x, all_v

def train(**kwargs):
    '''
    训练
    '''

    opt.parse(kwargs)  # 解析命令行参数

    setup_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)


    '''
    net = Net(opt)
    if opt.use_gpu:
        net.cuda()
    '''

    start = time.clock()
    print(f"{now()}: loading data")
    all_data, all_v = load_data()
    dictionary = {}
    #for stage_number in range(opt.stage[0]):
    stage_bias = [0]*opt.number
    minnum = np.load(opt.standardization)
    for number in range(opt.number):
    #for stage_number in [90]:

        # 对该网络归一化
        minn = minnum[number]
        all_v[number] = (np.array(all_v[number]) - minn).tolist()
        minx = min(all_data[number])[0]
        all_data[number] = (np.array(all_data[number]) - minx).tolist()
        # 对该网络归一化
        net = Net2(opt)
        net.cuda()
        net.reset_para()
        print(f"NN {number} begin to be trained")
        x_train, x_test, v_train, v_test = train_test_split(all_data[number], all_v[number], test_size=opt.test_size)
        train_data = Data(x_train, v_train)
        test_data = Data(x_test, v_test)
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

        print(f"{now()}, train data ready, train data size: {len(train_data)}, test data_size: {len(test_data)}, lr_rate: {opt.lr}, epochs: {opt.epochs}")
        loss_func = nn.MSELoss()  # 定义损失函数
        #loss_func = My_loss()  # 自定义损失函数
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

        acc_max = 1000000000000
        check = False
        for epoch in range(opt.epochs):
            net.train()
            scheduler.step()
            total_loss = 0.0
            for step, batch_data in enumerate(train_loader):
                x, v = batch_data
                v = torch.FloatTensor(v)
                x = torch.LongTensor(x)
                #print(x)
                #print(v)
                if opt.use_gpu:
                    v = v.cuda()
                    x = x.cuda()

                optimizer.zero_grad()
                output = net(x)
                loss = loss_func(output, v)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            #如果数据不全用来训练
            if opt.test_size != 0:
                train_bias, train_acc = test(net, train_loader)
                test_bias, test_acc = test(net, test_loader)
                print(f"{now()}: NN {number}, epoch {epoch}, Loss: {total_loss/len(train_data)}, train_acc= {train_acc}, test_acc={test_acc}.")
                if train_acc < acc_max:
                    stage_bias[number] = train_bias+test_bias
                    acc_max = train_acc
                    torch.save(net.state_dict(), opt.model_path+'net_{}_bk.pkl'.format(number))
                    print("\tnet_{}, epoch_{}, model updated, saved.".format(number, epoch))
                if acc_max <= 512:
                    print(f"{now()}, net:{number}, epoch:{epoch}, acc<=512, stop")
                    generate_dict(train_loader, number, dictionary, minn, minx)
                    generate_dict(test_loader, number, dictionary, minn, minx)
                    check = True
                    break

            #如果数据全用来训练的话
            else:
                bias, train_acc = test(net, train_loader)
                print(f"{now()}: NN {number}, epoch {epoch}, Loss: {total_loss/len(train_data)}, train_acc= {train_acc}.")
                if train_acc < acc_max:
                    stage_bias[number] = bias
                    acc_max = train_acc
                    torch.save(net.state_dict(), opt.model_path+'net_{}_bk.pkl'.format(number))
                    print("\tnet_{}, epoch_{}, model updated, saved.".format(number, epoch))
                if acc_max <= 512:
                    print(f"{now()}, net:{number}, epoch:{epoch}, acc<=512, stop")
                    generate_dict(train_loader, number, dictionary, minn, minx)
                    check = True
                    break
        if check==False:
            if opt.test_size != 0:
                generate_dict(train_loader, number, dictionary, minn, minx)
                generate_dict(test_loader, number, dictionary, minn, minx)
            else:
                generate_dict(train_loader, number, dictionary, minn, minx)

    print(f"totalbias:{sum(stage_bias)} totalnum:{opt.max_number}")
    print(f"average bias of is {sum(stage_bias) / opt.max_number}")

    np.save(opt.dic_path+'net_bk.npy', dictionary)
    print("dictionary saved")
    #print(dictionary)
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
    print(f"{now()}, {opt.number} nets' training stop.")


def test(net, test_loader):
    acc = 0
    num = 0
    net.eval()
    with torch.no_grad():
        for index, data in enumerate(test_loader):
            x, v = data
            num += len(v)
            v = torch.FloatTensor(v)
            x = torch.LongTensor(x)
            if opt.use_gpu:
                v = v.cuda()
                x = x.cuda()

            output = net(x)

            predict = torch.abs(output - v)

            acc += torch.sum(predict).item()
    net.train()

    return acc, (acc/num)

def generate_dict(test_loader, number, dictionary, minn ,minx):
    #首先加载最优网络
    net = torch.load("./models/structure/model.pkl")
    net.load_state_dict(torch.load(opt.model_path+"net_{}_bk.pkl".format(number)))
    net.eval()
    with torch.no_grad():
        for index, data in enumerate(test_loader):
            x, v = data
            v = torch.FloatTensor(v)
            x = torch.LongTensor(x)
            if opt.use_gpu:
                v = v.cuda()
                x = x.cuda()

            output = net(x)

            predict = torch.abs(output - v)
            #print(type(minn))
            #print(type(minx))
            for i in (predict > 1024).nonzero():
                dictionary[x[i[0].item()][0].item()+minx] = v[i[0].item()][0].item()+minn  # 创建字典需要根据维度修改

    print("number_{}_dictionary saved".format(number))


if __name__ == "__main__":
    import fire
    fire.Fire()
