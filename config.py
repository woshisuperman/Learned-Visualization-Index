# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

class Config():
    '''
    Config Parameters
    '''


    emb_size = [5, 20]  # 每一个维度embedding size
    input_size = [1680]  # 输入维度, 可变多维
    #fc_nums = [50, 70, 201]
    fc_nums = [20, 1]  # 每层神经网络形状
    number = 19973


    data_file = "./trainingset/lat/trainingset_V.npy"  # 数据集路径
    bin_file = "./trainingset/lat/trainingset_X.npy"  # 数据集路径
    standardization = "./models/lat/min.npy"  # 数据集路径
    model_path = "./models/lat/"
    dic_path = "./models/lat/"
    use_gpu = True
    gpu_id = 0

    seed = 2019
    max_number = 2**25
    dropout = 0.0
    epochs = 120

    test_size = 0.1
    lr = 0.05
    weight_decay = 0.0  # 1e-4
    batch_size = 16
    step_size = 5
    gamma = 0.7

    net_num = 1 #要训练的网络数量


def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


Config.parse = parse
opt = Config()
