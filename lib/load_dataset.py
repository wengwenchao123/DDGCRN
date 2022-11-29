import os
import numpy as np
import pandas as pd

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD3':
        data_path = os.path.join('./data/PeMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD4':
        data_path = os.path.join('./data/PeMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('./data/PeMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PeMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7(L)':
        data_path = os.path.join('./data/PEMS07(L)/PEMS07L.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7(M)':
        data_path = os.path.join('./data/PEMS07(M)/V_228.csv')
        data = np.array(pd.read_csv(data_path,header=None))  #onley the first dimension, traffic flow data
    elif dataset == 'METR-LA':
        data_path = os.path.join('./data/METR-LA/METR.h5')
        data = pd.read_hdf(data_path)
    elif dataset == 'BJ':
        data_path = os.path.join('./data/BJ/BJ500.csv')
        data = np.array(pd.read_csv(data_path, header=0, index_col=0))
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

#
# data_path = os.path.join('../data/PeMS07/PEMS07.npz')
# data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
