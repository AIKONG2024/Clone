import numpy as np

a = np.array(range(1,11))
size = 5
def split_x(datasets, size):
    a = []
    for i in range(len(datasets) - size + 1):
        datas = datasets[i: size + i]
        a.append(datas)
    return np.array(a)


def split_3d_x(datasets, size):
    a = []
    for i in range(len(datasets) - size + 1):
        datas = datasets[i: size + i]
        a.append(datas)
    np_datas = np.array(a)
    # np_datas = np_datas.reshape(np_datas.shape[0],np_datas.shape[1], -1)
    np_datas = np.expand_dims(np_datas, axis=2)
    return np_datas

b = split_3d_x(a , size)

print(b.shape) #(6, 5, 1)