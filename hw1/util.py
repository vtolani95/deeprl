import numpy as np
import pdb

def load(envname):
    data = np.load('./rollout_data/%s.npy'%(envname))[()]
    x, y =  data['observations'], data['actions']
    y = reshape_actions(y)
    x, y = shuffle_data(x, y)
    x_train, x_cv, y_train, y_cv = split_data(x, y)
    return x_train, x_cv, y_train, y_cv

def shuffle_data(x, y):
    assert len(x) == len(y)
    indices = np.random.permutation(len(x))
    x, y = x[indices], y[indices]
    return x, y

def split_data(x, y):
    index = int(.8*len(x))
    return x[:index], x[index:], y[:index], y[index:]

def reshape_actions(y):
    dim1 = len(y)
    dim2 = max(y.shape[1:])
    return np.reshape(y, (dim1, dim2))

def green(val):
  return "\033[92m%s\033[0m"%(str(val))
