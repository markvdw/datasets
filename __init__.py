import glob
import os
import numpy as np
import scipy.io
from .setup_scripts import setup_datasets


class Dataset(object):
    def __init__(self, X, Y, Xt, Yt, description, url, name, filename):
        self.X = np.ascontiguousarray(X)
        self.Y = np.ascontiguousarray(Y)
        self.Xt = np.ascontiguousarray(Xt)
        self.Yt = np.ascontiguousarray(Yt)
        self.description = description
        self.url = url
        self.name = name
        self.filename = filename

    def subset(self, N):
        perm = np.random.permutation(len(self.X))
        return Dataset(self.X[perm[:N]], self.Y[perm[:N]], self.Xt, self.Yt, self.description, self.url, self.name,
                       None)

    def __str__(self):
        return ("%-*s %-*s %s\n" % (23, self.filename.split('/')[-1], 20, str(self.name[0]), str(self.description[0])) +
                "D: %-*iN: %-*iNt: %i" % (21, self.X.shape[1], 18, self.X.shape[0], self.Xt.shape[0]))

    def __jug_hash__(self):
        from jug.hash import hash_one
        return hash_one({'type': 'Dataset', 'data': self.name})


def array_stripper(*args):
    args = list(args)
    for i in range(len(args)):
        if type(args[i][0]) is np.str_:
            args[i] = str(args[i][0])
    return tuple(args)


def load_data(filename):
    datasets = glob.glob(os.path.join(os.path.dirname(__file__), "*.mat"))
    dataset_names = [os.path.splitext(os.path.basename(x))[0] for x in datasets]

    # Feature: Allow the dataset name to simply be given
    if not os.path.exists(filename):
        filename_proc = os.path.splitext(os.path.basename(filename))[0]
        if filename_proc in dataset_names:
            filename = datasets[dataset_names.index(filename_proc)]
        else:
            raise ValueError("Dataset %s unknown..." % filename)

    loaded = scipy.io.loadmat(filename)
    d = Dataset(*array_stripper(*[loaded[k] for k in ['X', 'Y', 'tX', 'tY', 'description', 'url', 'name']] + [filename]))
    return d


def print_stats():
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "./*.mat")))
    for dataset_file in files:
        loaded = scipy.io.loadmat(dataset_file)
        Y, X, Yte, Xte = [loaded[k] for k in ('Y', 'X', 'tY', 'tX')]
        dataset_file_name = os.path.basename(dataset_file)
        print("%-*sD: %-*iN: %-*iNte: %i" % (35, dataset_file_name, 10, X.shape[1], 10, X.shape[0], Xte.shape[0]))


def print_descr():
    files = glob.glob("./*.mat")
    for dataset_file in files:
        loaded = scipy.io.loadmat(dataset_file)
        name, descr = [loaded[k] for k in ('name', 'description')]
        print("%-*s %-*s %s" % (23, dataset_file.split('/')[-1], 20, str(name[0]), str(descr[0])))


if __name__ == "__main__":
    print_stats()
