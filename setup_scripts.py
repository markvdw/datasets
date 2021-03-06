# Download and process all datasets
from __future__ import print_function

import os
import sys
import zipfile

import numpy as np
import numpy.random as rnd
import pandas as pd
import requests
from scipy.io import loadmat, savemat

basepath = os.path.dirname(os.path.realpath(__file__))

datasets_store_dir = os.path.join(basepath, ".")
download_target_folder = os.path.join(basepath, 'raw_download/')
process_temp_folder = os.path.join(basepath, 'proc_temp/')

required_directories = [datasets_store_dir, download_target_folder, process_temp_folder]

download_urls = {
    "YearPredMsd": "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip",
    "protein": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
    "kin40k": "http://www.tsc.uc3m.es/~miguel/code/datasets.zip",
    "house-electric": "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip",
    "naval": "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
    "snelson": "http://www.gatsby.ucl.ac.uk/~snelson/SPGP_dist.zip",
    "keggu": "https://archive.ics.uci.edu/ml/machine-learning-databases/00221/Reaction%20Network%20(Undirected).data",
    "kegg": "https://archive.ics.uci.edu/ml/machine-learning-databases/00220/Relation%20Network%20(Directed).data",
    "parkinsons": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
    "pumadyn32nm": "http://www.tsc.uc3m.es/~miguel/code/datasets.zip"}


def download_file(url, target_dir='.'):
    local_filename = "%s/%s" % (target_dir, url.split('/')[-1])
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # f.flush() commented by recommendation from J.F.Sebastian
    return local_filename


def save_splits(aX, aY, splits, name, descr, url):
    edge = int(np.floor(len(aX) * 0.9))
    for split in range(splits):
        perm = rnd.permutation(len(aX))
        trX = aX[perm[:edge], :]
        trY = aY[perm[:edge], :]
        teX = aX[perm[edge:], :]
        teY = aY[perm[edge:], :]
        savemat("%s/%s%i.mat" % (datasets_store_dir, name, split),
                {'X': trX, 'Y': trY, 'tX': teX, 'tY': teY, 'name': name,
                 'description': descr,
                 "url": url})


def process_yearpredmsd():
    print("Processing YearPredictionMSD.txt.zip")
    zf = zipfile.ZipFile("%s/YearPredictionMSD.txt.zip" % download_target_folder)
    zf.extractall(process_temp_folder)
    data = np.loadtxt("%s/YearPredictionMSD.txt" % process_temp_folder, delimiter=',')
    tr = data[:463715, :]
    te = data[463715:, :]
    assert te.shape[0] == 51630, "Testing dataset is the wrong size."

    trX = tr[:, 1:]
    trY = tr[:, 0, None]
    teX = te[:, 1:]
    teY = te[:, 0, None]

    savemat("%s/year-prediction-msd" % datasets_store_dir,
            {'X': trX, 'Y': trY, 'tX': teX, 'tY': teY, 'name': "yearpredmillionsong",
             'description': "Year prediction, Million Song Dataset",
             "url": download_urls["YearPredMsd"]})


def process_kin40k():
    print("Processing kin40k")
    zf = zipfile.ZipFile("%s/datasets.zip" % download_target_folder)
    zf.extractall(process_temp_folder)
    mat = loadmat("%s/datasets/kin40k.mat" % process_temp_folder)
    savemat("%s/kin40k" % datasets_store_dir,
            {'X': mat["X_tr"], 'Y': mat["T_tr"], 'tX': mat["X_tst"], 'tY': mat["T_tst"],
             "name": "kin40k", 'description': "kin40k, classic dataset", "url": download_urls["kin40k"]})

    aX = np.vstack((mat["X_tr"], mat["X_tst"]))
    aY = np.vstack((mat["T_tr"], mat["T_tst"]))
    save_splits(aX, aY, 5, "kin40k-30k", "kin40k, classic dataset, large training split", download_urls['kin40k'])


def process_pumadyn32nm():
    print("Processing pumadyn32nm")
    zf = zipfile.ZipFile("%s/datasets.zip" % download_target_folder)
    zf.extractall(process_temp_folder)
    mat = loadmat("%s/datasets/pumadyn32nm.mat" % process_temp_folder)
    m = np.mean(mat["X_tr"], 0)
    savemat("%s/pumadyn32nm" % datasets_store_dir,
            {'X': mat["X_tr"] - m[None, :], 'Y': mat["T_tr"], 'tX': mat["X_tst"] - m[None, :], 'tY': mat["T_tst"],
             "name": "pumadyn32nm", 'description': "pumadyn32nm, classic dataset", "url": download_urls["pumadyn32nm"]})


def process_household_electric():
    # Unfinished
    print("Processing household electric")
    zf = zipfile.ZipFile("%s/household_power_consumption.zip" % download_target_folder)
    zf.extractall(process_temp_folder)
    df = pd.read_csv("%s/household_power_consumption.txt" % process_temp_folder, delimiter=';', na_values='?',
                     parse_dates=[[0, 1]], infer_datetime_format=True)
    df = df[df.isnull().sum(axis=1) == 0]
    month = df.Date_Time.dt.month
    return df


def process_protein():
    print("Processing protein")
    data = np.loadtxt("%s/CASP.csv" % download_target_folder, delimiter=',', skiprows=1)
    aX = data[:, 1:]
    aX = (aX - np.mean(aX, 0)[None, :]) / np.std(aX, 0)[None, :]
    aY = np.log(data[:, 0, None] + 1)
    aY = (aY - np.mean(aY)) / np.std(aY)
    save_splits(aX, aY, 5, "protein", "Protein tertiary structure", download_urls["protein"])


def process_naval():
    print("Processing naval")
    path = "%s/naval/" % process_temp_folder
    if not os.path.exists(path):
        os.mkdir(path)
    zf = zipfile.ZipFile("%s/UCI%%20CBM%%20Dataset.zip" % download_target_folder)
    zf.extractall(path)
    data = np.loadtxt("%s/UCI CBM Dataset/data.txt" % path)
    aX = data[:, :16]
    aY = data[:, 16, None]
    edge = int(np.floor(len(aX) * 0.9))
    for split in range(5):
        perm = rnd.permutation(len(aX))
        trX = aX[perm[:edge], :]
        trY = aY[perm[:edge], :]
        teX = aX[perm[edge:], :]
        teY = aY[perm[edge:], :]
        savemat("%s/naval%i.mat" % (datasets_store_dir, split),
                {'X': trX, 'Y': trY, 'tX': teX, 'tY': teY, 'name': "naval",
                 'description': "Condition Based Maintenance of Naval Propulsion Plants",
                 "url": download_urls["naval"]})


def process_snelson():
    print("Processing snelson")
    zf = zipfile.ZipFile("%s/SPGP_dist.zip" % download_target_folder)
    zf.extractall(process_temp_folder)
    X = np.loadtxt('%s/SPGP_dist/train_inputs' % process_temp_folder)[:, None]
    Y = np.loadtxt('%s/SPGP_dist/train_outputs' % process_temp_folder)[:, None]
    savemat("%s/snelson1d.mat" % datasets_store_dir,
            {'X': X, 'Y': Y, 'tX': X, 'tY': Y, 'name': "snelson1d",
             'description': "Snelson 1d classic dataset",
             "url": download_urls["snelson"]})


def process_mnist():
    print("Processing MNIST")
    from tensorflow.examples.tutorials.mnist import input_data
    data = input_data.read_data_sets(download_target_folder, one_hot=False)
    savemat("%s/mnist.mat" % datasets_store_dir,
            {'X': data.train.images,
             'Y': data.train.labels.astype('float').T,
             'tX': data.test.images,
             'tY': data.test.labels.astype('float').T,
             'name': 'mnist',
             'url': "TensorFlow tutorial",
             'description': "MNIST: Classic classification dataset"
             })

    savemat("%s/mnist-bin.mat" % datasets_store_dir,
            {'X': data.train.images,
             'Y': data.train.labels.astype('float').T % 2,
             'tX': data.test.images,
             'tY': data.test.labels.astype('float').T % 2,
             'name': 'mnist',
             'url': "TensorFlow tutorial",
             'description': "Binaries MNIST. Even: 0, Odd 1"
             })


def process_kegg():
    print("Processing KEGG")
    d = pd.read_csv("%s/Reaction%%20Network%%20(Undirected).data" % download_target_folder, delimiter=',',
                    na_values='?', header=None)
    d = d[d.isnull().sum(1) == 0]
    d = d[d.iloc[:, 21] <= 1]
    d = d.iloc[:, np.hstack([np.arange(1, 10), np.arange(11, 29)])]
    X = np.array(d.iloc[:, :-1]).astype('float')
    Y = np.log(np.array(d.iloc[:, -1]).astype('float')[:, None])

    X = X - np.mean(X, 0)[None, :]
    X = X / np.std(X, 0)[None, :]

    Y -= np.mean(Y)
    Y /= np.std(Y)

    save_splits(X, Y, 3, "keggu", "KEGGU", download_urls["keggu"])


def process_parkinsons():
    df = pd.read_csv("%s/parkinsons_updrs.data" % download_target_folder, delimiter=",")
    y = np.array(df.iloc[:, 5][:, None])
    x = np.array(df.iloc[:, np.hstack([np.arange(0, 4), np.arange(6, 22)])])

    X = (x - np.mean(x, 0)[None, :]) / np.std(x, 0)[None, :]
    Y = (y - np.mean(y)) / np.std(y)

    save_splits(X, Y, 2, "parkinsons", "parkinsons", download_urls["parkinsons"])


def setup_datasets():
    for dir in required_directories:
        if not os.path.exists(dir):
            os.mkdir(dir)

    print("Downloading files... This may take a while...")
    for url in download_urls.values():
        local_filename = url.split("/")[-1]
        print("%-*s" % (40, local_filename), end="... ")
        sys.stdout.flush()
        if not os.path.exists("%s/%s" % (download_target_folder, local_filename)):
            download_file(url, download_target_folder)
        else:
            print("Skipping.", end="")
        print("")
    print("")

    print("Processing downloaded files...")
    # process_yearpredmsd()
    process_kin40k()
    process_protein()
    # process_household_electric()
    process_naval()
    process_snelson()
    process_mnist()
    process_kegg()
    process_parkinsons()
    process_pumadyn32nm()
