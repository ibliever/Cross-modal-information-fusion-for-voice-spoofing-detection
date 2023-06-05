# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from model import getmodel
import pickle
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    model = getmodel()
    #print(model.model_summary())
    train_datasets = pd.read_csv("ASVspoof2019.PA.cm.train.trn.txt", sep=' ', header=None,
                                 names=["SPEAKER_ID", "AUDIO_FILE_NAME", "SYSTEM_ID", "-", "KEY"])
    eval_datasets = pd.read_csv("ASVspoof2019.PA.cm.eval.trl.txt", sep=' ', header=None,
                                names=["SPEAKER_ID", "AUDIO_FILE_NAME", "SYSTEM_ID", "-", "KEY"])
    dev_datasets = pd.read_csv("ASVspoof2019.PA.cm.dev.trl.txt", sep=' ', header=None,
                               names=["SPEAKER_ID", "AUDIO_FILE_NAME", "SYSTEM_ID", "-", "KEY"])
    train_file = train_datasets.AUDIO_FILE_NAME
    train_label = train_datasets.KEY
    eval_file = eval_datasets.AUDIO_FILE_NAME
    eval_label = eval_datasets.KEY
    dev_file = dev_datasets.AUDIO_FILE_NAME
    dev_label = dev_datasets.KEY

    train_path = "./data/spect/train/"
    dev_path = "./data/spect/dev/"
    eval_path = "./data/spect/eval/"

    for i, file in enumerate(train_file):
        f = open(train_path + file + ".pkl", 'rb')
        data = pickle.load(f)
        data = np.reshape(data, newshape=(1, 598, 257, 2))
        y = model.predict(data)
        path = 'extract_pre/face/train/' + file + '.pkl'
        with open(path, 'ab') as f:
            pickle.dump(y, f)

    for i, file in enumerate(dev_file):
        f = open(dev_path + file + ".pkl", 'rb')
        data = pickle.load(f)
        data = np.reshape(data, newshape=(1,598,257,2))
        y = model.predict(data)
        path = 'extract_pre/face/dev/' + file + '.pkl'
        with open(path, 'ab') as f:
            pickle.dump(y, f)

    for i, file in enumerate(eval_file):
        f = open(eval_path + file + ".pkl", 'rb')
        data = pickle.load(f)
        data = np.reshape(data, newshape=(1, 598, 257, 2))
        y = model.predict(data)
        path = 'extract_pre/face/eval/' + file + '.pkl'
        with open(path, 'ab') as f:
            pickle.dump(y, f)