
import numpy as np
import data_util
import pandas as pd
import pickle as pkl

if __name__ == '__main__':
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

    train_face_path = './extract_pre/face/train'
    dev_face_path = './extract_pre/face/dev'
    eval_face_path = './extract_pre/face/eval'

    train_audio_path = './extract_pre/PA/lfcc_sedense/train'
    dev_audio_path = './extract_pre/PA/lfcc_sedense/dev'
    eval_audio_path = './extract_pre/PA/lfcc_sedense/eval'

    for file in train_file:
        face = pkl.load(open('extract_pre/face/train/'+file+'.pkl', 'rb'))
        audio = pkl.load(open('extract_pre/PA/lfcc_sedense/train/'+file+'.pkl', 'rb'))
        audio = np.reshape(audio, newshape=(1,128))
        fuse = np.concatenate([face, audio], axis=1)
        path = 'extract_pre/PA/fuse/train/' + file + '.pkl'
        with open(path, 'ab') as f:
            pkl.dump(fuse, f)

    for file in dev_file:
        face = pkl.load(open('extract_pre/face/dev/'+file+'.pkl', 'rb'))
        audio = pkl.load(open('extract_pre/PA/lfcc_sedense/dev/'+file+'.pkl', 'rb'))
        audio = np.reshape(audio, newshape=(1,128))
        fuse = np.concatenate([face, audio], axis=1)
        path = 'extract_pre/PA/fuse/dev/' + file + '.pkl'
        with open(path, 'ab') as f:
            pkl.dump(fuse, f)

    for file in eval_file:
        face = pkl.load(open('extract_pre/face/eval/'+file+'.pkl', 'rb'))
        audio = pkl.load(open('extract_pre/PA/lfcc_sedense/eval/'+file+'.pkl', 'rb'))
        audio = np.reshape(audio, newshape=(1,128))
        fuse = np.concatenate([face, audio], axis=1)
        path = 'extract_pre/PA/fuse/eval/' + file + '.pkl'
        with open(path, 'ab') as f:
            pkl.dump(fuse, f)