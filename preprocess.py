
import pandas as pd
from speaker import *

def txt2csv(filename):
    dataset = pd.read_csv(filename, sep=' ', header=0)
    dataset.to_csv('ASVspoof2019.LA.cm.dev.trl.csv', index=False)

def preprocess(csv_name):
    data = pd.read_csv(csv_name, header=None, names=["SPEAKER_ID", "AUDIO_FILE_NAME", "SYSTEM_ID", "-", "KEY"])
    files = data.AUDIO_FILE_NAME
    sb = Speaker(16000, 6, True, 400, 160, 512, 0.3, False, "data/audios/eval/", "data/audio_spectrograms/eval/", "data/flacfiles/eval/")
    for file in files:
        print(file)
        sb.extract_wav(file)
    #sb.extract_wav("LA_E_6809846")

if __name__ == '__main__':
    preprocess("ASVspoof2019.LA.cm.eval.trl.csv")