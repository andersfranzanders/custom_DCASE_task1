import csv
import sys
import librosa
import librosa.display as disp
import numpy as np
import matplotlib
import scipy.signal as sig

pathToDCASEapps = "/home/franz/Documents/DCASE2017/DCASE2017-baseline-system/applications/data/TUT-acoustic-scenes-2017-development/"
pathToMetaFile = pathToDCASEapps + "meta.txt"

_nfft = 1024
win_length_s = 0.06
hop_length_s = win_length_s / 2


def main():

    filenames = getFileNames(pathToMetaFile)

    for fn in filenames[0:1]:
        print(fn[0])

        y, sr = librosa.load(pathToDCASEapps + fn[0])
        y = librosa.core.to_mono(y=y)
        hop_length_n = hop_length_s * sr
        win_length_n= win_length_s*sr
        print(hop_length_n)
        print(win_length_n)
        Y = librosa.core.stft(y, n_fft=_nfft, hop_length=hop_length_n, win_length= win_length_n, window=sig.hamming(_nfft), center=False)
        P = np.abs(Y)**2

        print(sr)


        disp.waveplot(y=y, sr=sr)
        disp.specshow(librosa.core.power_to_db(P, ref=np.max), sr=sr, hop_length=hop_length_s*sr,
                      x_axis="time", y_axis="linear")

    #matplotlib.pyplot.show()

def getFileNames(pathToMetaFile):

    f = open(pathToMetaFile, 'rb')
    reader = csv.reader(f, delimiter='\t')
    return [(row[0], row[1]) for row in reader]


main()