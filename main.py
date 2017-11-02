import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.signal as sig
import librosa
import librosa.display as disp

pathToDCASEapps = "/home/franz/Documents/DCASE2017/DCASE2017-baseline-system/applications/data/TUT-acoustic-scenes-2017-development/"
#pathToDCASEapps = "/Users/franzanders/Documents/Promotion/DCASE2017/DCASE2017-baseline-system/applications/data/TUT-acoustic-scenes-2017-development/"
pathToMetaFile = pathToDCASEapps + "meta.txt"

_nfft = 1024
win_length_s = 0.04
hop_length_s = win_length_s / 2


def main():

    filenames = getFileNames(pathToMetaFile)

    for fn in filenames:
        print(fn[0])

        sr, y = loadAudioFile(fn[0])
        MFCCs, MFCCs_delta, MFCCs_deltadelta, P = extractMfccFeatures(_y=y, _sr=sr, _nfft=_nfft,
                                                                      _win_length_n=int(win_length_s*sr), _hop_length_n=int(hop_length_s * sr),
                                                                      _n_mels=40,_fmax=10000, _n_mfcc=20)
        feats_out = np.concatenate((MFCCs, MFCCs_delta, MFCCs_deltadelta), axis=0)
        #visualize(MFCCs, P, int(hop_length_s * sr), sr, y)

        np.savetxt("audioFeatures/"+fn[0][6:]+".txt", feats_out, delimiter=' ')  # X is an array


def loadAudioFile(filename):
    y, sr = librosa.load(pathToDCASEapps + filename)
    y = librosa.core.to_mono(y=y)
    return sr, y


def visualize(MFCCs, P, hop_length_n, sr, y):
    plt.figure()
    plt.subplot(3, 1, 1)
    disp.waveplot(y=y, sr=sr)
    plt.subplot(3, 1, 2)
    disp.specshow(librosa.core.power_to_db(P, ref=np.max), sr=sr, hop_length=hop_length_n,
                  x_axis="time", y_axis="linear")
    plt.subplot(3, 1, 3)
    disp.specshow(MFCCs, sr=sr)
    plt.show()


def extractMfccFeatures(_y, _sr, _nfft, _win_length_n, _hop_length_n, _n_mels, _fmax, _n_mfcc):

    S = librosa.core.stft(_y, n_fft=_nfft, hop_length=_hop_length_n, win_length=_win_length_n,
                          window=sig.hamming(_win_length_n), center=False)
    P = np.abs(S) ** 2
    M = librosa.feature.melspectrogram(S=P, sr=_sr, n_mels=_n_mels, fmax=_fmax)
    MFCCs = librosa.feature.mfcc(S=librosa.power_to_db(M), n_mfcc=_n_mfcc)
    MFCCs_delta = librosa.feature.delta(MFCCs, width=9, order=1)
    MFCCs_deltadelta = librosa.feature.delta(MFCCs, width=9, order=2)
    return MFCCs, MFCCs_delta, MFCCs_deltadelta, P


def getFileNames(pathToMetaFile):

    f = open(pathToMetaFile, 'rt', encoding="utf-8")
    reader = csv.reader(f, delimiter='\t')
    return [(row[0], row[1]) for row in reader]


main()