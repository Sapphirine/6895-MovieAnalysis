# get audio file
import subprocess

# ffmpeg -i <infile> -ac 2 -f wav <outfile>
command = "ffmpeg -i ./Downfall.mp4 -ac 2 -f wav audio.wav"

subprocess.call(command, shell=True)


# import libraries
import glob
from python_speech_features.base import mfcc, fbank, logfbank, ssc
import scipy.io.wavfile as wav
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.svm import SVC
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent

sound = AudioSegment.from_wav("audio.wav")

# consider it silent if quieter than -40 dBFS
nonsilent = detect_nonsilent(sound, min_silence_len = 300, silence_thresh = -40)
for i, (start_i, end_i) in enumerate(nonsilent):
    sound[max(0, start_i - 100) : end_i + 100].export("./audio_chunk/chunk{0}.wav".format(i), format="wav")

clf2 = joblib.load('audio_SVC_model.pkl')
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

cepCount = 13 #no of MFCC coefficients
nfeatures = 7 #features per coefficient

emotions = ["anger", "boredom", "disgust", "fear", "happy", "sadness", "neutral"]


def audio_read(datafs):
    (data, fs) = wav.read(datafs)
    ceps = mfcc(fs, numcep = cepCount)
    feat2 = ssc(fs, samplerate = 16000, winlen = 0.025, winstep=0.01,
                nfilt = 26, nfft = 512, lowfreq = 0, highfreq = None, preemph=0.97)
    ls = []
    for i in range(ceps.shape[1]):
        temp = ceps[:,i]
        dtemp = np.gradient(temp)
        lfeatures  = [np.mean(temp), np.var(temp), np.amax(temp), np.amin(temp),
        np.var(dtemp), np.mean(temp[0:temp.shape[0]/2]), np.mean(temp[temp.shape[0]/2:temp.shape[0]])]
        temp2 = np.array(lfeatures)
        ls.append(temp2)

    ls2 = []
    for i in range(feat2.shape[1]):
        temp = feat2[:,i]
        dtemp = np.gradient(temp)
        lfeatures = [np.mean(temp), np.var(temp), np.amax(temp), np.amin(temp),
        np.var(dtemp), np.mean(temp[0:temp.shape[0]/2]), np.mean(temp[temp.shape[0]/2:temp.shape[0]])]
        temp2 = np.array(lfeatures)
        ls2.append(temp2)

    source = np.array(ls).flatten()
    source = np.append(source, np.array(ls2).flatten())
    return source

participants = glob.glob("./audio_chunk/*")
X = []
for data in participants:
    X.append(audio_read(data))
X = np.array(X)
X2 = scaler.fit_transform(X)
ind = clf2.predict(X2)

import csv
with open("audio_emotion.csv", "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for i, (start_i, end_i) in enumerate(nonsilent):
        writer.writerow((start_i, end_i, emotions[ind[i]]))

