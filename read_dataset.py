import pandas as pd
import librosa
from os import listdir
from os.path import isfile, join
import numpy as np
import json
from Corpora import dict_corpora_func, vectorise_string

sample_rate = 16000
Audio_Path = 'gu-in-Test/Audios/'

data = pd.read_csv('gu-in-Test/transcription.csv',sep='\t' ,names = ['line'] , index_col = 0).to_dict()['line']
onlyfiles = [f for f in listdir(Audio_Path) if isfile(join(Audio_Path, f))]
dict_corpora = dict_corpora_func()
inputs = list()
labels = list()
for f in listdir(Audio_Path):
    if isfile(join(Audio_Path,f)):
        y,_ = librosa.load(join(Audio_Path, f), sr = sample_rate)
        file = librosa.feature.mfcc(y, sr = sample_rate , n_mfcc = 16)
        inputs.append(file.T.tolist())
        labels.append(vectorise_string(data[int(f.strip('.wav'))].replace(','," "), dict_corpora))

np.save("test_labels.npy", np.array(labels))
np.save("test_inputs.npy", np.array(inputs))

Audio_Path = 'gu-in-Train/Audios/'

data = pd.read_csv('gu-in-Train/transcription.csv',sep='\t' ,names = ['line'] , index_col = 0).to_dict()['line']
onlyfiles = [f for f in listdir(Audio_Path) if isfile(join(Audio_Path, f))]
dict_corpora = dict_corpora_func()
inputs = list()
labels = list()
for f in listdir(Audio_Path):
    if isfile(join(Audio_Path,f)):
        y,_ = librosa.load(join(Audio_Path, f), sr = sample_rate)
        file = librosa.feature.mfcc(y, sr = sample_rate , n_mfcc = 16)
        inputs.append(file.T.tolist())
        labels.append(vectorise_string(data[int(f.strip('.wav'))].replace(','," "), dict_corpora))

np.save("train_labels.npy", np.array(labels))
np.save("train_inputs.npy", np.array(inputs))
