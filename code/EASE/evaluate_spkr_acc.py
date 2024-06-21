import os
import torch
import torchaudio
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import random
import torch.nn.functional as F
import pickle5 as pickle
import ast
from torch.autograd import Function
from speaker_classifier import SpeakerModel, create_dataset

torch.set_printoptions(profile="full")
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.autograd.set_detect_anomaly(True)
#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

# model = SpeakerModel()
model = torch.load('code/EASE/EASE.pth', map_location=device)
model.to(device)
model.eval()

speaker_dict = {}
for ind in range(11, 21):
    speaker_dict["00"+str(ind)] = ind-11

def getspkrlabel(file_name):
    # spkr_name = file_name[:4]
    spkr_name = file_name[11:15]
    spkr_label = speaker_dict[spkr_name]
    return spkr_label

def get_speaker_out(loader, ground_truth):
    results = np.array([], dtype=np.int64)
    for i, data in enumerate(tqdm(loader)):
        speaker_feat, labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
        if ground_truth:
            results = np.concatenate((results, labels.detach().cpu().numpy()))
        else:
            outputs, _, _ = model(speaker_feat)
            results = np.concatenate((results, np.argmax(outputs.detach().cpu().numpy(), axis=1)))
        # break
    return results

def get_speaker_out_test(folder, bs=72):
    preds, labels = np.array([], dtype=np.int64), []
    inputs, filenames = [], os.listdir(folder)
    for i, filename in enumerate(tqdm(filenames)):
        # preds
        inputs.append(np.load(os.path.join(folder+'_spkr', filename.replace(".wav", ".npy"))))
        if (i+1) % bs == 0 or i == len(filenames)-1:
            speaker_feat = torch.tensor(np.array(inputs)).to(device)
            with torch.no_grad():
                outputs, _, _ = model(speaker_feat)
                preds = np.concatenate((preds, np.argmax(outputs.cpu().numpy(), axis=1)))
            inputs = []
        # labels
        labels.append(getspkrlabel(filename))
        # if i >= 1000:
        #     break
    return preds, np.array(labels[:len(preds)], dtype=np.int64)

# val_loader = create_dataset("val")
# label = get_speaker_out(val_loader, ground_truth=True)
# print(len(label), label)

# test_loader = create_dataset("test")
# pred = get_speaker_out(test_loader, ground_truth=False)
# print(len(pred), pred)

# acc = np.sum(pred==label) / len(label)
# print('acc:', acc)

preds, labels = get_speaker_out_test('DSDT/reconstruct_emospk')

acc = np.sum(preds==labels) / len(labels)
print('acc:', acc)
