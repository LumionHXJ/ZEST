import os
import torch
import torchaudio
from einops.layers.torch import Rearrange
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from tqdm import tqdm
import random
import torch.nn.functional as F
# from config import hparams
import pickle5 as pickle
import ast
# from pitch_attention_adv import create_dataset, PitchModel
from torch.autograd import Function
from emotion_classifier import SACE

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

model = torch.load('code/SACE/SACE.pth', map_location=device)
model.to(device)
model.eval()

def getemolabel(file_name):
    file_name = int(file_name[-10:-4])
    # file_name = int(file_name[5:11])
    if file_name <=350:
        return 0
    elif file_name > 350 and file_name <=700:
        return 1
    elif file_name > 700 and file_name <= 1050:
        return 2
    elif file_name > 1050 and file_name <= 1400:
        return 3
    else:
        return 4

def get_emo_out(loader, ground_truth):
    results = np.array([], dtype=np.int64)
    for i, data in enumerate(tqdm(loader)):
        inputs, labels = torch.tensor(data["audio"]), \
                         torch.tensor(data["labels"])

        if ground_truth:
            results = np.concatenate((results, labels.numpy()))
        else:
            aud, alpha = inputs, 1.0
            with torch.no_grad():
                inputs = model.processor(aud, sampling_rate=16000, return_tensors="pt")
                emo_out, _, _, _ = model.encoder(inputs['input_values'], alpha)
                results = np.concatenate((results, np.argmax(emo_out.numpy(), axis=1)))
        break
    return results

def get_emo_out_test(folder, bs=72):
    preds, labels = np.array([], dtype=np.int64), []
    inputs, max_length, filenames = [], 0, os.listdir(folder)
    for i, filename in enumerate(tqdm(filenames)):
        # preds
        # inputs.append(torchaudio.load(os.path.join(folder, filename))[0].numpy()[0, :])
        inputs.append(np.load(os.path.join(folder+'_emo', filename.replace('.wav', '.npy'))))
        max_length = max(max_length, inputs[-1].shape[0])
        if (i+1) % bs == 0 or i == len(filenames)-1:
            for j in range(len(inputs)):
                inputs[j] = np.concatenate((inputs[j], np.zeros((max_length-inputs[j].shape[0], inputs[j].shape[1]))), 0)
            aud, alpha = torch.tensor(np.array(inputs)).to(device=device, dtype=torch.float32), 1.0
            with torch.no_grad():
                # inputs = model.processor(aud, sampling_rate=16000, return_tensors="pt")
                emo_out, _, _, _ = model(aud, alpha)
                preds = np.concatenate((preds, np.argmax(emo_out.cpu().numpy(), axis=1)))
            inputs, max_length = [], 0
        # labels
        labels.append(getemolabel(filename))
        # if i >= 1000:
        #     break
    return preds, np.array(labels[:len(preds)], dtype=np.int64)

# val_loader = create_dataset("val")
# label = get_emo_out(val_loader, ground_truth=True)
# print(len(label), label)

# test_loader = create_dataset("test")
# pred = get_emo_out(test_loader, ground_truth=False)
# print(len(pred), pred)

# acc = np.sum(pred==label) / len(label)
# print('acc:', acc)

preds, labels = get_emo_out_test('DSDT/reconstruct_emospk')

acc = np.sum(preds==labels) / len(labels)
print('acc:', acc)
