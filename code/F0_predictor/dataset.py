import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import random
import torch.nn.functional as F
import pickle5 as pickle
import ast
import yaml

with open('code/config.yml', 'r') as file:
    config = yaml.safe_load(file)


class MyDataset(Dataset):
    def __init__(self, folder, token_file):
        self.folder = folder
        wav_files = os.listdir(folder)
        wav_files = [x for x in wav_files if ".wav" in x]
        self.wav_files = wav_files
        self.sr = 16000
        self.tokens = {}
        with open(config['F0']['ground_truth_f0'], 'rb') as handle:
            self.f0_feat = pickle.load(handle)
        with open(token_file) as f:
            lines = f.readlines()
            for l in lines:
                d = ast.literal_eval(l)
                name, tokens = d["audio"], d["hubert"]
                tokens_l = tokens.split(" ")
                self.tokens[name.split(os.sep)[-1]] = np.array(tokens_l).astype(int)

    def __len__(self):
        return len(self.wav_files) 

    def getfeat(self, file_name):
        speaker_feature = np.load(os.path.join(config['EASE']['embedding'], file_name.replace(".wav", ".npy")))
        emotion_feature = np.load(os.path.join(config['SACE']['embedding'], file_name.replace(".wav", ".npy")))

        return speaker_feature, emotion_feature
        
    def __getitem__(self, audio_ind):
        tokens = self.tokens[self.wav_files[audio_ind]]
        speaker_feat, emotion_feat = self.getfeat(self.wav_files[audio_ind])
        
        f0 = self.f0_feat[self.wav_files[audio_ind]]

        return f0, tokens, emotion_feat, speaker_feat, self.wav_files[audio_ind]


def custom_collate(data):
    new_data = {"mask":[], "hubert":[], "f0":[], "speaker":[], "emotion":[], "names":[]}
    max_len_f0, max_len_hubert, max_len_emo = 0, 0, 0
    for ind in range(len(data)):
        max_len_f0 = max(data[ind][0].shape[-1], max_len_f0)
        max_len_hubert = max(data[ind][1].shape[-1], max_len_hubert)
        max_len_emo = max(data[ind][2].shape[0], max_len_emo)
    for i in range(len(data)):
        emo_feat = np.concatenate((data[i][2], 
                                   np.zeros((max_len_emo-data[i][2].shape[0],
                                             config['SACE']['hidden_channel']))), 0)
        f0_feat = np.concatenate((data[i][0], np.zeros((max_len_f0-data[i][0].shape[-1]))), -1)
        mask = data[i][1].shape[-1]
        hubert_feat = np.concatenate((data[i][1], 100*np.ones((max_len_f0-data[i][1].shape[-1]))), -1)
        speaker_feat = data[i][3]
        names = data[i][4]
        new_data["f0"].append(f0_feat)
        new_data["mask"].append(torch.tensor(mask))
        new_data["hubert"].append(hubert_feat)
        new_data["speaker"].append(speaker_feat)
        new_data["names"].append(names)
        new_data['emotion'].append(emo_feat)
    new_data["mask"] = np.array(new_data["mask"])
    new_data["hubert"] = np.array(new_data["hubert"])
    new_data["f0"] = np.array(new_data["f0"])
    new_data["speaker"] = np.array(new_data["speaker"], dtype=np.float32)
    new_data["emotion"] = np.array(new_data["emotion"], dtype=np.float32)
    return new_data

def create_dataset_ddp(mode, bs=32):
    language = config['language']
    token_file = config['HuBERT'][language][mode]
    folder = os.path.join(config['dataset']['data_root'], config['dataset'][language]['wav'][mode])
    dataset = MyDataset(folder, token_file)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True if mode == 'Train' else False,
                    drop_last=False,
                    collate_fn=custom_collate,
                    sampler=sampler)
    return loader, sampler

def create_dataset(mode, bs=32):
    language = config['language']
    token_file = config['HuBERT'][language][mode]
    folder = os.path.join(config['dataset']['data_root'], config['dataset'][language]['wav'][mode])
    dataset = MyDataset(folder, token_file)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True if mode == 'Train' else False,
                    drop_last=False,
                    collate_fn=custom_collate)
    return loader