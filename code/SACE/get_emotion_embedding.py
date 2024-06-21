from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
import torchaudio
import numpy as np
from tqdm import tqdm
import yaml
import torch

with open('code/config.yml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained(config['SACE']['pretrained'])
wav2vec = Wav2Vec2ForCTC.from_pretrained(config['SACE']['pretrained'], output_hidden_states=True).to(device)

language = config['language']
for mode in ['train', 'val', 'test']:
    folder = os.path.join(config['dataset']['data_root'], config['dataset'][language]['wav'][mode])
    target_folder = os.path.join(config['dataset']['data_root'], config['dataset'][language]['wav2vec2'][mode])
    os.makedirs(target_folder, exist_ok=True)
    wav_files = os.listdir(folder)
    wav_files = [x for x in wav_files if ".wav" in x]
    wav_files = [x for x in wav_files if ".npy" not in x]

    for i, wav_file in enumerate(tqdm(wav_files)):
        sig, sr = torchaudio.load(os.path.join(folder, wav_file))
        inputs = processor(sig, sampling_rate=sr, return_tensors="pt") 
        h = wav2vec(inputs['input_values'].squeeze(0).to(device))['hidden_states'] 
        h = sum(list(h)).squeeze(0) # (?, 1024)
        target_file = os.path.join(target_folder, wav_file.replace(".wav", ".npy"))
        np.save(target_file, h.cpu().detach().numpy())