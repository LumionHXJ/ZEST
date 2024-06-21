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
from config import hparams
import pickle5 as pickle
import ast
from dataset import create_dataset
from model import PitchModel
import yaml

mode = 'emospk'

with open('code/config.yml', 'r') as file:
    config = yaml.safe_load(file)

torch.set_printoptions(profile="full")
#Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
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

def getemolabel(file_name):
    file_name = int(file_name[5:-4])
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


def get_f0():
    os.makedirs(config['DSDT']['pred_F0']+f"_{mode}", exist_ok=True)
    test_loader = create_dataset("test", 1)
    model = PitchModel().to(device)
    model.load_state_dict(torch.load(config['F0']['checkpoint'], map_location=device))
    model.eval()
    sources = ["0011_000021.wav", "0012_000022.wav", "0013_000025.wav",
               "0014_000032.wav", "0015_000034.wav", "0016_000035.wav",
               "0017_000038.wav", "0018_000043.wav", "0019_000023.wav",
               "0020_000047.wav"]

    with torch.no_grad():
        for source in sources:
            for i, data in enumerate(test_loader):
                mask ,tokens, f0_trg = torch.tensor(data["mask"]).to(device),\
                    torch.tensor(data["hubert"]).to(device),\
                    torch.tensor(data["f0"]).to(device)
                speaker = torch.tensor(data["speaker"]).to(device)
                emotion = torch.tensor(data['emotion']).to(device)
            
                names = data["names"] 
                if names[0] == source:
                    source_name = names[0].replace(".wav", "")
                    tokens_s, mask_s, speaker_s, emotion_s = tokens, mask, speaker, emotion

            for i, data in enumerate(test_loader):
                mask ,tokens, f0_trg = torch.tensor(data["mask"]).to(device),\
                    torch.tensor(data["hubert"]).to(device),\
                    torch.tensor(data["f0"]).to(device)
                speaker = torch.tensor(data["speaker"]).to(device)
                emotion = torch.tensor(data['emotion']).to(device)
                speaker_t, emotion_t = speaker, emotion
                names = data["names"] 
                speaker = source[:5]
                if speaker not in names[0] and getemolabel(names[0]) > 0 and (int(names[0][5:11]) - int(source[5:11]))%350 != 0:
                    if mode == "emo":
                        pitch_pred, _, = model(tokens_s, speaker_s, emotion_t, mask_s) # 替换了audio
                    elif mode == "spk":
                        pitch_pred, _, = model(tokens_s, speaker_t, emotion_s, mask_s) # 替换了audio
                    elif mode == "emospk":
                        pitch_pred, _, = model(tokens_s, speaker_t, emotion_t, mask_s) # 替换了audio
                    # pitch_pred = torch.exp(pitch_pred) - 1
                    final_name = source_name + names[0]
                    final_name = final_name.replace(".wav", ".npy")
                    np.save(os.path.join(config['DSDT']['pred_F0']+f"_{mode}", final_name), pitch_pred[0, :].cpu().detach().numpy()) 

if __name__ == "__main__":
    get_f0()