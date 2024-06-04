import os
import torch
import torchaudio
from einops.layers.torch import Rearrange
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import logging
import numpy as np
import random
from tqdm import tqdm
import random
from dataset import create_dataset
import yaml
from model import PitchModel

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

def get_f0():
    os.makedirs(config['F0']['contour'], exist_ok=True)
    
    model = PitchModel().to(device)
    model.load_state_dict(torch.load(config['F0']['checkpoint'], map_location=device))
    model.eval()
    loader = create_dataset("test", 1)
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            mask ,tokens, f0_trg = torch.tensor(data["mask"]).to(device),\
                torch.tensor(data["hubert"]).to(device),\
                torch.tensor(data["f0"]).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            emotion = torch.tensor(data['emotion']).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            names = data["names"] 
            for ind in range(len(names)):
                target_file_name = names[ind].split(os.sep)[-1].replace("wav", "npy")
                pitch_pred, _ = model(tokens, speaker, emotion, mask)
                # pitch_pred = torch.exp(pitch_pred) - 1
                np.save(os.path.join(config['F0']['contour'], target_file_name), 
                        pitch_pred[ind, :].cpu().detach().numpy()) 
    loader = create_dataset("train", 1)
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            mask ,tokens, f0_trg = torch.tensor(data["mask"]).to(device),\
                torch.tensor(data["hubert"]).to(device),\
                torch.tensor(data["f0"]).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            emotion = torch.tensor(data['emotion']).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            names = data["names"] 
            for ind in range(len(names)):
                target_file_name = names[ind].split(os.sep)[-1].replace("wav", "npy")
                pitch_pred, _ = model(tokens, speaker, emotion, mask)
                # pitch_pred = torch.exp(pitch_pred) - 1
                np.save(os.path.join(config['F0']['contour'], target_file_name), 
                        pitch_pred[ind, :].cpu().detach().numpy())  
    loader = create_dataset("val", 1)
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            mask ,tokens, f0_trg = torch.tensor(data["mask"]).to(device),\
                torch.tensor(data["hubert"]).to(device),\
                torch.tensor(data["f0"]).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            emotion = torch.tensor(data['emotion']).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            names = data["names"] 
            for ind in range(len(names)):
                target_file_name = names[ind].split(os.sep)[-1].replace("wav", "npy")
                pitch_pred, _ = model(tokens, speaker, emotion, mask)
                # pitch_pred = torch.exp(pitch_pred) - 1
                np.save(os.path.join(config['F0']['contour'], target_file_name), 
                        pitch_pred[ind, :].cpu().detach().numpy()) 


if __name__ == "__main__":
    get_f0()
