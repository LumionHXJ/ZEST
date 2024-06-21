import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import yaml
from torch.autograd import Function

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
with open('code/config.yml', 'r') as file:
    config = yaml.safe_load(file)

class MyDataset(Dataset):
    def __init__(self, speaker_folder):
        self.speaker_folder = speaker_folder
        wav_files = os.listdir(speaker_folder)
        wav_files = [x for x in wav_files if ".npy" in x]
        wav_files = [x.replace("_gen.wav", ".wav") for x in wav_files]
        speaker_features = os.listdir(speaker_folder)
        speaker_features = [x for x in speaker_features if ".npy" in x]
        self.wav_files = wav_files
        self.speaker_features = speaker_features
        self.sr = 16000
        self.speaker_dict = {}
        for ind in range(11, 21):
            self.speaker_dict["00"+str(ind)] = ind-11

    def __len__(self):
        return len(self.wav_files) 

    def getspkrlabel(self, file_name):
        spkr_name = file_name[-15:][:4]
        spkr_label = self.speaker_dict[spkr_name]

        return spkr_label

    def getemolabel(self, file_name):
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
        
    def __getitem__(self, audio_ind):
        speaker_feat = np.load(os.path.join(self.speaker_folder, self.wav_files[audio_ind].replace(".wav", ".npy")))
        speaker_label = self.getspkrlabel(self.wav_files[audio_ind][-15:])
        class_id = self.getemolabel(self.wav_files[audio_ind]) 

        return speaker_feat, speaker_label, class_id, self.wav_files[audio_ind]

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha # 这里进行了反转（即对抗训练），alpha为缩放因子

        return output, None

class SpeakerModel(nn.Module):
    def __init__(self, spkr_class=10, emo_class=5):
        super().__init__()
        hidden_channel = config['EASE']['hidden_channel']

        self.fc1 = nn.Linear(192, hidden_channel)
        self.ln1 = nn.LayerNorm(hidden_channel)
        
        self.fc = nn.Linear(hidden_channel, hidden_channel)
        self.ln2 = nn.LayerNorm(hidden_channel)
        
        self.fc_embed = nn.Linear(hidden_channel, hidden_channel)
        self.ln3 = nn.LayerNorm(hidden_channel)
        
        self.fc2 = nn.Linear(hidden_channel, hidden_channel)
        self.ln4 = nn.LayerNorm(hidden_channel)
        
        self.fc_embed_1 = nn.Linear(hidden_channel, hidden_channel)
        self.ln5 = nn.LayerNorm(hidden_channel)
        
        self.fc3 = nn.Linear(hidden_channel, spkr_class)
        
        self.fc4 = nn.Linear(hidden_channel, hidden_channel)
        self.ln6 = nn.LayerNorm(hidden_channel)
        
        self.fc_embed_2 = nn.Linear(hidden_channel, hidden_channel)
        self.ln7 = nn.LayerNorm(hidden_channel)
        
        self.fc5 = nn.Linear(hidden_channel, emo_class)
    
    def forward(self, feat, alpha=1.0):
        feat = self.ln1(self.fc1(feat))
        feat = self.ln2(self.fc(self.ln3(self.fc_embed(feat))))
        
        reverse = ReverseLayerF.apply(feat, alpha)
        
        out = self.ln5(self.fc_embed_1(self.ln4(self.fc2(feat))))
        out = self.fc3(out)
        
        emo_out = self.ln7(self.fc_embed_2(self.ln6(self.fc4(reverse))))
        emo_out = self.fc5(emo_out)
        
        return out, emo_out, feat

def create_dataset(mode, bs=32):
    language = config['language']
    speaker_folder = os.path.join(config['dataset']['data_root'], config['dataset'][language]['ecapa'][mode])
    dataset = MyDataset(speaker_folder)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False)
    return loader

def train():
    
    train_loader = create_dataset("train")
    val_loader = create_dataset("val")
    model = SpeakerModel(emo_class=config['EASE']['emotion_classes'],
                         spkr_class=config['EASE']['speaker_classes'])
    model.to(device)
    base_lr = 1e-4
    parameters = list(model.parameters()) 
    optimizer = Adam([{'params':parameters, 'lr':base_lr}], weight_decay=0.1)
    final_val_loss = 1e20

    for e in range(10):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        pred_tr_emo = []
        gt_tr_emo = []
        pred_val_emo = []
        gt_val_emo = []
        for i, data in enumerate(train_loader):
            model.train()
            #p = float(i + e * len(train_loader)) / 100 / len(train_loader)
            #alpha = 2. / (1. + np.exp(-10 * p)) - 1
            speaker_feat, labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs, out_emo, _ = model(speaker_feat, alpha=1.0)
            loss = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)(outputs, labels)
            loss_emo = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)(out_emo, emo_labels)
            loss += loss_emo
            tot_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(outputs, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)

            pred = torch.argmax(out_emo, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr_emo.extend(pred)
            labels = emo_labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr_emo.extend(labels)
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                speaker_feat, labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
                outputs, out_emo, _ = model(speaker_feat)
                loss = nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
                val_loss += loss.detach().item()
                pred = torch.argmax(outputs, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)

                pred = torch.argmax(out_emo, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val_emo.extend(pred)
                labels = emo_labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val_emo.extend(labels)
        if val_loss < final_val_loss:
            torch.save(model, config['EASE']['checkpoint'])
            final_val_loss = val_loss
        train_loss = tot_loss/len(train_loader)
        train_f1 = accuracy_score(gt_tr, pred_tr)
        train_f1_emo = accuracy_score(gt_tr_emo, pred_tr_emo)
        val_loss_log = val_loss/len(val_loader)
        val_f1 = accuracy_score(gt_val, pred_val)
        val_f1_emo = accuracy_score(gt_val_emo, pred_val_emo)
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}\
                    Emotion Accuracy {train_f1_emo}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}\
                    Emotion Accuracy {val_f1_emo}")

def get_embedding():
    train_loader = create_dataset("train", 1)
    val_loader = create_dataset("val", 1)
    test_loader = create_dataset("test", 1)
    model = torch.load(config['EASE']['checkpoint'], map_location=device)
    model.to(device)
    model.eval()
    os.makedirs(config['EASE']['embedding'], exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            speaker_feat, labels = data[0].to(device), data[1].to(device)
            names = data[3]
            _, _, embedded = model(speaker_feat)
            for ind in range(len(names)):
                target_file_name = names[ind].replace("wav", "npy")
                np.save(os.path.join(config['EASE']['embedding'], target_file_name), embedded[ind, :].cpu().detach().numpy())

        for i, data in enumerate(val_loader):
            speaker_feat, labels = data[0].to(device), data[1].to(device)
            names = data[3]
            _, _, embedded = model(speaker_feat)
            for ind in range(len(names)):
                target_file_name = names[ind].replace("wav", "npy")
                np.save(os.path.join(config['EASE']['embedding'], target_file_name), embedded[ind, :].cpu().detach().numpy())  
        
        for i, data in enumerate(test_loader):
            speaker_feat, labels = data[0].to(device), data[1].to(device)
            names = data[3]
            _, _, embedded = model(speaker_feat)
            for ind in range(len(names)):
                target_file_name = names[ind].replace("wav", "npy")
                np.save(os.path.join(config['EASE']['embedding'], target_file_name), embedded[ind, :].cpu().detach().numpy())  

if __name__ == "__main__":
    train()
    get_embedding()
