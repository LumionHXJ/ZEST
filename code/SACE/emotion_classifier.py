import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
import torch.nn as nn
import random
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import yaml
from torch.autograd import Function
import torch.nn.functional as F

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

class SACE(nn.Module):
    def __init__(self,
                 emo_claases, # 5 classes
                 speaker_class):
        
        super().__init__()
        
        embedding_dim = config['SACE']['input_channel'] # 256
        feat_dim = config['SACE']['hidden_channel']
        self.out_emo = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//4),
            nn.ReLU(),
            nn.Linear(feat_dim//4, emo_claases)
        )
        self.out_spkr = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//4),
            nn.ReLU(),
            nn.Linear(feat_dim//4, speaker_class) # output_speaker
        )

        self.adapter = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim//2, kernel_size=5, padding=2),
            nn.BatchNorm1d(embedding_dim//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=embedding_dim//2, out_channels=feat_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )

        self.query = nn.Parameter(torch.randn((1, feat_dim))) # 0 for emo, 1 for spkr
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(feat_dim, 8, feat_dim,
                                       activation=F.gelu, 
                                       batch_first=True,
                                       norm_first=True),
            num_layers=3
        )

        
    def forward(self, embedded, alpha=1.0):
        bs = embedded.size()[0]
        embedded = embedded.permute(0, 2, 1) # bs, ?, 1024 -> bs, 1024, ?
        emo_embedded = self.adapter(embedded)
        emo_embedded = emo_embedded.permute(0, 2, 1) # output for attention

        outputs = self.decoder(self.query[None].repeat(bs, 1, 1), 
                               emo_embedded) # bs, 2, 256
        out_emo = self.out_emo(outputs[:, 0]) # bs, 256

        emo_hidden = torch.mean(emo_embedded, 1).squeeze(1)
        reverse_feature = ReverseLayerF.apply(emo_hidden, alpha)
        output_spkr = self.out_spkr(reverse_feature) # adv of speaker (disentangle)
        
        return out_emo, output_spkr, emo_embedded, outputs[:, 0]

def collate(data):
    max_len_aud = 0
    audios = []
    spkr_labels = []
    emo_labels = []
    files = []
    for ind in range(len(data)):
        max_len_aud = max(data[ind][0].shape[0], max_len_aud)
    for i in range(len(data)):
        final_sig = np.concatenate((data[i][0], 
                                    np.zeros((max_len_aud-data[i][0].shape[0], 
                                              config['SACE']['input_channel']))), 0)
        audios.append(final_sig)
        spkr_labels.append(data[i][1])
        emo_labels.append(data[i][2])
        files.append(data[i][3])
    audios = np.array(audios, dtype=np.float32)
    return torch.tensor(audios, dtype=torch.float32), torch.tensor(spkr_labels), torch.tensor(emo_labels), files

def create_dataset(mode, bs=32):
    language = config['language']
    speaker_folder = os.path.join(config['dataset']['data_root'], config['dataset'][language]['wav2vec2'][mode])
    dataset = MyDataset(speaker_folder)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    num_workers=16,
                    drop_last=False,
                    collate_fn=collate)
    return loader

def train():    
    train_loader = create_dataset("train")
    val_loader = create_dataset("val")
    model = SACE(emo_claases=config['SACE']['emotion_classes'],
                 speaker_class=config['SACE']['speaker_classes'])
    model.to(device)
    base_lr = 1e-2
    parameters = list(model.parameters()) 
    optimizer = SGD([{'params':parameters, 
                       'lr':base_lr}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    final_val_loss = 1e20

    for e in range(15):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        pred_tr_sp = []
        gt_tr_sp = []
        pred_val_sp = []
        gt_val_sp = []
        for i, data in enumerate(tqdm(train_loader)):
            model.train()
            # p = float(- i + (11 - e) * len(train_loader)) / 100 / len(train_loader)
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1
            wav2vec_feat, spkr_labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
            out_emo, output_spkr, _, _ = model(wav2vec_feat, alpha=1.0)
            loss_adv = nn.CrossEntropyLoss(reduction='mean')(output_spkr, spkr_labels)
            loss_emo = nn.CrossEntropyLoss(reduction='mean')(out_emo, emo_labels)
            loss = loss_emo + loss_adv
            tot_loss += loss_emo.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(out_emo, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)

            labels = emo_labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)

            pred_sp = torch.argmax(output_spkr, dim = 1)
            pred_sp = pred_sp.detach().cpu().numpy()
            pred_sp = list(pred_sp)
            pred_tr_sp.extend(pred_sp)

            labels = spkr_labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr_sp.extend(labels)
        
        scheduler.step()
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                wav2vec_feat, spkr_labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
                out_emo, output_spkr, _, _ = model(wav2vec_feat)
                loss = nn.CrossEntropyLoss(reduction='mean')(out_emo, emo_labels)
                val_loss += loss.detach().item()

                pred = torch.argmax(out_emo, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = emo_labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)

                pred_sp = torch.argmax(output_spkr, dim = 1)
                pred_sp = pred_sp.detach().cpu().numpy()
                pred_sp = list(pred_sp)
                pred_val_sp.extend(pred_sp)

                labels = spkr_labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val_sp.extend(labels)

        if val_loss < final_val_loss:
            torch.save(model, config['SACE']['checkpoint'])
            final_val_loss = val_loss
        train_loss = tot_loss/len(train_loader)
        train_f1 = accuracy_score(gt_tr, pred_tr)
        val_loss_log = val_loss/len(val_loader)
        val_f1 = accuracy_score(gt_val, pred_val)

        train_f1_sp = accuracy_score(gt_tr_sp, pred_tr_sp)
        val_f1_sp = accuracy_score(gt_val_sp, pred_val_sp)

        e_log = e + 1
        logger.info(f"Epoch {e_log}, Training Loss {train_loss}, Training Accuracy {train_f1}, Speaker Acc {train_f1_sp}")
        logger.info(f"Epoch {e_log}, Validation Loss {val_loss_log}, Validation Accuracy {val_f1},  Speaker Acc {val_f1_sp}")

def get_embedding():
    train_loader = create_dataset("train", 1)
    val_loader = create_dataset("val", 1)
    test_loader = create_dataset("test", 1)
    model = torch.load(config['SACE']['checkpoint'], map_location=device)
    model.to(device)
    model.eval()
    os.makedirs(config['SACE']['embedding'], exist_ok=True)
    os.makedirs(config['SACE']['emotion'], exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader)):
            wav2vec_feat, emo_labels = data[0].to(device), data[2].to(device)
            names = data[3]
            _, _, embedded, emotion = model(wav2vec_feat) # 1, ?, 128
            for ind in range(len(names)):
                target_file_name = names[ind].replace("wav", "npy")
                np.save(os.path.join(config['SACE']['embedding'], target_file_name), 
                        embedded[ind, :].cpu().detach().numpy())
                np.save(os.path.join(config['SACE']['emotion'], target_file_name), 
                        emotion[ind, :].cpu().detach().numpy())


        for i, data in enumerate(tqdm(val_loader)):
            wav2vec_feat, emo_labels = data[0].to(device), data[2].to(device)
            names = data[3]
            _, _, embedded, emotion = model(wav2vec_feat) # 1, ?, 128
            for ind in range(len(names)):
                target_file_name = names[ind].replace("wav", "npy")
                np.save(os.path.join(config['SACE']['embedding'], target_file_name), 
                        embedded[ind, :].cpu().detach().numpy())
                np.save(os.path.join(config['SACE']['emotion'], target_file_name), 
                        emotion[ind, :].cpu().detach().numpy())
                
        for i, data in enumerate(tqdm(test_loader)):
            wav2vec_feat, emo_labels = data[0].to(device), data[2].to(device)
            names = data[3]
            _, _, embedded, emotion = model(wav2vec_feat) # 1, ?, 128
            for ind in range(len(names)):
                target_file_name = names[ind].replace("wav", "npy")
                np.save(os.path.join(config['SACE']['embedding'], target_file_name), 
                        embedded[ind, :].cpu().detach().numpy())
                np.save(os.path.join(config['SACE']['emotion'], target_file_name), 
                        emotion[ind, :].cpu().detach().numpy())
                
if __name__ == "__main__":
    #train()
    get_embedding()
