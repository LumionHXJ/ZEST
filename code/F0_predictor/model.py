import os
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.nn.init as init

import yaml

with open('code/config.yml', 'r') as file:
    config = yaml.safe_load(file)

class PitchModel(nn.Module):
    def __init__(self):
        super(PitchModel, self).__init__()
        hidden_channel = config['F0']['hidden_channel']
        self.embedding = nn.Embedding(101, hidden_channel, padding_idx=100) # hubert 100 + 1 pad     
        self.fusion = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden_channel, 4, hidden_channel * 4,
                activation=F.gelu, 
                batch_first=True,
                norm_first=True
            ),
            num_layers=6
        )
        self.es_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_channel, 4, hidden_channel * 4,
                activation=F.gelu, 
                batch_first=True,
                norm_first=True
            ),
            num_layers=2
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=(3,), padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channel, 1, kernel_size=(1,), padding=0),
            #nn.Sigmoid()
        )
        self.speaker_linear = nn.Linear(hidden_channel, hidden_channel)
        self.emotion_linear = nn.Linear(hidden_channel, hidden_channel)

    def forward(self, tokens, speaker, emo_embedded, lengths):
        # speaker from EASE, emo_embedded from SACE
        device = emo_embedded.device
        hidden = self.embedding(tokens.int()) # hubert
        speaker_temp = self.speaker_linear(speaker) # adapter
        emo_embedded = self.emotion_linear(emo_embedded)
        #speaker_temp = speaker_temp.unsqueeze(1).repeat(1, emo_embedded.shape[1], 1) # EASE
        emo_embedded = torch.concatenate([emo_embedded, speaker_temp.unsqueeze(1)], dim=1) # EASE + SACE (不一定好)
        emo_embedded = self.es_fusion(emo_embedded)
        pred_pitch = self.fusion(hidden, emo_embedded) # Cross-Attn (HuBERT, EASE + SACE)
        # print(pred_pitch.min(), pred_pitch.max())
        pred_pitch = pred_pitch.permute(0, 2, 1)
        pred_pitch = self.cnn(pred_pitch)
        pred_pitch = pred_pitch.squeeze(1)
        mask = torch.arange(hidden.shape[1]).expand(hidden.shape[0], hidden.shape[1]).to(device) < lengths.unsqueeze(1)
        pred_pitch = pred_pitch.masked_fill(~mask, 0.0)
        mask = mask.int()

        return pred_pitch, mask