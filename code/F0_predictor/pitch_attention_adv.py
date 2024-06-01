import os
import torch
import logging
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
import random
import torch.nn.functional as F
import ast
import time
from tensorboardX import SummaryWriter
from datetime import datetime
import yaml
from model import PitchModel
from dataset import create_dataset_ddp

with open('code/config.yml', 'r') as file:
    config = yaml.safe_load(file)

def setup_logger():
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    os.environ['PYTHONHASHSEED'] = str(seed)

torch.set_printoptions(profile="full")
log_freq = 10
val_freq = 5
accumulation_grad = 4
SEED = 1234
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

set_seed(1234)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

# init ddp
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

if local_rank == 0:
    logger = setup_logger()
    writer = SummaryWriter(log_dir=f'run_log/{now}')
else:
    logger = None
    writer = None

lambda_l1 = 0.1
ckpt = config['F0']['checkpoint']

def l2_loss(input, target):
    return F.l1_loss(
        input=input.float(),
        target=target.float(),
        reduction='none'
    )

def train():
    train_loader, train_sampler = create_dataset_ddp("train")
    val_loader, val_sampler = create_dataset_ddp("val")
    model = PitchModel().to(device)
    if ckpt != None:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    
    if torch.cuda.device_count() > 1:
        if local_rank == 0:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = DDP(model, 
                    device_ids=[local_rank], 
                    find_unused_parameters=True)
    model.to(device)

    base_lr = 1e-4
    parameters = list(model.parameters()) 
    optimizer = Adam([{'params':parameters, 'lr':base_lr}])
    final_val_loss = 1e20
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='min', 
                                  factor=0.1, 
                                  patience=1, 
                                  threshold=0.01,
                                  verbose=True)
    scaler = GradScaler()
    f0_l1 = nn.L1Loss(reduction='mean')
    emo_ce = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05)
    spkr_ce = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05)

    for e in range(50):
        model.train()
        val_loss, val_acc = 0.0, 0.0
        train_sampler.set_epoch(e)
        epoch_start_time = time.time()

        for i, data in enumerate(train_loader):
            model.train()
            mask ,tokens, f0_trg = torch.tensor(data["mask"]).to(device),\
                torch.tensor(data["hubert"]).to(device),\
                torch.tensor(data["f0"]).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            emotion = torch.tensor(data['emotion']).to(device)

            with autocast():
                pitch_pred, mask_loss = model(tokens, speaker, emotion, mask)
                pitch_pred = torch.exp(pitch_pred) - 1
                loss = (mask_loss * f0_l1(pitch_pred, f0_trg.float().detach())).mean() * lambda_l1 # pitch loss
                
                loss = loss / accumulation_grad
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_grad == 0:
                scaler.step(optimizer)
                scaler.update()

            if (i + 1) % log_freq == 0 and local_rank == 0:
                iteration = e * len(train_loader) + i + 1
                writer.add_scalars(f'Loss',
                                   {"Loss":loss.item() * accumulation_grad},
                                    global_step=iteration)
                writer.add_scalar('Train/Learning Rate',
                                  scalar_value=optimizer.param_groups[0]['lr'], 
                                  global_step=iteration)
                writer.add_text('Time', 
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                iteration)
                logger.info(f"Epoch {e+1}, Iter {i + 1}/{len(train_loader)}, F0 reconstruction Loss {loss * 4}")
        
        if (i + 1) % accumulation_grad != 0: # lasting grad
            scaler.step(optimizer)
            scaler.update()
        if (i + 1) % log_freq != 0 and local_rank == 0:
            logger.info(f"Epoch {e+1}, Iter {i + 1} / {len(train_loader)}, F0 reconstruction Loss {loss * 4}")
            iteration = e * len(train_loader) + i + 1
            writer.add_scalars('Loss',
                                {"Train Loss":loss.item() * accumulation_grad},
                                global_step=iteration)
            writer.add_scalar('Train/Learning Rate',
                                scalar_value=optimizer.param_groups[0]['lr'], 
                                global_step=iteration)
            writer.add_text('Time', 
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                            iteration)
        if local_rank == 0:
            logger.info(f'Epoch {e+1}, Elapsed Time {time.time() - epoch_start_time}')

        if (e + 1) % val_freq != 0:
            continue

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                mask ,tokens, f0_trg = torch.tensor(data["mask"]).to(device),\
                    torch.tensor(data["hubert"]).to(device),\
                    torch.tensor(data["f0"]).to(device)
                speaker = torch.tensor(data["speaker"]).to(device)
                emotion = torch.tensor(data['emotion']).to(device)


                pitch_pred, mask_loss = model(tokens, speaker, emotion, mask)
                pitch_pred = torch.exp(pitch_pred) - 1
                loss = (mask_loss * f0_l1(pitch_pred, f0_trg.float().detach())).mean() * lambda_l1
                
                val_loss += loss.detach().item()
        if val_loss < final_val_loss and local_rank == 0:
            torch.save(model.state_dict(), os.path.join(config['F0']['checkpoint_savedir'], 
                                                        f'f0_predictor_epoch_{e+1}.pth'))
            final_val_loss = val_loss
        
        val_loss_log = val_loss/len(val_loader)

        scheduler.step(val_loss_log)
        if local_rank == 0:
            logger.info(f"Epoch {e+1}, Validation Loss {val_loss_log}")
            writer.add_scalars('Loss', {"Val Loss": val_loss_log}, e * len(train_loader) + i + 1)
            writer.add_text('Time', 
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                            iteration)
    
    dist.destroy_process_group()
    if local_rank == 0:
        writer.close()


if __name__ == "__main__":
    train()