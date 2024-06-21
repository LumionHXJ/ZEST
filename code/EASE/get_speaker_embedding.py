from speechbrain.pretrained import EncoderClassifier
import os
import torchaudio
import numpy as np
from tqdm import tqdm
import yaml

with open('code/config.yml', 'r') as file:
    config = yaml.safe_load(file)
classifier = EncoderClassifier.from_hparams(source=config['EASE']['pretrained'], 
                                            run_opts={"device":"cuda"})
language = config['language']
for mode in config['dataset']['mode']:
    folder = os.path.join(config['dataset']['data_root'], config['dataset'][language]['wav'][mode])
    target_folder = os.path.join(config['dataset']['data_root'], config['dataset'][language]['ecapa'][mode])
    os.makedirs(target_folder, exist_ok=True)
    wav_files = os.listdir(folder)
    wav_files = [x for x in wav_files if ".wav" in x]
    wav_files = [x for x in wav_files if ".npy" not in x]

    for i, wav_file in enumerate(tqdm(wav_files)):
        sig, sr = torchaudio.load(os.path.join(folder, wav_file))
        embeddings = classifier.encode_batch(sig.cuda())[0, 0, :]
        target_file = os.path.join(target_folder, wav_file.replace(".wav", ".npy"))
        np.save(target_file, embeddings.cpu().detach().numpy())