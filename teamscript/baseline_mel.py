import os, glob, random
import torch, transformers
import torch.nn as nn
import torch.nn.functional as F

import augment

import numpy as np 
import pandas as pd 
import soundfile as sf
import pytorch_lightning as pl

from linformer import Linformer

from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from sklearn.model_selection import train_test_split

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import IPython.display as ipd

from my_utils import *
from transforms import * 
from kaggle_utils import *

from tqdm import tqdm


path = "/kaggle/input/birdclef-2021/"
train_labels = pd.read_csv(path+'train_soundscape_labels.csv')
train_meta = pd.read_csv(path+'train_metadata.csv')
test_data = pd.read_csv(path+'test.csv')
samp_subm = pd.read_csv(path+'sample_submission.csv')
meta_audios = glob.glob("/kaggle/input/birdclef-2021/train_short_audio/*/*.ogg")

audio_map = {}
not_added = []

from concurrent.futures import ThreadPoolExecutor

# aud_cache = {}
# for i, x in tqdm(enumerate(meta_audios)):
#     a_idx = x.split("/")[-1]
#     audio_map[a_idx] = x
#     audio, _ = sf.read(x)
#     aud_cache[a_idx] = audio
    
#     with ThreadPoolExecutor(max_workers=num_workers) as ex:
#         predictions = ex.map(process_file, range(len(videos)))
    
    

aud_cache = {}


def process_file(i):
    x = meta_audios[i]
    a_idx = x.split("/")[-1]
    audio, _ = sf.read(x)
    return [a_idx, audio]

with ThreadPoolExecutor(max_workers=64) as ex:
    predictions = ex.map(process_file, range(len(meta_audios)))
    
aud_cache = {}
for x, y in predictions:
    aud_cache[x] = y

b2n = {} ## bird -> number
n2b = {} ## number -> bird
for i, b in enumerate(train_meta.primary_label.unique()):
    b2n[b] = i
    n2b[i] = b

## data aug-
# t_meta, v_meta = train_test_split(train_meta, test_size=0.2, stratify=train_meta['primary_label'])

# t_meta.to_csv("t_meta.csv", index=None)
# v_meta.to_csv("v_meta.csv", index=None)

t_meta = pd.read_csv("t_meta.csv")
v_meta = pd.read_csv("v_meta.csv")

class CLFMetaDataset(Dataset):
    def __init__(self, audios, data, augmentation=None, b2n=b2n, n2b=n2b):
        
        self.audios = audios
        self.data = data.values
        self.sr = 32000
        self.augmentation = augmentation
        self.b2n = b2n
        self.n2b = n2b
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        primary_label = row[0]
        secondary_label = row[1]

        a_id = self.audios[row[9]]
        #audio, sr = sf.read(a_id)
        
        sr = 32000
        audio = aud_cache[a_id]
        
        max_ = len(audio) // sr
        rnd_start = np.random.randint(0, max_)
        if rnd_start > max_ - 12:
            rnd_start = 0

        audio = audio[int((rnd_start) * self.sr) : int((rnd_start + 11) * self.sr)]

        label = torch.zeros(397)
        a_label = self.b2n[primary_label]
        label[a_label] = 1

        ## encode secondary labels
        
        raw = secondary_label.split(",")
        raw[0] = raw[0][2:-1]
        raw = [x.replace("'", "").replace("]", "").replace(" ", "") for x in raw]
        for sl in raw:
            if sl != '':
                try:
                    label[self.b2n[sl]] = 1
                except:
                    continue

        ## add secondary labels
        frame_len = 10
        
        
        
        if len(audio) < (frame_len * sr):
            overhead = np.zeros(frame_len * sr)
            overhead[:len(audio)] = audio
            audio = overhead
        
        audio = audio[0: (frame_len * sr)]
        if self.augmentation and random.random() < 0.5:
            audio = torch.as_tensor(np.array(audio[0: (frame_len * sr)], dtype=np.float32))
            audio = self.augmentation.apply(audio, src_info={'rate': self.sr}, target_info={'rate': self.sr})[0].numpy()
            
        audio = audio[0: (frame_len * sr)]
        return audio, label

from sklearn.metrics import f1_score, average_precision_score

from torch.autograd import Variable

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CLFLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, y_hat, y):

        ## less adjustment when nocall
       
        loss = self.criterion(y_hat, y)
        return loss



# from torchlibrosa.stft import LogmelFilterBank, Spectrogram
# from torchlibrosa.augmentation import SpecAugmentation
from torch.cuda.amp import autocast 
from transformers import AdamW
class PANNsCNN14Att(pl.LightningModule):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, mixup, learning_rate, wd, criterion):
        super(PANNsCNN14Att, self).__init__()

        self.mixup = False
        self.criterion = CLFLoss()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
       
        
        self.interpolate_ratio = 32
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)  
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=3)
        self.effnet0 = EfficientNet.from_pretrained('efficientnet-b4')

        ## updated drop in GRU
        self.gru = torch.nn.GRU(input_size=1792, hidden_size=1792//2, 
                        num_layers=2, dropout=0.2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(1792, 2048)
        self.att_block = AttBlock(2048, classes_num, activation='linear')
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
#         init_layer(self.fc_audioset)

    def preprocess(self, input, mixup_lambda=None):
        # t1 = time.time()

        with autocast(enabled=False):
            x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        return x, frames_num
    
    def forward(self, input, mixup_lambda=None):
        x, frames_num = self.preprocess(input.float(), mixup_lambda=mixup_lambda)
        x = self.effnet0(x.float())
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.15, training=self.training, inplace=True)        
        x = torch.mean(x, dim=3)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.3, training=self.training)
        x = x.transpose(1, 2)
        (x, _) = self.gru(x)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.3, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

        return output_dict

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.mixup:
            x, y_a, y_b, lam = mixup_data(x, y, 1., True)
            x, y_a, y_b = map(Variable, (x, y_a, y_b))

        y_hat = self(x)['logit']
     
        if self.mixup: loss = mixup_criterion(self.criterion, y_hat, y_a, y_b, lam)
        if not self.mixup: loss = self.criterion(y_hat, y)

        score = self.calc_f1(y_hat, y)
        self.log("train/loss", loss, sync_dist=True)
        self.log("train/score", score, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)['logit']

        loss = self.criterion(y_hat, y)
        score = self.calc_f1(y_hat, y)
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/score", score, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        epochs = 100
        tlen = len(train_loader)
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = {'scheduler': transformers.get_linear_schedule_with_warmup(optimizer, tlen * 5, tlen*epochs), 'interval': 'step'}
        return [optimizer], [scheduler]

    def calc_f1(self, logits, y, threshold=0.5):
        preds = logits.sigmoid().cpu().detach().numpy()
        gts   = y.cpu().detach().numpy()
        preds = (preds > threshold)
        score = f1_score(gts, preds, average='micro')
        return score

    
## B7
class B7BCE(pl.LightningModule):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, mixup, learning_rate, wd, criterion): 
        super(B7BCE, self).__init__()
        
        
        self.learning_rate = 1e-3
        self.wd = 0
        self.mixup = False
        self.criterion = CLFLoss()
        
        self.save_hyperparameters('mixup', 'mel_bins', 'learning_rate', 'wd', 'criterion')

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        self.interpolate_ratio = 32
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        
        self.bn0 = nn.BatchNorm2d(64)  
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=3)
        self.effnet0 = EfficientNet.from_pretrained('efficientnet-b7')

        ## updated drop in GRU
        self.gru = torch.nn.GRU(input_size=2560, hidden_size=1280, 
                        num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2560, 2560)
        self.att_block = AttBlock(2560, classes_num, activation='sigmoid')

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        
        
    def preprocess(self, input, mixup_lambda=None):
        # t1 = time.time()

        with autocast(enabled=False):
            x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        return x, frames_num
    
    
    def forward(self, input, mixup_lambda=None):
        input = input.float()
        x, frames_num = self.preprocess(input, mixup_lambda=mixup_lambda)
        x = self.effnet0(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)        
        x = torch.mean(x, dim=3)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        (x, _) = self.gru(x)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

        return output_dict
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.mixup:
            x, y_a, y_b, lam = mixup_data(x, y, 1., True)
            x, y_a, y_b = map(Variable, (x, y_a, y_b))

        y_hat = self(x)['logit']
     
        if self.mixup: loss = mixup_criterion(self.criterion, y_hat, y_a, y_b, lam)
        if not self.mixup: loss = self.criterion(y_hat, y)

        score = self.calc_f1(y_hat, y)
        self.log("train/loss", loss, sync_dist=True)
        self.log("train/score", score, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)['logit']

        loss = self.criterion(y_hat, y)
        score = self.calc_f1(y_hat, y)
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/score", score, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        epochs = 200
        tlen = len(train_loader)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.wd)
        scheduler = {'scheduler': transformers.get_linear_schedule_with_warmup(optimizer, tlen, tlen*epochs), 'interval': 'step'}
        return [optimizer], [scheduler]

    def calc_f1(self, logits, y, threshold=0.5):
        preds = logits.sigmoid().cpu().detach().numpy()
        gts   = y.cpu().detach().numpy()
        preds = (preds > threshold)
        score = f1_score(gts, preds, average='micro')
        return score

    
    
if __name__ == '__main__':
    from losses import *
    
    model_config = {
        "sample_rate": 32000,
        "window_size": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 14000,
        "classes_num": 264,
        "mixup": False,
        "learning_rate": 1e-3,
        "wd": 0,
        "criterion": "clf_loss"
        
    }
    def get_augs():
        noise_generator = lambda: torch.zeros((1, 320000)).uniform_()
        random_pitch_shift = lambda: np.random.randint(0, +200)
        effect_chain = augment.EffectChain().time_dropout(max_seconds=1.0).additive_noise(noise_generator, snr=15).pitch(random_pitch_shift)
        return effect_chain

    train_augs = get_augs()

    train_dataset = CLFMetaDataset(audio_map, t_meta, augmentation=train_augs)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=64, pin_memory=True)

    valid_dataset = CLFMetaDataset(audio_map, v_meta, augmentation=None)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=64, pin_memory=True)


    # for i, x in enumerate(train_loader):
    #     print(x[0].shape)
    #     if i == len(train_loader):
    #         break
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # ## init
    device = torch.device("cuda")
    train_cfg = {'epochs': 100,
                 'base_lr': 1e-2}
    
    
    ## LOADING B4
    model = PANNsCNN14Att(**model_config).to(device)
    ckpt = torch.load('/kaggle/input/sed-resnests/best_b4_lastfit.pth', map_location='cpu')['model_state_dict']
    #print(ckpt.keys())
    model.load_state_dict(ckpt)
    model.att_block = AttBlockV2(2048, 397, activation='linear')

    ## LOADING B7
    #model = B7BCE(**model_config).to(device)
    #ckpt = torch.load('/kaggle/input/sed-resnests/best_multilabel_b7.pth', map_location='cpu')['model_state_dict']
    #model.load_state_dict(ckpt)
    #model.att_block = AttBlockV2(2560, 397, activation='linear')
    ## CHANGE MEL
    
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    
    mc = {
        "sample_rate": 32000,
        "window_size": 1024,
        "hop_size": 320,
        "mel_bins": 224,
        "fmin": 0,
        "fmax": 16000,
        "classes_num": 264
    }
    window_size = mc['window_size']
    hop_size = mc['hop_size']
    window_size = mc['window_size']
    sample_rate = mc['sample_rate']
    mel_bins = mc['mel_bins']
    fmin = mc['fmin']
    fmax = mc['fmax']
    model.interpolate_ratio = 32
    # Spectrogram extractor
    model.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
        win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
        freeze_parameters=True)

    # Logmel feature extractor
    model.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
        n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
        freeze_parameters=True)

    # Spec augmenter
    model.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
        freq_drop_width=8, freq_stripes_num=2)

    model.bn0 = nn.BatchNorm2d(224)  
    
    ckpt = torch.load("b4_224_2_ep94.ckpt", map_location='cpu')['state_dict']
    model.load_state_dict(ckpt)
    
    ##################
    
    name = 'b4_224_bce'
    
    cp_f1 = ModelCheckpoint(
        monitor="val/score", mode="max", save_top_k=1
    )
    cp_loss = ModelCheckpoint(
        monitor="val/loss", mode="min", save_top_k=1
    )

    logger = TensorBoardLogger(
        save_dir="logs/", name="lightning_logs", version=f"{name}"
    )
    
    ## TODO: LOG HYPERPARAMETERS

    trainer = pl.Trainer(
        #limit_train_batches = 0.5,
        #limit_val_batches   = 0.2,
        #resume_from_checkpoint='b4_bcefocal_115ep.ckpt',
        max_epochs=train_cfg['epochs'],
        precision=16,
        gpus=1,
        logger=logger,
        callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='step')],
        track_grad_norm=2,
        checkpoint_callback=[cp_f1, cp_loss],
        gradient_clip_val=5.0,
        stochastic_weight_avg=True,

    )
    
    trainer.fit(model, train_loader, valid_loader)