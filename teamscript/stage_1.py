import os, glob, random, argparse
import torch, transformers
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata


import augment

import numpy as np 
import pandas as pd 
import soundfile as sf
import pytorch_lightning as pl


from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split
from transformers import AdamW, get_cosine_schedule_with_warmup

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torch.autograd import Variable
from torch.cuda.amp import autocast
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
from sklearn.metrics import f1_score, average_precision_score

from effnet import *
from transforms import * 



# =================================================
# Utilities #
# =================================================

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

## data aug-
# t_meta, v_meta = train_test_split(train_meta, test_size=0.2, stratify=train_meta['primary_label'])

# t_meta.to_csv("t_meta.csv", index=None)
# v_meta.to_csv("v_meta.csv", index=None)



class WaveformDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 img_size=224,
                 waveform_transforms=None,
                 period=5,
                 validation=False):
        self.df = df
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.period = period
        self.validation = validation
        self.datadir = "/kaggle/input/birdclef-2021/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        datadir = Path(self.datadir)
        wav_name = sample["filename"]
        ebird_code = sample["primary_label"]
        secondary_labels = eval(sample["secondary_labels"])

        y, sr = sf.read(datadir / 'train_short_audio' / ebird_code / wav_name)

        len_y = len(y)
        effective_length = sr * self.period
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            if not self.validation:
                start = np.random.randint(effective_length - len_y)
            else:
                start = 0
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            if not self.validation:
                start = np.random.randint(len_y - effective_length)
            else:
                start = 0
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        y = np.nan_to_num(y)
        if np.isnan(y).any():
            y = np.zeros(len(y))

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        y = np.nan_to_num(y)

        labels = np.zeros(len(b2n), dtype=float)
        labels[b2n[ebird_code]] = 1.0

        mask = np.ones(len(b2n), dtype=float)
        for secondary_label in secondary_labels:
            if secondary_label in b2n:
                mask[b2n[secondary_label]] = 0.0

        return y, labels, mask



# =================================================
# Criterion #
# =================================================
# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets, mask=None):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        if mask is not None:
            loss = loss * mask
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, input, target, mask):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target, mask)
        aux_loss = self.focal(clipwise_output_with_max, target, mask)

        return self.weights[0] * loss + self.weights[1] * aux_loss


__CRITERIONS__ = {
    "BCEFocalLoss": BCEFocalLoss,
    "BCEFocal2WayLoss": BCEFocal2WayLoss
}


def get_criterion(name):
    if hasattr(nn, name):
        return nn.__getattribute__(name)()
    elif __CRITERIONS__.get(name) is not None:
        return __CRITERIONS__[name]()
    else:
        raise NotImplementedError


# =================================================
# Model #
# =================================================
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def do_mixup(x: torch.Tensor, mixup_lambda: torch.Tensor):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0::2].transpose(0, -1) * mixup_lambda[0::2] +
           x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)
    return out


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(
                self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return torch.from_numpy(np.array(mixup_lambdas, dtype=np.float32))


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


def gem(x: torch.Tensor, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        
class EffNetSED(pl.LightningModule):
    def __init__(self, hparams):
        super(EffNetSED, self).__init__()        
        self.save_hyperparameters()
        
        sample_rate = hparams['model_config']['sample_rate']
        window_size = hparams['model_config']['window_size']
        hop_size    = hparams['model_config']['hop_size']
        mel_bins    = hparams['model_config']['mel_bins']
        fmin        = hparams['model_config']['fmin']
        fmax        = hparams['model_config']['fmax']
        classes_num = hparams['model_config']['classes_num']
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
       
        print(hparams)
        
                
        self.epochs    = hparams['train']['epochs'] 
        self.mixup     = hparams['train']['mixup']
        self.base_lr   = hparams['train']['base_lr']
        self.wd        = hparams['train']['wd']
        self.criterion = get_criterion(hparams['train']['criterion'])
        
        if self.mixup:
            print('mixup training...')
            self.mixup_augmenter = Mixup(mixup_alpha=1.)

        
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
        self.spec_augmenter = SpecAugmentation(time_drop_width=hparams['model_config']['spec_twd'], time_stripes_num=hparams['model_config']['spec_tsn'], 
            freq_drop_width=hparams['model_config']['spec_fdw'], freq_stripes_num=hparams['model_config']['spec_fsn'])

        self.bn0 = nn.BatchNorm2d(mel_bins)  
        self.effnet0 = EfficientNet.from_pretrained('efficientnet-b4')

        ## updated drop in GRU
        self.gru = torch.nn.GRU(input_size=1792, hidden_size=1792//2, 
                        num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(1792, 2048)
        self.att_block = AttBlockV2(2048, classes_num, activation='linear')
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def preprocess(self, input, mixup_lambda=None):

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
        x, y, mask = batch
        
        if self.mixup:
            mixup_lambda = self.mixup_augmenter.get_lambda(batch_size=len(x)).to(self.device)
            y_hat = self(x, mixup_lambda)            
            y = do_mixup(y, mixup_lambda)
            mask = do_mixup(mask, mixup_lambda)
        else:
            y_hat = self(x)

        loss = self.criterion(y_hat, y, mask)

        #f1_03 = self.calc_f1(y_hat, y, threshold=0.5)
        #f1_05 = self.calc_f1(y_hat, y, threshold=0.3)
        #f1_07 = self.calc_f1(y_hat, y, threshold=0.7)

        self.log("train/loss", loss, sync_dist=True)
        
        #self.log("train/f1_03", f1_03, sync_dist=True)
        #self.log("train/f1_05", f1_05, prog_bar=True, sync_dist=True)
        #self.log("train/f1_07", f1_07, sync_dist=True)

        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y, mask)
        
        f1_03 = self.calc_f1(y_hat, y, threshold=0.5)
        f1_05 = self.calc_f1(y_hat, y, threshold=0.3)
        f1_07 = self.calc_f1(y_hat, y, threshold=0.7)

        self.log("val/loss", loss, sync_dist=True)
        
        self.log("val/f1_03", f1_03, sync_dist=True)
        self.log("val/f1_05", f1_05, prog_bar=True, sync_dist=True)
        self.log("val/f1_07", f1_07, sync_dist=True)

    def configure_optimizers(self):
        epochs = self.epochs
        tlen = len(trn_loader)
        
        optimizer = AdamW(self.parameters(), lr=self.base_lr)
        scheduler = {'scheduler': transformers.get_linear_schedule_with_warmup(optimizer, tlen * (epochs * 0.1), tlen*epochs), 'interval': 'step'}
        return [optimizer], [scheduler]

    def calc_f1(self, logits, y, threshold=0.5):
        preds = logits['logit'].sigmoid().cpu().detach().numpy()
        gts   = y.cpu().detach().numpy()
        preds = (preds > threshold)
        score = f1_score(gts, preds, average='micro')
        return score

    
if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', 
                    type=str, 
                    help='name for the experiment')



    args = parser.parse_args()
    
    ## GET experiment @click
    path = "/kaggle/input/birdclef-2021/"
    train_labels = pd.read_csv(path+'train_soundscape_labels.csv')
    train_meta = pd.read_csv(path+'train_metadata.csv')
    test_data = pd.read_csv(path+'test.csv')
    samp_subm = pd.read_csv(path+'sample_submission.csv')
    meta_audios = glob.glob("/kaggle/input/birdclef-2021/train_short_audio/*/*.ogg")


    b2n = {} ## bird -> number
    n2b = {} ## number -> bird
    for i, b in enumerate(train_meta.primary_label.unique()):
        b2n[b] = i
        n2b[i] = b
     
    device = torch.device("cuda")
    hparams = {
        "train": {
            "epochs": 50,
            "base_lr": 1e-3,
            "mixup": True,
            "wd": 0.,
            "criterion": "BCEFocal2WayLoss"},
        "model_config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 768,
            "mel_bins": 224,
            "fmin": 300,
            "fmax": 16000,
            "spec_twd": 64,
            "spec_tsn": 2,
            "spec_fdw": 16,
            "spec_fsn": 2,
            "classes_num": 397},
        "data": {
            "image_size": 224, 
            "batch_size": 128,
            "num_workers": 64, 
            "period": 20,
            "trn_augs": "...",
            "val_augs": "..."
        },
        "transforms": {'train': [{"name": "Normalize"}, 
                                 {"name": "PinkNoise", "params": {"min_snr": 5}}],
                       'valid': [{"name": "Normalize"}] 
                      }
    }
    
    # =================================================
    # Data & augmentations                            #
    # =================================================
    t_meta = pd.read_csv("birdclef_csvs/t_meta.csv")
    v_meta = pd.read_csv("birdclef_csvs/v_meta.csv")

    train_augs = get_transforms(hparams, 'train')
    
    ## train meta
    trn_loader = torchdata.DataLoader(WaveformDataset(t_meta, hparams['data']['image_size'], None, hparams['data']['period'], False), batch_size=hparams['data']['batch_size'], num_workers=hparams['data']['num_workers'], shuffle=True, pin_memory=True, drop_last=True)
    val_loader = torchdata.DataLoader(WaveformDataset(v_meta, hparams['data']['image_size'], None, hparams['data']['period'], False), batch_size=hparams['data']['batch_size'], num_workers=hparams['data']['num_workers'], shuffle=False, pin_memory=True)

    
    # =================================================
    # Warm-start                                      #
    # =================================================
    model = EffNetSED(hparams).to(device)
    ckpt = torch.load('./pretrained_weights/b4_224_2_ep94.ckpt', map_location='cuda')['state_dict']
    model.load_state_dict(ckpt, strict=False) 
    
    # =================================================
    # Run                                             #
    # =================================================
    name = f'{args.name}'
    
    cp_f1 = ModelCheckpoint(monitor="val/f1_05", mode="max", save_top_k=1)
    cp_loss = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1)
    logger = TensorBoardLogger(save_dir="logs/", name="lightning_logs", version=f"{name}")
    

    ## TODO: LOG HYPERPARAMETERS
    

    trainer = pl.Trainer(
        #limit_train_batches = 0.5,
        #limit_val_batches   = 0.2,
        #resume_from_checkpoint='b4_bcefocal_115ep.ckpt',
        max_epochs=hparams['train']['epochs'],
        precision=16,
        gpus=1,
        logger=logger,
        callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='step')],
        track_grad_norm=2,
        checkpoint_callback=cp_loss,
        gradient_clip_val=5.0,
        stochastic_weight_avg=True,

    )
    
    trainer.fit(model, trn_loader, val_loader)