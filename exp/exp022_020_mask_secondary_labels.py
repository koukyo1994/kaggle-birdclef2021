import gc
import os
import math
import random
import warnings

import albumentations as A
import colorednoise as cn
import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from joblib import Parallel, delayed
from pathlib import Path
from typing import List

from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl import Runner, SupervisedRunner
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn import model_selection
from sklearn import metrics
from timm.models.layers import SelectAdaptivePool2d
from torch.optim.optimizer import Optimizer
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm


# =================================================
# Config #
# =================================================
class CFG:
    ######################
    # Globals #
    ######################
    seed = 1213
    epochs = 75
    train = True
    folds = [0]
    img_size = 224
    main_metric = "epoch_f1_at_05"
    minimize_metric = False

    ######################
    # Data #
    ######################
    train_datadir = Path("../input/birdclef-2021/train_short_audio")
    train_csv = "../input/birdclef-2021/train_metadata.csv"
    train_soundscape = "../input/birdclef-2021/train_soundscape_labels.csv"
    train_background = Path("../input/birdclef-2021/train_background")

    ######################
    # Dataset #
    ######################
    transforms = {
        "train": [{"name": "Normalize"}, {"name": "PinkNoise", "params": {"min_snr": 5}}],
        "valid": [{"name": "Normalize"}]
    }
    period = 20
    n_mels = 224
    fmin = 300
    fmax = 16000
    n_fft = 2048
    hop_length = 768
    sample_rate = 32000
    melspectrogram_parameters = {
        "n_mels": 224,
        "fmin": 20,
        "fmax": 16000
    }

    target_columns = [
        'acafly', 'acowoo', 'aldfly', 'ameavo', 'amecro',
        'amegfi', 'amekes', 'amepip', 'amered', 'amerob',
        'amewig', 'amtspa', 'andsol1', 'annhum', 'astfly',
        'azaspi1', 'babwar', 'baleag', 'balori', 'banana',
        'banswa', 'banwre1', 'barant1', 'barswa', 'batpig1',
        'bawswa1', 'bawwar', 'baywre1', 'bbwduc', 'bcnher',
        'belkin1', 'belvir', 'bewwre', 'bkbmag1', 'bkbplo',
        'bkbwar', 'bkcchi', 'bkhgro', 'bkmtou1', 'bknsti', 'blbgra1',
        'blbthr1', 'blcjay1', 'blctan1', 'blhpar1', 'blkpho',
        'blsspa1', 'blugrb1', 'blujay', 'bncfly', 'bnhcow', 'bobfly1',
        'bongul', 'botgra', 'brbmot1', 'brbsol1', 'brcvir1', 'brebla',
        'brncre', 'brnjay', 'brnthr', 'brratt1', 'brwhaw', 'brwpar1',
        'btbwar', 'btnwar', 'btywar', 'bucmot2', 'buggna', 'bugtan',
        'buhvir', 'bulori', 'burwar1', 'bushti', 'butsal1', 'buwtea',
        'cacgoo1', 'cacwre', 'calqua', 'caltow', 'cangoo', 'canwar',
        'carchi', 'carwre', 'casfin', 'caskin', 'caster1', 'casvir',
        'categr', 'ccbfin', 'cedwax', 'chbant1', 'chbchi', 'chbwre1',
        'chcant2', 'chispa', 'chswar', 'cinfly2', 'clanut', 'clcrob',
        'cliswa', 'cobtan1', 'cocwoo1', 'cogdov', 'colcha1', 'coltro1',
        'comgol', 'comgra', 'comloo', 'commer', 'compau', 'compot1',
        'comrav', 'comyel', 'coohaw', 'cotfly1', 'cowscj1', 'cregua1',
        'creoro1', 'crfpar', 'cubthr', 'daejun', 'dowwoo', 'ducfly', 'dusfly',
        'easblu', 'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eletro',
        'eucdov', 'eursta', 'fepowl', 'fiespa', 'flrtan1', 'foxspa', 'gadwal',
        'gamqua', 'gartro1', 'gbbgul', 'gbwwre1', 'gcrwar', 'gilwoo',
        'gnttow', 'gnwtea', 'gocfly1', 'gockin', 'gocspa', 'goftyr1',
        'gohque1', 'goowoo1', 'grasal1', 'grbani', 'grbher3', 'grcfly',
        'greegr', 'grekis', 'grepew', 'grethr1', 'gretin1', 'greyel',
        'grhcha1', 'grhowl', 'grnher', 'grnjay', 'grtgra', 'grycat',
        'gryhaw2', 'gwfgoo', 'haiwoo', 'heptan', 'hergul', 'herthr',
        'herwar', 'higmot1', 'hofwoo1', 'houfin', 'houspa', 'houwre',
        'hutvir', 'incdov', 'indbun', 'kebtou1', 'killde', 'labwoo', 'larspa',
        'laufal1', 'laugul', 'lazbun', 'leafly', 'leasan', 'lesgol', 'lesgre1',
        'lesvio1', 'linspa', 'linwoo1', 'littin1', 'lobdow', 'lobgna5', 'logshr',
        'lotduc', 'lotman1', 'lucwar', 'macwar', 'magwar', 'mallar3', 'marwre',
        'mastro1', 'meapar', 'melbla1', 'monoro1', 'mouchi', 'moudov', 'mouela1',
        'mouqua', 'mouwar', 'mutswa', 'naswar', 'norcar', 'norfli', 'normoc', 'norpar',
        'norsho', 'norwat', 'nrwswa', 'nutwoo', 'oaktit', 'obnthr1', 'ocbfly1',
        'oliwoo1', 'olsfly', 'orbeup1', 'orbspa1', 'orcpar', 'orcwar', 'orfpar',
        'osprey', 'ovenbi1', 'pabspi1', 'paltan1', 'palwar', 'pasfly', 'pavpig2',
        'phivir', 'pibgre', 'pilwoo', 'pinsis', 'pirfly1', 'plawre1', 'plaxen1',
        'plsvir', 'plupig2', 'prowar', 'purfin', 'purgal2', 'putfru1', 'pygnut',
        'rawwre1', 'rcatan1', 'rebnut', 'rebsap', 'rebwoo', 'redcro', 'reevir1',
        'rehbar1', 'relpar', 'reshaw', 'rethaw', 'rewbla', 'ribgul', 'rinkin1',
        'roahaw', 'robgro', 'rocpig', 'rotbec', 'royter1', 'rthhum', 'rtlhum',
        'ruboro1', 'rubpep1', 'rubrob', 'rubwre1', 'ruckin', 'rucspa1', 'rucwar',
        'rucwar1', 'rudpig', 'rudtur', 'rufhum', 'rugdov', 'rumfly1', 'runwre1',
        'rutjac1', 'saffin', 'sancra', 'sander', 'savspa', 'saypho', 'scamac1',
        'scatan', 'scbwre1', 'scptyr1', 'scrtan1', 'semplo', 'shicow', 'sibtan2',
        'sinwre1', 'sltred', 'smbani', 'snogoo', 'sobtyr1', 'socfly1', 'solsan',
        'sonspa', 'soulap1', 'sposan', 'spotow', 'spvear1', 'squcuc1', 'stbori',
        'stejay', 'sthant1', 'sthwoo1', 'strcuc1', 'strfly1', 'strsal1', 'stvhum2',
        'subfly', 'sumtan', 'swaspa', 'swathr', 'tenwar', 'thbeup1', 'thbkin',
        'thswar1', 'towsol', 'treswa', 'trogna1', 'trokin', 'tromoc', 'tropar',
        'tropew1', 'tuftit', 'tunswa', 'veery', 'verdin', 'vigswa', 'warvir',
        'wbwwre1', 'webwoo1', 'wegspa1', 'wesant1', 'wesblu', 'weskin', 'wesmea',
        'westan', 'wewpew', 'whbman1', 'whbnut', 'whcpar', 'whcsee1', 'whcspa',
        'whevir', 'whfpar1', 'whimbr', 'whiwre1', 'whtdov', 'whtspa', 'whwbec1',
        'whwdov', 'wilfly', 'willet1', 'wilsni1', 'wiltur', 'wlswar', 'wooduc',
        'woothr', 'wrenti', 'y00475', 'yebcha', 'yebela1', 'yebfly', 'yebori1',
        'yebsap', 'yebsee1', 'yefgra1', 'yegvir', 'yehbla', 'yehcar1', 'yelgro',
        'yelwar', 'yeofly1', 'yerwar', 'yeteup1', 'yetvir']

    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {
            "batch_size": 64,
            "num_workers": 16,
            "shuffle": True
        },
        "valid": {
            "batch_size": 64,
            "num_workers": 16,
            "shuffle": False
        },
        "test": {
            "batch_size": 64,
            "num_workers": 20,
            "shuffle": False
        }
    }

    ######################
    # Split #
    ######################
    split = "StratifiedKFold"
    split_params = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 1213
    }

    ######################
    # Model #
    ######################
    base_model_name = "tf_efficientnet_b0_ns"
    pooling = "max"
    pretrained = True
    num_classes = 397
    in_channels = 1

    ######################
    # Criterion #
    ######################
    loss_name = "BCEFocal2WayLoss"
    loss_params: dict = {}

    ######################
    # Optimizer #
    ######################
    optimizer_name = "Adam"
    base_optimizer = "Adam"
    optimizer_params = {
        "lr": 0.001
    }
    # For SAM optimizer
    base_optimizer = "Adam"

    ######################
    # Scheduler #
    ######################
    scheduler_name = "CosineAnnealingLR"
    scheduler_params = {
        "T_max": 10
    }


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


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def prepare_model_for_inference(model, path: Path):
    if not torch.cuda.is_available():
        ckpt = torch.load(path, map_location="cpu")
    else:
        ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def calc_duration_of_audio_clip(path: Path):
    y, _ = sf.read(path)
    return len(y)


# =================================================
# Split #
# =================================================
def get_split():
    if hasattr(model_selection, CFG.split):
        return model_selection.__getattribute__(CFG.split)(**CFG.split_params)
    else:
        return MultilabelStratifiedKFold(**CFG.split_params)


# =================================================
# Dataset #
# =================================================
def normalize_melspec(X: np.ndarray):
    eps = 1e-6
    mean = X.mean()
    X = X - mean
    std = X.std()
    Xstd = X / (std + eps)
    norm_min, norm_max = Xstd.min(), Xstd.max()
    if (norm_max - norm_min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


class WaveformDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 background_paths: list,
                 img_size=224,
                 waveform_transforms=None,
                 period=20,
                 validation=False,
                 n_mix=1):
        self.df = df
        self.datadir = datadir
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.period = period
        self.validation = validation
        self.background_paths = background_paths
        self.n_mix = n_mix

    def __len__(self):
        return len(self.df)

    def __read_audio(self, path: Path, duration: float):
        SR = 32000
        len_y = duration
        effective_length = SR * self.period
        if len_y < effective_length:
            buffer = np.zeros(effective_length, dtype=np.float32)
            if not self.validation:
                start = np.random.randint(effective_length - len_y)
            else:
                start = 0
            y, _ = sf.read(path)
            buffer[start:start + len_y] = y
            return buffer.astype(np.float32)
        elif len_y > effective_length:
            if not self.validation:
                start = np.random.randint(len_y - effective_length)
            else:
                start = 0
            y, _ = sf.read(path, frames=effective_length, start=start)
            return y.astype(np.float32)
        else:
            y, _ = sf.read(path)
            return y.astype(np.float32)

    def __resolve_nan(self, y: np.ndarray):
        y = np.nan_to_num(y)
        if np.isnan(y).any():
            y = np.zeros(len(y))
        return y

    def __normalize_audio(self, y: np.ndarray):
        return y / np.abs(y).max()

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        ebird_code = sample["primary_label"]
        secondary_labels = eval(sample["secondary_labels"])
        duration = sample["duration"]

        labels = np.zeros(len(CFG.target_columns), dtype=float)
        labels[CFG.target_columns.index(ebird_code)] = 1.0

        mask = np.ones(len(CFG.target_columns), dtype=float)
        for secondary_label in secondary_labels:
            if secondary_label in CFG.target_columns:
                mask[CFG.target_columns.index(secondary_label)] = 0.0

        y = self.__read_audio(self.datadir / ebird_code / wav_name, duration)

        y = self.__resolve_nan(y)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        y = self.__resolve_nan(y)

        if not self.validation:
            for _ in range(self.n_mix):
                if np.random.rand() < 0.5:
                    sample = self.df.sample(1).T.squeeze()
                    if sample.filename == wav_name:
                        continue
                    wav_name = sample["filename"]
                    ebird_code = sample["primary_label"]
                    secondary_labels = eval(sample["secondary_labels"])
                    duration = sample["duration"]

                    y_ = self.__read_audio(self.datadir / ebird_code / wav_name,
                                           duration)

                    y_ = self.__resolve_nan(y_)

                    y = self.__resolve_nan(y)

                    y_n = self.__normalize_audio(y)
                    y_n_ = self.__normalize_audio(y_)
                    y = y_n + y_n_
                    y = self.__normalize_audio(y)

                    labels[CFG.target_columns.index(ebird_code)] = 1.0
                    for secondary_label in secondary_labels:
                        if secondary_label in CFG.target_columns:
                            mask[CFG.target_columns.index(
                                secondary_label)] = 0.0

        mask = mask + labels
        mask[mask > 1.0] = 1.0

        SR = 32000
        background_length = SR * 60
        effective_length = SR * self.period
        start = np.random.randint(background_length - effective_length)
        background_path = np.random.choice(self.background_paths)
        y_background, sr = sf.read(
            background_path, start=start, frames=effective_length)
        y_background = y_background.astype(np.float32)

        y_background = self.__resolve_nan(y_background)

        y_normalized = self.__normalize_audio(y)
        y_background_normalized = self.__normalize_audio(y_background)

        rate = 0.35 + 0.5 * np.random.rand()
        y = rate * y_normalized + (1.0 - rate) * y_background_normalized

        y = y.astype(np.float32)
        y = self.__normalize_audio(y)

        y = self.__resolve_nan(y)

        return {
            "image": y,
            "targets": labels,
            "mask": mask
        }


class SingleChannelDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 img_size=224,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 period=20,
                 validation=False):
        self.df = df
        self.datadir = datadir
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.period = period
        self.validation = validation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        ebird_code = sample["primary_label"]

        y, sr = sf.read(self.datadir / ebird_code / wav_name)

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

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        melspec = librosa.feature.melspectrogram(
            y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(image=melspec)["image"]

        norm_melspec = normalize_melspec(melspec)
        height, width = norm_melspec.shape
        image = cv2.resize(
            norm_melspec, (int(width * self.img_size / height), self.img_size))
        height, width = image.shape
        image = image.reshape(1, height, width)
        image = (image / 255.0).astype(np.float32)

        labels = np.zeros(len(CFG.target_columns), dtype=float)
        labels[CFG.target_columns.index(ebird_code)] = 1.0

        return {
            "image": image,
            "targets": labels
        }


# =================================================
# Transforms #
# =================================================
def get_transforms(phase: str):
    transforms = CFG.transforms
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else \
                trns_conf["params"]
            if globals().get(trns_name) is not None:
                trns_cls = globals()[trns_name]
                trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return Compose(trns_list)
        else:
            return None


def get_waveform_transforms(config: dict, phase: str):
    return get_transforms(config, phase)


def get_spectrogram_transforms(config: dict, phase: str):
    transforms = config.get('spectrogram_transforms')
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else \
                trns_conf["params"]
            if hasattr(A, trns_name):
                trns_cls = A.__getattribute__(trns_name)
                trns_list.append(trns_cls(**trns_params))
            else:
                trns_cls = globals().get(trns_name)
                if trns_cls is not None:
                    trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return A.Compose(trns_list, p=1.0)
        else:
            return None


class Normalize:
    def __call__(self, y: np.ndarray):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class NewNormalize:
    def __call__(self, y: np.ndarray):
        y_mm = y - y.mean()
        return y_mm / y_mm.abs().max()


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5, sr=32000):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=32000):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=32000):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5, sr=32000):
        super().__init__(always_apply, p)
        self.max_range = max_range
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, self.sr, n_steps)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1, sr=32000):
        super().__init__(always_apply, p)
        self.max_rate = max_rate
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10**(db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        y = trns(y)
        return y


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs


def drop_stripes(image: np.ndarray, dim: int, drop_width: int, stripes_num: int):
    total_width = image.shape[dim]
    lowest_value = image.min()
    for _ in range(stripes_num):
        distance = np.random.randint(low=0, high=drop_width, size=(1,))[0]
        begin = np.random.randint(
            low=0, high=total_width - distance, size=(1,))[0]

        if dim == 0:
            image[begin:begin + distance] = lowest_value
        elif dim == 1:
            image[:, begin + distance] = lowest_value
        elif dim == 2:
            image[:, :, begin + distance] = lowest_value
    return image


class TimeFreqMasking(ImageOnlyTransform):
    def __init__(self,
                 time_drop_width: int,
                 time_stripes_num: int,
                 freq_drop_width: int,
                 freq_stripes_num: int,
                 always_apply=False,
                 p=0.5):
        super().__init__(always_apply, p)
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.freq_drop_width = freq_drop_width
        self.freq_stripes_num = freq_stripes_num

    def apply(self, img, **params):
        img_ = img.copy()
        if img.ndim == 2:
            img_ = drop_stripes(
                img_, dim=0, drop_width=self.freq_drop_width, stripes_num=self.freq_stripes_num)
            img_ = drop_stripes(
                img_, dim=1, drop_width=self.time_drop_width, stripes_num=self.time_stripes_num)
        return img_


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


class TimmModel(nn.Module):
    def __init__(self, base_model_name="tf_efficientnet_b0_ns", pooling="GeM", pretrained=True, num_classes=24, in_channels=1):
        super().__init__()
        self.base_model = timm.create_model(
            base_model_name, pretrained=pretrained, in_chans=in_channels)
        if hasattr(self.base_model, "fc"):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.base_model, "classifier"):
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise NotImplementedError

        if pooling == "GeM":
            self.base_model.global_pool = GeM()
        elif pooling == "max":
            self.base_model.global_pool = SelectAdaptivePool2d(
                pool_type="max", flatten=True)

        self.init_layer()

    def init_layer(self):
        init_layer(self.base_model.classifier)

    def forward(self, x):
        return self.base_model(x)


class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=24, in_channels=1):
        super().__init__()
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=CFG.n_fft, hop_length=CFG.hop_length,
                                                 win_length=CFG.n_fft, window="hann", center=True, pad_mode="reflect",
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=CFG.sample_rate, n_fft=CFG.n_fft,
                                                 n_mels=CFG.n_mels, fmin=CFG.fmin, fmax=CFG.fmax, ref=1.0, amin=1e-10, top_db=None,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(CFG.n_mels)

        base_model = timm.create_model(
            base_model_name, pretrained=pretrained, in_chans=in_channels)
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def forward(self, input):
        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        with torch.cuda.amp.autocast(False):
            # (batch_size, 1, time_steps, mel_bins)
            x = self.logmel_extractor(x)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x = x.transpose(2, 3)
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
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


# =================================================
# Optimizer and Scheduler #
# =================================================
version_higher = (torch.__version__ >= "1.5.0")


class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients
               NeurIPS 2020 Spotlight
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, weight_decouple=False, fixed_decay=False, rectify=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdaBelief, self).__init__(params, defaults)
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMS enabled in AdaBelief')

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']
                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(
                    p.data,
                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(
                    p.data,
                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                state = self.state[p]
                beta1, beta2 = group['betas']
                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(group['weight_decay'], p.data)
                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    1 - beta2, grad_residual, grad_residual)
                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var,
                              out=max_exp_avg_var)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.add_(group['eps']).sqrt(
                    ) / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt(
                    ) / math.sqrt(bias_correction2)).add_(group['eps'])
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:  # Rectified update
                    # calculate rho_t
                    state['rho_t'] = state['rho_inf'] - 2 * state['step'] * beta2 ** state['step'] / (
                        1.0 - beta2 ** state['step'])
                    if state['rho_t'] > 4:  # perform Adam style update if variance is small
                        rho_inf, rho_t = state['rho_inf'], state['rho_t']
                        rt = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf / \
                            (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t
                        rt = math.sqrt(rt)
                        step_size = rt * group['lr'] / bias_correction1
                        p.data.addcdiv_(-step_size, exp_avg, denom)
                    else:  # perform SGD style update
                        p.data.add_(-group['lr'], exp_avg)
        return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # the closure should do a full forward-backward pass
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                        ]),
            p=2
        )
        return norm


__OPTIMIZERS__ = {
    "AdaBelief": AdaBelief,
    "SAM": SAM,
}


def get_optimizer(model: nn.Module):
    optimizer_name = CFG.optimizer_name
    if optimizer_name == "SAM":
        base_optimizer_name = CFG.base_optimizer
        if __OPTIMIZERS__.get(base_optimizer_name) is not None:
            base_optimizer = __OPTIMIZERS__[base_optimizer_name]
        else:
            base_optimizer = optim.__getattribute__(base_optimizer_name)
        return SAM(model.parameters(), base_optimizer, **CFG.optimizer_params)

    if __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(),
                                              **CFG.optimizer_params)
    else:
        return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                      **CFG.optimizer_params)


def get_scheduler(optimizer):
    scheduler_name = CFG.scheduler_name

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **CFG.scheduler_params)


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

    def forward(self, input, target, mask=None):
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


def get_criterion():
    if hasattr(nn, CFG.loss_name):
        return nn.__getattribute__(CFG.loss_name)(**CFG.loss_params)
    elif __CRITERIONS__.get(CFG.loss_name) is not None:
        return __CRITERIONS__[CFG.loss_name](**CFG.loss_params)
    else:
        raise NotImplementedError


# =================================================
# Callbacks #
# =================================================
class SchedulerCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Scheduler)

    def on_loader_end(self, state: IRunner):
        lr = state.scheduler.get_last_lr()
        state.epoch_metrics["lr"] = lr[0]
        if state.is_train_loader:
            state.scheduler.step()


class SampleF1Callback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "f1",
                 threshold=0.5):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.threshold = threshold

    def on_loader_start(self, state: IRunner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: IRunner):
        targ = state.input[self.input_key].detach().cpu().numpy()
        out = state.output[self.output_key]

        clipwise_output = out["clipwise_output"].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        y_pred = clipwise_output > self.threshold
        score = metrics.f1_score(targ, y_pred, average="samples")

        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: IRunner):
        y_pred = np.concatenate(self.prediction, axis=0) > self.threshold
        y_true = np.concatenate(self.target, axis=0)
        score = metrics.f1_score(y_true, y_pred, average="samples")

        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" +
                                self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score


class mAPCallback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 model_output_key: str = "clipwise_output",
                 prefix: str = "mAP"):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix

    def on_loader_start(self, state: IRunner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: IRunner):
        targ = state.input[self.input_key].detach().cpu().numpy()
        out = state.output[self.output_key]

        clipwise_output = out[self.model_output_key].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        try:
            score = metrics.average_precision_score(
                targ, clipwise_output, average=None)
        except ValueError:
            import pdb
            pdb.set_trace()
        score = np.nan_to_num(score).mean()
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: IRunner):
        y_pred = np.concatenate(self.prediction, axis=0)
        y_true = np.concatenate(self.target, axis=0)
        score = metrics.average_precision_score(y_true, y_pred, average=None)
        score = np.nan_to_num(score).mean()
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" +
                                self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score


def get_callbacks():
    return [
        SchedulerCallback(),
        SampleF1Callback(prefix="f1_at_05", threshold=0.5),
        SampleF1Callback(prefix="f1_at_03", threshold=0.3),
        SampleF1Callback(prefix="f1_at_07", threshold=0.7),
        mAPCallback()
    ]


# =================================================
# Runner #
# =================================================
class SAMRunner(Runner):
    def predict_batch(self, batch, **kwargs):
        return super().predict_batch(batch, **kwargs)

    def _handle_batch(self, batch):
        input_, target = batch["image"], batch["targets"]
        mask = batch["mask"]

        input_ = input_.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)

        out = self.model(input_)

        loss = self.criterion(out, target, mask)
        self.batch_metrics.update({
            "loss": loss
        })

        self.input = batch
        self.output = {"logits": out}

        if self.is_train_loader:
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            self.criterion(self.model(input_), target, mask).backward()
            self.optimizer.second_step(zero_grad=True)


class MaslkedLossRunner(Runner):
    def predict_batch(self, batch, **kwargs):
        return super().predict_batch(batch, **kwargs)

    def _handle_batch(self, batch):
        input_, target = batch["image"], batch["targets"]
        mask = batch["mask"]

        input_ = input_.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)

        out = self.model(input_)

        loss = self.criterion(out, target, mask)
        self.batch_metrics.update({
            "loss": loss
        })

        self.input = batch
        self.output = {"logits": out}

        if self.is_train_loader:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def get_runner(device: torch.device):
    if CFG.optimizer_name == "SAM":
        return SAMRunner(device=device)
    else:
        return MaslkedLossRunner(device=device)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # logging
    filename = __file__.split("/")[-1].replace(".py", "")
    logdir = Path(f"../out/{filename}")
    logdir.mkdir(exist_ok=True, parents=True)
    if (logdir / "train.log").exists():
        os.remove(logdir / "train.log")
    logger = init_logger(log_file=logdir / "train.log")

    # environment
    set_seed(CFG.seed)
    device = get_device()

    # validation
    splitter = get_split()

    # data
    train = pd.read_csv(CFG.train_csv)
    if "duration" not in train.columns:
        durations = Parallel(n_jobs=20, verbose=10)([
            delayed(calc_duration_of_audio_clip)(
                CFG.train_datadir / ebird_code / wav_name)
            for ebird_code, wav_name in zip(
                train["primary_label"].values,
                train["filename"].values
            )
        ])
        train["duration"] = durations
        train.to_csv(CFG.train_csv, index=False)

    train_background_paths = list(CFG.train_background.glob("*.wav"))

    if CFG.train:
        for i, (trn_idx, val_idx) in enumerate(splitter.split(train, y=train["primary_label"])):
            if i not in CFG.folds:
                continue
            logger.info("=" * 120)
            logger.info(f"Fold {i} Training")
            logger.info("=" * 120)

            trn_df = train.loc[trn_idx, :].reset_index(drop=True)
            val_df = train.loc[val_idx, :].reset_index(drop=True)

            loaders = {
                phase: torchdata.DataLoader(
                    WaveformDataset(
                        df_,
                        CFG.train_datadir,
                        background_paths=train_background_paths,
                        img_size=CFG.img_size,
                        waveform_transforms=get_transforms(phase),
                        period=CFG.period,
                        validation=(phase == "valid")
                    ),
                    **CFG.loader_params[phase])  # type: ignore
                for phase, df_ in zip(["train", "valid"], [trn_df, val_df])
            }

            model = TimmSED(
                base_model_name=CFG.base_model_name,
                pretrained=CFG.pretrained,
                num_classes=CFG.num_classes,
                in_channels=CFG.in_channels)
            criterion = get_criterion()
            optimizer = get_optimizer(model)
            scheduler = get_scheduler(optimizer)
            callbacks = get_callbacks()
            runner = get_runner(device)
            runner.train(
                model=model,
                criterion=criterion,
                loaders=loaders,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=CFG.epochs,
                verbose=True,
                logdir=logdir / f"fold{i}",
                callbacks=callbacks,
                main_metric=CFG.main_metric,
                minimize_metric=CFG.minimize_metric,
                fp16=True)

            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
