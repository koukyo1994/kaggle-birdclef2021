import gc
import os
import random
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torchaudio
import torchaudio.transforms as T

from joblib import Parallel, delayed
from pathlib import Path
from typing import Dict, List

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn import model_selection
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
    folds = [0, 1, 2, 3, 4]
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
        "train": [{"name": "TorchPinkNoise",  "params": {"min_snr": 5}}],
        "valid": None
    }
    period = 20
    n_mels = 224
    fmin = 300
    fmax = 16000
    n_fft = 2048
    hop_length = 768
    sample_rate = 32000

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
            "batch_size": 48,
            "num_workers": 20,
            "shuffle": True
        },
        "valid": {
            "batch_size": 64,
            "num_workers": 20,
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
    weights_paths = [
        "../out/exp032_027_5fold/fold0/checkpoints/best.pth",
        "../out/exp032_027_5fold/fold1/checkpoints/best.pth",
        "../out/exp032_027_5fold/fold2/checkpoints/best.pth",
        "../out/exp032_027_5fold/fold3/checkpoints/best.pth",
        "../out/exp032_027_5fold/fold4/checkpoints/best.pth"
    ]

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


def check_allzero(path: Path):
    y = torchaudio.load(path)[0][0].float()
    if (y == 0.0).all():
        return True
    else:
        return False


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
class WaveformInferenceDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 waveform_transforms=None,
                 period=20):
        df["seconds"] = df["duration"] / 32000
        df["n_chunks"] = df["seconds"].map(
            lambda x: int(x / period) if x % period == 0 else int(x / period) + 1)

        filenames = []
        primary_labels = []
        chunk_indices = []
        for _, row in df.iterrows():
            if row.allzero:
                continue
            n_chunks = row.n_chunks
            filename = row.filename
            primary_label = row.primary_label
            for i in range(n_chunks):
                filenames.append(filename)
                primary_labels.append(primary_label)
                chunk_indices.append(i)
        self.filenames = filenames
        self.primary_labels = primary_labels
        self.chunk_indices = chunk_indices

        self.datadir = datadir

        self.waveform_transforms = waveform_transforms
        self.period = period

    def __len__(self):
        return len(self.filenames)

    def __read_audio(self, path: Path, index: int):
        SR = 32000
        num_frames = self.period * SR
        frame_offset = index * self.period * SR
        y = torchaudio.load(
            path, num_frames=num_frames, frame_offset=frame_offset)[0][0].float()
        if len(y) / SR != self.period:
            buffer = torch.zeros(SR * self.period, dtype=torch.float)
            buffer[:len(y)] = y
            return buffer
        return y

    def __normalize_audio(self, y: torch.FloatTensor):
        if y.abs().max() > 0:
            return y / y.abs().max()
        else:
            return y

    def __getitem__(self, idx: int):
        wav_name = self.filenames[idx]
        ebird_code = self.primary_labels[idx]
        chunk_index = self.chunk_indices[idx]

        y = self.__read_audio(
            self.datadir / ebird_code / wav_name, chunk_index)
        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        y = self.__normalize_audio(y)
        return {
            "audio": y,
            "filename": wav_name
        }


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


class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=24, in_channels=1):
        super().__init__()
        # melextractor
        self.logmel_extractor = nn.Sequential(
            T.MelSpectrogram(sample_rate=CFG.sample_rate, n_fft=CFG.n_fft, win_length=CFG.n_fft,
                             hop_length=CFG.hop_length, power=2.0, n_mels=CFG.n_mels),
            T.AmplitudeToDB()
        )

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
        with torch.cuda.amp.autocast(False):
            x = self.logmel_extractor(input).unsqueeze(1)

        frames_num = x.size(3)

        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        if self.training:
            x = self.spec_augmenter(x.transpose(2, 3)).transpose(2, 3)

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

        output_dict = {
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "clipwise_output": clipwise_output
        }

        return output_dict


# ==============================
# Inference Loop
# ==============================
def do_inference(loader: torchdata.DataLoader,
                 model: nn.Module,
                 device: torch.device):
    filenames: List[str] = []
    predictions: List[np.ndarray] = []
    filenames2predictions: Dict[str, List[np.ndarray]] = {}
    model.eval()
    for batch in tqdm(loader, desc="inference"):
        filenames.extend(batch["filename"])
        audio = batch["audio"].to(device)
        with torch.no_grad():
            out = model(audio)
            segmentwise_output = out["segmentwise_output"].detach(
            ).cpu().numpy()
        predictions.append(segmentwise_output)
    predictions = np.concatenate(predictions, axis=0)

    for i, filename in enumerate(filenames):
        if filenames2predictions.get(filename) is None:
            filenames2predictions[filename] = [predictions[i]]
        else:
            filenames2predictions[filename].append(predictions[i])

    keys = list(filenames2predictions.keys())
    for key in keys:
        filenames2predictions[key] = np.concatenate(
            filenames2predictions[key], axis=0)
    return filenames2predictions


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # logging
    filename = __file__.split("/")[-1].replace(".py", "")
    logdir = Path(f"../out/{filename}")
    logdir.mkdir(exist_ok=True, parents=True)
    (logdir / "soft_labels").mkdir(exist_ok=True, parents=True)
    if (logdir / "inference.log").exists():
        os.remove(logdir / "inference.log")
    logger = init_logger(log_file=logdir / "inference.log")

    # environment
    set_seed(CFG.seed)
    device = get_device()

    # validation
    splitter = get_split()

    # data
    train = pd.read_csv(CFG.train_csv)
    if "allzero" not in train.columns:
        allzero = Parallel(n_jobs=20, verbose=10)([
            delayed(check_allzero)(
                CFG.train_datadir / ebird_code / wav_name)
            for ebird_code, wav_name in zip(
                train["primary_label"].values,
                train["filename"].values
            )
        ])
        train["allzero"] = allzero
        train.to_csv(CFG.train_csv, index=False)

    for i, (trn_idx, val_idx) in enumerate(splitter.split(train, y=train["primary_label"])):
        logger.info("=" * 120)
        logger.info(f"Fold {i} Inference")
        logger.info("=" * 120)

        val_df = train.loc[val_idx, :].reset_index(drop=True)
        dataset = WaveformInferenceDataset(
            val_df,
            CFG.train_datadir,
            waveform_transforms=None,
            period=CFG.period)
        loader = torchdata.DataLoader(dataset, **CFG.loader_params["valid"])
        model = TimmSED(
            base_model_name=CFG.base_model_name,
            pretrained=CFG.pretrained,
            num_classes=CFG.num_classes,
            in_channels=CFG.in_channels)
        model = prepare_model_for_inference(
            model, CFG.weights_paths[i]).to(device)
        filenames2predictions = do_inference(loader, model, device)
        for key in tqdm(filenames2predictions, desc="Saving soft labels", total=len(filenames2predictions)):
            np.savez_compressed(
                logdir / "soft_labels" / key,
                filenames2predictions[key])
