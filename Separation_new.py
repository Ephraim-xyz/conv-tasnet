#!/usr/bin/env python

# wujian@2018

import sys
sys.path.append('./options')

import os
import argparse

import torch as th
import numpy as np
from utils import get_logger
from Conv_TasNet_3_Unet import ConvTasNet

import tqdm
from option import parse
logger = get_logger(__name__)
import optparse
import numpy as np
import scipy.io.wavfile as wf
import os
MAX_INT16 = np.iinfo(np.int16).max


def write_wav(fname, samps, fs=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    if normalize:
        samps = samps * MAX_INT16
    # scipy.io.wavfile.write could write single/multi-channel files
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)
    # same as MATLAB and kaldi
    samps_int16 = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wf.write(fname, fs, samps_int16)


def read_wav(fname, normalize=True, return_rate=False):
    """
    Read wave files using scipy.io.wavfile(support multi-channel)
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    samp_rate, samps_int16 = wf.read(fname)
    # N x C => C x N
    samps = samps_int16.astype(np.float)
    # tranpose because I used to put channel axis first
    if samps.ndim != 1:
        samps = np.transpose(samps)
    # normalize like MATLAB and librosa
    if normalize:
        samps = samps / MAX_INT16
    if return_rate:
        return samp_rate, samps
    return samps


def parse_scripts(scp_path, value_processor=lambda x: x, num_tokens=2):
    """
    Parse kaldi's script(.scp) file
    If num_tokens >= 2, function will check token number
    """
    scp_dict = dict()
    line = 0
    with open(scp_path, "r") as f:
        for raw_line in f:
            scp_tokens = raw_line.strip().split()
            line += 1
            if num_tokens >= 2 and len(scp_tokens) != num_tokens or len(
                    scp_tokens) < 2:
                raise RuntimeError(
                    "For {}, format error in line[{:d}]: {}".format(
                        scp_path, line, raw_line))
            if num_tokens == 2:
                key, value = scp_tokens
            else:
                key, value = scp_tokens[0], scp_tokens[1:]
            if key in scp_dict:
                raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                    key, scp_path))
            scp_dict[key] = value_processor(value)
    return scp_dict


class Reader(object):
    """
        Basic Reader Class
    """

    def __init__(self, scp_path, value_processor=lambda x: x):
        self.index_dict = parse_scripts(
            scp_path, value_processor=value_processor, num_tokens=2)
        self.index_keys = list(self.index_dict.keys())

    def _load(self, key):
        # return path
        return self.index_dict[key]

    # number of utterance
    def __len__(self):
        return len(self.index_dict)

    # avoid key error
    def __contains__(self, key):
        return key in self.index_dict

    # sequential index
    def __iter__(self):
        for key in self.index_keys:
            yield key, self._load(key)

    # random index, support str/int as index
    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError("Unsupported index type: {}".format(type(index)))
        if type(index) == int:
            # from int index to key
            num_utts = len(self.index_keys)
            if index >= num_utts or index < 0:
                raise KeyError(
                    "Interger index out of range, {:d} vs {:d}".format(
                        index, num_utts))
            index = self.index_keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))
        return self._load(index)


class WaveReader(Reader):
    """
        Sequential/Random Reader for single channel wave
        Format of wav.scp follows Kaldi's definition:
            key1 /path/to/wav
            ...
    """

    def __init__(self, wav_scp, sample_rate=None, normalize=True):
        super(WaveReader, self).__init__(wav_scp)
        self.samp_rate = sample_rate
        self.normalize = normalize

    def _load(self, key):
        # return C x N or N
        samp_rate, samps = read_wav(
            self.index_dict[key], normalize=self.normalize, return_rate=True)
        # if given samp_rate, check it
        if self.samp_rate is not None and samp_rate != self.samp_rate:
            raise RuntimeError("SampleRate mismatch: {:d} vs {:d}".format(
                samp_rate, self.samp_rate))
        return samps



class NnetComputer(object):
    def __init__(self, cpt_dir, gpuid, model):
        self.device = th.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else th.device("cpu")
        opt = parse(cpt_dir, is_tain=False)
        nnet = self._load_nnet(opt, model)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()

    def _load_nnet(self, opt, model):
        nnet = ConvTasNet(**opt['net_conf'])
        cpt = th.load(model, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            model, cpt["epoch"]))
        return nnet

    def compute(self, samps):
        with th.no_grad():
            raw = th.tensor(samps, dtype=th.float32, device=self.device)
            sps = self.nnet(raw)
            sp_samps = [np.squeeze(s.detach().cpu().numpy()) for s in sps]
            return sp_samps


def run(args, gpuid):
    mix_input = WaveReader(args.mix_scp, sample_rate=args.fs)
    computer = NnetComputer(args.yaml, gpuid, args.model)
    for key, mix_samps in tqdm.tqdm(mix_input):
        #logger.info("Compute on utterance {}...".format(key))
        spks = computer.compute(mix_samps)
        norm = np.linalg.norm(mix_samps, np.inf)
        for idx, samps in enumerate(spks):
            samps = samps[:mix_samps.size]
            # norm
            samps = samps * norm / np.max(np.abs(samps))
            write_wav(
                os.path.join(args.save_path, "s{}/{}.wav".format(
                    idx + 1, key)),
                samps,
                fs=args.fs)
    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '-mix_scp', type=str, default='../create_scp/tt_mix.scp', help='Path to mix scp file.')
    parser.add_argument(
        '-yaml', type=str, default='./options/train/train_3_unet.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='./Conv_Tasnet_3_Unet_2/best.pt', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='./result', help='save result path')
    parser.add_argument(
        '-fs', type=str, default=8000, help='save result path')
    args=parser.parse_args()
    gpuid=0
    run(args, gpuid)