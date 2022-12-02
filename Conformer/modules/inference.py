import os
import numpy as np
import torchaudio

import torch
import torch.nn as nn
from torch import Tensor

from modules.vocab import KoreanSpeechVocabulary
from modules.data import load_audio
import re


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        labels = list(self.labels)
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)
    return torch.FloatTensor(feature).transpose(0, 1)


def single_infer(model, audio_path):
    device = 'cuda'
    feature = parse_audio(audio_path, del_silence=True)
    input_length = torch.LongTensor([len(feature)])
    vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')

    if isinstance(model, nn.DataParallel):
        model = model.module

    greedy_decoder = GreedyCTCDecoder(labels=vocab.labels, blank=vocab.real_blank_id)

    model.set_decoder(greedy_decoder)

    model.eval()

    model.device = device
    y_hats = model.recognize(feature.unsqueeze(0).to(device), input_length)
    y_hats_revised = re.sub("[^가-힣 ]", "", y_hats[0])
    return y_hats_revised