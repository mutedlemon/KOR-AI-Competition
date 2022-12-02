# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import logging
import platform
# from omegaconf import DictConfig

from modules.kospeech.optim.lr_scheduler.lr_scheduler import LearningRateScheduler
from modules.kospeech.vocabs import Vocabulary
from torch import optim
from modules.kospeech.optim import (
    RAdam,
    AdamP,
    Novograd,
)
from modules.kospeech.criterion import (
    LabelSmoothedCrossEntropyLoss,
    JointCTCCrossEntropyLoss,
    TransducerLoss,
)
from modules.kospeech.optim.lr_scheduler import (
    TriStageLRScheduler,
    TransformerLRScheduler,
)


logger = logging.getLogger(__name__)


def check_envirionment(use_cuda: bool) -> torch.device:
    """
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    logger.info(f"Operating System : {platform.system()} {platform.release()}")
    logger.info(f"Processor : {platform.processor()}")

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
        logger.info(f"PyTorch version : {torch.__version__}")

    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"PyTorch version : {torch.__version__}")

    return device


def get_optimizer(model: nn.Module, config):
    supported_optimizer = {
        'adam': optim.Adam,
        'radam': RAdam,
        'adamp': AdamP,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'novograd': Novograd,
    }
    assert config.optimizer.lower() in supported_optimizer.keys(), \
        f"Unsupported Optimizer: {config.optimizer}\n" \
        f"Supported Optimizer: {supported_optimizer.keys()}"

    if config.architecture == 'conformer':
        return optim.Adam(
            model.parameters(),
            betas=config.optimizer_betas,
            eps=config.optimizer_eps,
            weight_decay=config.weight_decay,
        )

    return supported_optimizer[config.optimizer](
        model.module.parameters(),
        lr=config.init_lr,
        weight_decay=config.weight_decay,
    )


def get_criterion(config, vocab: Vocabulary) -> nn.Module:
    if config.architecture in ('deepspeech2', 'jasper'):
        criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=config.reduction, zero_infinity=True)
    elif config.architecture in ('las', 'transformer') and config.joint_ctc_attention:
        criterion = JointCTCCrossEntropyLoss(
            num_classes=len(vocab),
            ignore_index=vocab.pad_id,
            reduction=config.reduction,
            ctc_weight=config.ctc_weight,
            cross_entropy_weight=config.cross_entropy_weight,
            blank_id=vocab.blank_id,
            dim=-1,
            smoothing=config.label_smoothing,
        )
    elif config.architecture == 'conformer':
        if config.decoder == 'rnnt':
            criterion = TransducerLoss(blank_id=vocab.blank_id)
        else:
            criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=config.reduction, zero_infinity=True)
    elif config.architecture == 'rnnt':
        criterion = TransducerLoss(blank_id=vocab.blank_id)
    elif config.architecture == 'transformer' and config.label_smoothing <= 0.0:
        criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_id,
            reduction=config.reduction,
        )
    else:
        criterion = LabelSmoothedCrossEntropyLoss(
            num_classes=len(vocab),
            ignore_index=vocab.pad_id,
            smoothing=config.label_smoothing,
            reduction=config.reduction,
            dim=-1,
        )

    return criterion


def get_lr_scheduler(config, optimizer, epoch_time_step) -> LearningRateScheduler:
    if config.lr_scheduler == "tri_stage_lr_scheduler":
        lr_scheduler = TriStageLRScheduler(
            optimizer=optimizer,
            init_lr=config.init_lr,
            peak_lr=config.peak_lr,
            final_lr=config.final_lr,
            init_lr_scale=config.init_lr_scale,
            final_lr_scale=config.final_lr_scale,
            warmup_steps=config.warmup_steps,
            total_steps=int(config.num_epochs * epoch_time_step),
        )
    elif config.lr_scheduler == "transformer_lr_scheduler":
        lr_scheduler = TransformerLRScheduler(
            optimizer=optimizer,
            peak_lr=config.peak_lr,
            final_lr=config.final_lr,
            final_lr_scale=config.final_lr_scale,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
        )
    else:
        raise ValueError(f"Unsupported Learning Rate Scheduler: {config.lr_scheduler}")

    return lr_scheduler
