import torch
import torch.nn as nn
import numpy as np
import math
from dataclasses import dataclass
import time
from nsml import DATASET_PATH
import nsml

def trainer(mode, config, dataloader, optimizer, model, criterion, metric, train_begin_time, device):

    log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] {mode} Start')
    epoch_begin_time = time.time()
    cnt = 0

    # 1. train_epoches
    architecture = config.architecture
    if architecture == 'conformer':
        if isinstance(model, nn.DataParallel):
            architecture = 'conformer_t' if model.module.decoder is not None else 'conformer_ctc'
        else:
            architecture = 'conformer_t' if model.decoder is not None else 'conformer_ctc'

    for inputs, targets, input_lengths, target_lengths in dataloader:
        begin_time = time.time()
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        model = model.to(device)

        # 2. model_forward

        # deepspeech2
        # outputs, output_lengths = model(inputs, input_lengths)
        # loss = criterion(
        #     outputs.transpose(0, 1),
        #     targets[:, 1:],
        #     tuple(output_lengths),
        #     tuple(target_lengths)
        # )
        # y_hats = outputs.max(-1)[1]
        if architecture in ('rnnt', 'conformer_t'):
            outputs = model(inputs, input_lengths, targets, target_lengths)
            loss = criterion(
                outputs, targets[:, 1:].contiguous().int(), input_lengths.int(), target_lengths.int()
            )
        elif architecture == 'conformer_ctc':
            outputs, output_lengths = model(inputs, input_lengths, targets, target_lengths)
            loss = criterion(outputs.transpose(0, 1), targets[:, 1:], output_lengths, target_lengths)

        if architecture not in ('rnnt', 'conformer_t'):
            y_hats = outputs.max(-1)[1]
            # cer = metric(targets[:, 1:], y_hats)

        # if isinstance(model, nn.DataParallel):
        #     y_hats = model.module.recognize(inputs, input_lengths)
        # else:
        #     y_hats = model.recognize(inputs, input_lengths)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(model)

        total_num += int(input_lengths.sum())
        epoch_loss_total += loss.item()

        torch.cuda.empty_cache()

        if cnt % config.print_every == 0:

            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0
            train_elapsed = (current_time - train_begin_time) / 3600.0
            cer = metric(targets[:, 1:], y_hats)
            print(log_format.format(
                cnt, len(dataloader), loss,
                cer, elapsed, epoch_elapsed, train_elapsed,
                optimizer.get_lr(),
            ))

        # if cnt % 1000 == 0:
        #     nsml.save(cnt)
        cnt += 1
    return model, epoch_loss_total/len(dataloader), metric(targets[:, 1:], y_hats)


