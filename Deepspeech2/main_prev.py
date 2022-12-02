import torch
import os
import random
import warnings
import argparse
from glob import glob

from modules.preprocess import preprocessing
from modules.kospeech.utils import (
    get_optimizer,
    get_criterion,
    get_lr_scheduler,
)
from modules.kospeech.model_builder import build_model
from modules.vocab import KoreanSpeechVocabulary
# from modules.data import split_dataset, collate_fn
from modules.kospeech.data.data_loader import split_dataset
from modules.kospeech.optim import Optimizer
from modules.kospeech.metrics import get_metric
from modules.inference import single_infer
import torch.nn as nn

import nsml
from nsml import DATASET_PATH

from modules.kospeech.trainer.supervised_trainer import (
    SupervisedTrainer,
)

def revise(sentence):
    words = sentence[0].split()
    result = []
    for word in words:
        tmp = ''
        for t in word:
            if not tmp:
                tmp += t
            elif tmp[-1]!= t:
                tmp += t
        if tmp == '스로':
            tmp = '스스로'
        result.append(tmp)
    return ' '.join(result)

def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.


def inference(path, model, **kwargs):
    model.eval()

    results = []
    for i in glob(os.path.join(path, '*')):
        print(single_infer(model, i))
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer(model, i)
            }
        )
    return sorted(results, key=lambda x: x['filename'])

if __name__ == '__main__':
    # import subprocess
    # subprocess.call(['pip', 'install', '--upgrade', 'pip'])
    # subprocess.call(['pip', 'install', 'rnnt_warp'])

    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args.add_argument('--use_cuda', type=bool, default=True)
    args.add_argument('--seed', type=int, default=777)
    # args.add_argument('--num_epochs', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--save_result_every', type=int, default=10)
    args.add_argument('--checkpoint_every', type=int, default=1)
    args.add_argument('--print_every', type=int, default=50)
    args.add_argument('--dataset', type=str, default='kspon')
    args.add_argument('--output_unit', type=str, default='character')
    args.add_argument('--num_workers', type=int, default=8)
    args.add_argument('--num_threads', type=int, default=16)
    args.add_argument('--init_lr', type=float, default=1e-06)
    args.add_argument('--final_lr', type=float, default=1e-06)
    args.add_argument('--peak_lr', type=float, default=1e-04)
    args.add_argument('--init_lr_scale', type=float, default=1e-02)
    args.add_argument('--final_lr_scale', type=float, default=5e-02)
    args.add_argument('--max_grad_norm', type=int, default=400)
    # args.add_argument('--warmup_steps', type=int, default=1000)
    args.add_argument('--weight_decay', type=float, default=1e-05)
    args.add_argument('--reduction', type=str, default='mean')
    args.add_argument('--optimizer', type=str, default='adam')
    # args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler')
    args.add_argument('--total_steps', type=int, default=200000)

    args.add_argument('--architecture', type=str, default='deepspeech2')
    # args.add_argument('--use_bidirectional', type=bool, default=True)
    # args.add_argument('--dropout', type=float, default=3e-01)
    # args.add_argument('--num_encoder_layers', type=int, default=3)
    args.add_argument('--hidden_dim', type=int, default=1024)
    # args.add_argument('--rnn_type', type=str, default='gru')
    args.add_argument('--max_len', type=int, default=400)
    args.add_argument('--activation', type=str, default='hardtanh')
    args.add_argument('--teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--teacher_forcing_step', type=float, default=0.0)
    args.add_argument('--min_teacher_forcing_ratio', type=float, default=1.0)
    # args.add_argument('--joint_ctc_attention', type=bool, default=True)
    args.add_argument('--audio_extension', type=str, default='pcm')
    args.add_argument('--transform_method', type=str, default='fbank')
    args.add_argument('--feature_extract_by', type=str, default='kaldi')
    args.add_argument('--sample_rate', type=int, default=16000)
    args.add_argument('--frame_length', type=int, default=20)
    args.add_argument('--frame_shift', type=int, default=10)
    args.add_argument('--n_mels', type=int, default=80)
    args.add_argument('--freq_mask_para', type=int, default=18)
    args.add_argument('--time_mask_num', type=int, default=4)
    args.add_argument('--freq_mask_num', type=int, default=2)
    args.add_argument('--normalize', type=bool, default=True)
    args.add_argument('--del_silence', type=bool, default=True)
    args.add_argument('--spec_augment', type=bool, default=True)
    args.add_argument('--input_reverse', type=bool, default=False)

    if args.parse_args().architecture.lower() == 'rnnt':
        args.add_argument('--num_encoder_layers', type=int, default=4)
        args.add_argument('--num_decoder_layers', type=int, default=1)
        args.add_argument('--encoder_hidden_state_dim', type=int, default=320)
        args.add_argument('--decoder_hidden_state_dim', type=int, default=512)
        args.add_argument('--output_dim', type=int, default=512)
        args.add_argument('--rnn_type', type=str, default="lstm")
        args.add_argument('--bidirectional', type=bool, default=True)
        args.add_argument('--encoder_dropout_p', type=float, default=0.2)
        args.add_argument('--decoder_dropout_p', type=float, default=0.2)
        args.add_argument('--label_smoothing', type=float, default=0.1)
        args.add_argument('--joint_ctc_attention', type=bool, default=False)
        args.add_argument('--num_epochs', type=int, default=20)
        args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler')
        args.add_argument('--warmup_steps', type=int, default=400)
        # rnnt train 환경설정
        # optimizer: str = "adam"
        # init_lr: float = 1e-06
        # final_lr: float = 1e-06
        # peak_lr: float = 1e-04
        # warmup_steps: int = 400
        # num_epochs: int = 20
        # reduction: str = "mean"
        # label_smoothing: float = 0.1
        # lr_scheduler: str = 'tri_stage_lr_scheduler'

    if args.parse_args().architecture.lower() == 'transformer':
        args.add_argument('--extractor', type=str, default="vgg")
        args.add_argument('--use_bidirectional', type=bool, default=True)
        args.add_argument('--dropout', type=float, default=0.3)
        args.add_argument('--d_model', type=int, default=512)
        args.add_argument('--d_ff', type=int, default=2048)
        args.add_argument('--num_heads', type=int, default=8)
        args.add_argument('--num_encoder_layers', type=int, default=12)
        args.add_argument('--num_decoder_layers', type=int, default=6)
        args.add_argument('--label_smoothing', type=float, default=0.0)
        args.add_argument('--decay_steps', type=int, default=80000)
        args.add_argument('--cross_entropy_weight', type=float, default=0.7)
        args.add_argument('--ctc_weight', type=float, default=0.3)
        args.add_argument('--mask_conv', type=bool, default=True)
        args.add_argument('--joint_ctc_attention', type=bool, default=True)
        args.add_argument('--num_epochs', type=int, default=40)
        args.add_argument('--lr_scheduler', type=str, default='transformer_lr_scheduler')
        args.add_argument('--warmup_steps', type=int, default=4000)
        # transformer train 환경설정
        # optimizer: str = "adam"
        # init_lr: float = 1e-06
        # final_lr: float = 1e-06
        # peak_lr: float = 1e-04

        # warmup_steps: int = 4000
        # decay_steps: int = 80000
        # num_epochs: int = 40
        # reduction: str = "mean"
        # label_smoothing: float = 0.0
        # lr_scheduler: str = 'transformer_lr_scheduler'

    config = args.parse_args()
    warnings.filterwarnings('ignore')
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = 'cuda' if config.use_cuda == True else 'cpu'
    if hasattr(config, "num_threads") and int(config.num_threads) > 0:
        torch.set_num_threads(config.num_threads)

    vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')

    model = build_model(config, vocab, device)
    optimizer = get_optimizer(model, config)
    bind_model(model, optimizer=optimizer)
    # metric = get_metric(metric_name='CER', vocab=vocab)

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        config.dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data')
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
        preprocessing(label_path, os.getcwd())

        epoch_time_step, trainset_list, validset = split_dataset(config, os.path.join(os.getcwd(), 'transcripts.txt'),
                                                                 vocab)
        lr_scheduler = get_lr_scheduler(config, optimizer, epoch_time_step)

        optimizer = Optimizer(optimizer, lr_scheduler, config.total_steps, config.max_grad_norm)
        criterion = get_criterion(config, vocab)

        trainer = SupervisedTrainer(
            optimizer=optimizer,
            criterion=criterion,
            trainset_list=trainset_list,
            validset=validset,
            num_workers=config.num_workers,
            device=device,
            teacher_forcing_step=config.teacher_forcing_step,
            min_teacher_forcing_ratio=config.min_teacher_forcing_ratio,
            print_every=config.print_every,
            save_result_every=config.save_result_every,
            checkpoint_every=config.checkpoint_every,
            architecture=config.architecture,
            vocab=vocab,
            joint_ctc_attention=config.joint_ctc_attention,
        )

        model = trainer.train(
            model=model,
            batch_size=config.batch_size,
            epoch_time_step=epoch_time_step,
            num_epochs=config.num_epochs,
            teacher_forcing_ratio=config.teacher_forcing_ratio,
            # resume=config.resume,
        )

