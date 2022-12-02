#nsml: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
from distutils.core import setup

setup(
    name='kospeech_nsml',
    version='latest',
    install_requires=[
        # 'torch==1.7.0',
        # 'levenshtein',
        'librosa >= 0.7.0',
        'numpy',
        'pandas',
        'tqdm',
        'matplotlib',
        'astropy',
        'sentencepiece',
        'torchaudio==0.12.1',
        'pydub',
        'glob2'
    ],
)

