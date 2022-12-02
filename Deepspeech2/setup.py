# nsml: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
from distutils.core import setup
import pkg_resources

pkg_resources.require(['pip >= 19.3.1'])

setup(
    name='kospeech_nsml',
    version='latest',
    url='https://github.com/sooftware/KoSpeech',
    install_requires=[
        # 'levenshtein',
        # 'pip >= 19',
        'librosa >= 0.7.0',
        'numpy',
        'pandas',
        'tqdm',
        'matplotlib',
        'astropy',
        'sentencepiece',
        'torchaudio==0.12.1',
        'pydub',
        'glob2',
        'pybind11'
        # 'warp-rnnt'
    ]
    # python_requires='>=3'
)

