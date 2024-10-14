# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#configs">Configs</a> •
  <a href="#license">License</a>
</p>

## About

This repo contains [Deep Speech 2](https://arxiv.org/abs/1512.02595) model for Automatic Speech Recognition trained on Librispeech dataset. It achieves 0.2 WER metric on test-clean part (no background noise and correct pronunciation) when used with LM during inference. 

This project was completed during HSE FCS Deep Learning course. See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

Toggle "lm_decoding" parameter in text_encoder section to choose whether to use LM model during inference. Notice that LM model is used only on metrics that are calculated with beam_search (toggle use_beam_search parameter in metrics config). 

Inference on test-clean can be run with simple inference.yaml config. To run on test-other choose the suitable dataset in config. 

## Configs

Detailed description of experiments conducted during this project can be seen by this WanDb report link.
1. deepspeech.yaml - train on clean dataset.
2. deepspeech2.yaml - train on clean dataset with augmentations.

## Additional Options

Also possible to use on custom folders with audio data and predictions (look up inference_custom_data.yaml). Please store it the following order:
```
SomeDir
├── audio
│   ├── UtteranceID1.wav # may be flac or mp3
│   ├── UtteranceID2.wav
│   .
│   .
│   .
│   └── UtteranceIDn.wav
└── transcriptions # ground truth, may not exist
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt
```

Run metrics_check_script.py to calculate WER and CER on two folders with predicted and true texts. Check -h option for details.
```
├── predicted_text_folder
│   ├── input_1.txt # should be plain text
│   ├── input_2.txt
│   .
│   .
│   .
│   └── UtteranceIDn.wav
└── transcriptions # ground truth. Names should match with files in transcription folders.
    ├── input_1.txt
    ├── input_1.txt
    .
    .
    .
    └── UtteranceIDn.txt
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
