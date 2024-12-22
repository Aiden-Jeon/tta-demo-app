# tta-demo-app

## How to start?
### Install dependencies

First, install poetry:

```bash
pip install poetry
```

Next, install python packages with poetry:

```bash
poetry install
```

### Prepare dataset

Download related data, here we use `wget` package, please install `wget` before running below code:

```bash
make prepare-cifar10
```

### Download fine-tuned model
If not downloaded you can use only pre-trained model that is supported from `torchvision` and cannot use fine-tuned model_checkpoint.

Get model from [NOTE Github](https://github.com/TaesikGong/NOTE):  
- [CIFAR10-C](https://drive.google.com/file/d/1YsyHY3rFCaWWDTOVh-RuI1I2bJ5i9Yey/view?usp=sharing)


### Run App

```bash
make run-app
```
