# Multi-Scale Fusion and Decomposition Network for Single Image Deraining (MFDNet)
This is an implementation of the MFDNet model.

## Installation

The model is built in Python 3.7, PyTorch 1.13.1, CUDA11.6.

For installing, follow these intructions

```
conda env create -f mfd.yml
```

## Training

- Download training dataset ((raw images)[Baidu Cloud](https://pan.baidu.com/s/1usedYAf3gYOgAJJUDlrwWg), (**Password:4qnh**) (.npy)[Baidu Cloud](https://pan.baidu.com/s/1hOmO-xrZ2I6sI4lXiqhStA), (**Password:gd2s**)) and place it in `./Datasets/train/`

- Train the model with default arguments by running

```
python train.py
```


## Evaluation

1. The pretrained_model is provided in `./checkpoints/checkpoints_mfd.pth`

2. Download test datasets (Test100, Rain100H, Rain100L, Test1200, Test2800) from [here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing) and place them in `./Datasets/test/`

3. Run
```
python test.py
```
