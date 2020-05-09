# Status: Work in Progress
This is the repository which contains the code of our audio embedding system.
The development of this work will be finished by September with the release
of pretrained models. For a detailed explanation of the work, please refer to
our upcoming paper.

### Learning sound embedding based using MOVE framework
- General machine learning framework is based on [MOVE](https://github.com/furkanyesiler/move/)
- VGG model: Input spectrogram (128x128) -> Embedding 256
- Online Triplet Mining

### How to run
```
cd src
python3 model_main.py --dataset_name tag_dense
```

### Visualize the training on Tensorboard
```
python3 tb.py
```