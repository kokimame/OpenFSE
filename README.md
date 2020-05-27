# Status: Work in Progress
This is the repository which contains the code of our audio embedding system.
The development of this work will be finished by September with the release
of pre-trained models. For a detailed explanation of the work, please refer to
our upcoming paper.

### Goal
- Learn an audio representation that can serve as a base for general machine listening systems, including unsupervised learning (clustering)
- Prototype application toward Search Result Clustering on [freesound.org](https://freesound.org)

### Key questions
- Is the triplet loss approach appropriate to obtain an audio representation for general purpose?
- Is metric learning approach a good choice for clustering?
Assuming yes since clustering often relies on a similarity measure
and metric-based loss captures the similarity among data more precisely than class-based loss (e.g. cross-entropy)

### About implementation
- Workflow/basic framework is based on [MOVE](https://github.com/furkanyesiler/move/)
- VGG model: Input spectrogram (128x128) -> Embedding 256
- Online Triplet Mining

### How to run
```
cd src
python3 model_main.py --dataset_name {your dataset name}
python3 tb.py # Visualize the training on Tensorboard
```
