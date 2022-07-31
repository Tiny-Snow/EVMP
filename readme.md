# EVMP Transformer
This repository contains the code for the paper *EVMP-Transformer: Deep Learning Enhancement Framework for
Synthetic Promoter Representation*. The paper is available at [https://arxiv.org/abs/xxxxxxxx](https://arxiv.org/abs/xxxxxxxx). EVMP-Transformer is a framework of promoter prediction, including a series of machine learning models, a set of experiments and the benchmark.
# Instructions
## Code Structure
* `data/` This folder contains `synthetic_promoter.csv`, which contains the data for the problem. 
* `model/` The code for EVMP Promoter Encoder Framework and Promoter Transformer.
* `output/` The output of the experiments.
* `save/` The saved models, logs, figures, and supporting materials.
* `utils/` The utility functions including dataloader and cycleLR.
* `config.py` The configuration file. Change the parameters in this file to change the experiment settings.
* `main.py` The main file. Run this file to run the experiments.
* `plot.py` The plotting functions.
## Installation
```
pip install -r requirements.txt
```
Required dependencies:
* `torch` == 1.10.2
* `torchvision` == 0.8.2+cu110
* `numpy` == 1.18.5
* `matplotlib` == 3.4.2


# Usage
## Data Preparation
`Prepare the training data in the format of:`
```
ID, Mother Promoter, Promoter, ACT
```
ID is the ID of the promoter. Mother Promoter is the base promoter of the mutated promoter. Promoter is the content of the promoter. ACT is the activity value.

`Prepare the testing data in the format of:`
```
ID, Mother Promoter, Promoter
```
As the testing data is only used to predict the activity value, the ACT column is not needed. There is a sample data in the `data/` folder.
## Training
Run the following command to train the model:
```
python main.py

```
In order to run the experiments, you need to change the parameters in the `config.py` file. The output log is like the following:

```
>>>>> Loading data ...
>>>>> Size of training data: 3548
>>>>> Size of training set: 3184
>>>>> Size of validation set: 364
>>>>> Not use test set
>>>>> Not use predict set
>>>>> DEVICE: cuda
>>>>> Start  Training...
[001/10000] Train MAE Loss: 0.460293 | Val MAE loss: 0.501961
>>>>> Saving model with loss 0.501961
Learning rate -> 1.000000e-04
[002/10000] Train MAE Loss: 0.356633 | Val MAE loss: 0.512425
Learning rate -> 1.000000e-04
[003/10000] Train MAE Loss: 0.324723 | Val MAE loss: 0.274957
>>>>> Saving model with loss 0.274957
Learning rate -> 1.000000e-04
[004/10000] Train MAE Loss: 0.313141 | Val MAE loss: 0.320510
Learning rate -> 1.000000e-04
[005/10000] Train MAE Loss: 0.303527 | Val MAE loss: 0.258869
>>>>> Saving model with loss 0.258869
Learning rate -> 1.000000e-04
[006/10000] Train MAE Loss: 0.293457 | Val MAE loss: 0.357512
Learning rate -> 1.000000e-04
[007/10000] Train MAE Loss: 0.289899 | Val MAE loss: 0.287305
Learning rate -> 1.000000e-04
[008/10000] Train MAE Loss: 0.284617 | Val MAE loss: 0.395426
Learning rate -> 1.000000e-04
[009/10000] Train MAE Loss: 0.275195 | Val MAE loss: 0.310768
Learning rate -> 1.000000e-04
[010/10000] Train MAE Loss: 0.270260 | Val MAE loss: 0.229189
>>>>> Saving model with loss 0.229189
```
To use the pretrain model, you need to change the `pretrain` parameter.

### Save model immediately
Instead of the Ctrl+C canceling, `signal.txt` is used to control the saving process. During the training process, a model is saved while its Val MAE loss is the lowest. Once you set the `train` in `signal.txt` to `false`, the model will be saved and the training process will be terminated after the epoch ends.


## Testing
Set the `if_test` and `pretrain` parameter to true in the `config.py` file, then run the following command to test the model:
```
python main.py
```
The pretrained model will be used to predict the activity value, and the output will be saved in `test_result_path`, which can be modified in the `config.py` file. Also, the visualization of the predicted activity value will be saved in `save_fig`.
