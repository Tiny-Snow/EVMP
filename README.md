# EVMP

This repository contains the code for the paper *EVMP: Enhancing machine learning models for synthetic promoter strength prediction by Extended-Vision-Mutant-Priority framework*. The paper is available at [https://doi.org/10.1101/2022.10.15.512354](https://doi.org/10.1101/2022.10.15.512354). EVMP is a framework for synthetic promoter strength prediction, including a series of machine learning models, a set of experiments and the benchmark.
# Instructions
## Code Structure
* `data/` This folder contains `synthetic_promoter.csv`, which contains the data for the problem. 
* `model/` The code for EVMP Framework.
* `output/` The output of the experiments, including visualized results.
* `save/` The saved models and logs.
* `utils/` The utility functions, including training, loading data, plot and learning rate adjustment.
* `config.py` The configuration file. Change the parameters in this file to change the experiment settings.
* `main.py` The main file. Run this file to run the experiments.
* `plot.py` The plotting functions.
## Dependencies
Required dependencies:
* `torch` == 1.10.2+cu113
* `numpy` == 1.19.0
* `matplotlib` == 3.5.2


# Usage
## Data Preparation
Prepare the training data at `synthetic_data_path ` and  `wild_data_path` in the format of:
```
ID, Mother Promoter, Promoter, ACT
```
`ID` is the ID of the promoter. `Mother Promoter` is the base promoter. `Promoter` is the synthetic promoter.  `ACT` is the strength of synthetic promoter.

Prepare the testing data at `predict_data_path` in the format of:
```
ID, Mother Promoter, Promoter
```
As the testing data is only used to predict the strength, the `ACT` column is not needed. 
## Training
Run the following command to train the model:
```
python main.py
```
In order to run the experiments, you need to change the parameters in the `config.py` file. The output log is like the following:

```
>>>>> Loading data ...
>>>>> Size of training set: 2832
>>>>> Size of validation set: 364
>>>>> Size of test set: 351
>>>>> Not use predict set
>>>>> DEVICE: cuda
>>>>> Start  Training...
[001/10000] Train MAE Loss: 0.799914 | Val MAE loss: 0.496356
>>>>> Saving model with loss 0.496356
Learning rate -> 1.000000e-04
[002/10000] Train MAE Loss: 0.446956 | Val MAE loss: 0.352508
>>>>> Saving model with loss 0.352508
Learning rate -> 1.000000e-04
[003/10000] Train MAE Loss: 0.352595 | Val MAE loss: 0.419848
Learning rate -> 1.000000e-04
[004/10000] Train MAE Loss: 0.320540 | Val MAE loss: 0.306786
>>>>> Saving model with loss 0.306786
Learning rate -> 1.000000e-04
[005/10000] Train MAE Loss: 0.299916 | Val MAE loss: 0.318937
Learning rate -> 1.000000e-04
[006/10000] Train MAE Loss: 0.281108 | Val MAE loss: 0.261879
>>>>> Saving model with loss 0.261879
Learning rate -> 1.000000e-04
[007/10000] Train MAE Loss: 0.265449 | Val MAE loss: 0.351830
Learning rate -> 1.000000e-04
[008/10000] Train MAE Loss: 0.259729 | Val MAE loss: 0.266807
Learning rate -> 1.000000e-04
[009/10000] Train MAE Loss: 0.246711 | Val MAE loss: 0.286405
Learning rate -> 1.000000e-04
[010/10000] Train MAE Loss: 0.248280 | Val MAE loss: 0.261508
>>>>> Saving model with loss 0.261508
Learning rate -> 1.000000e-04
[011/10000] Train MAE Loss: 0.239651 | Val MAE loss: 0.239710
>>>>> Saving model with loss 0.239710
Learning rate -> 1.000000e-04
[012/10000] Train MAE Loss: 0.231826 | Val MAE loss: 0.278435
Learning rate -> 1.000000e-04
[013/10000] Train MAE Loss: 0.225443 | Val MAE loss: 0.227194
>>>>> Saving model with loss 0.227194
......
>>>>> End with signal!
>>>>> Training Complete! Start Testing...
Final Model: 
    train MAE: 0.160957, train R2: 0.78
    valid MAE: 0.202348, valid R2: 0.73
    test  MAE: 0.183557, test  R2: 0.79
```
To use the pretrain model, you need to change the `pretrain` parameter.

### Save model immediately
Instead of the Ctrl+C canceling, `signal.txt` is used to control the saving process. During the training process, a model is saved while its Val MAE loss is the lowest. Once you set the `train` in `signal.txt` to `false`, the model will be saved and the training process will be terminated after the epoch ends.


## Testing
Set the `if_test` and `pretrain` parameter to true in the `config.py` file, then run the following command to test the model:
```
python main.py
```
The pretrained model will be used to predict the strength, and the output will be saved in `predict_result_path`, which can be modified in the `config.py` file.

