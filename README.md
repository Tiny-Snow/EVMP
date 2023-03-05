# EVMP

This repository contains the code for the paper *EVMP: Enhancing machine learning models for synthetic promoter strength prediction by Extended Vision Mutant Priority framework*. The paper is available at [https://doi.org/10.1101/2022.10.15.512354](https://doi.org/10.1101/2022.10.15.512354). EVMP is a framework for synthetic promoter strength prediction, including a series of machine learning models, a set of experiments and the benchmark.

# Instructions

## File Structure
* The `Dataset/` folder contains the sample data, and the sample data should be manually moved to the `data/` folder mentioned below.
* The `EVMP/` folder contains the code for all the models. The following `Code Structure` refers to the structure of these folders:
```
Baseline-GBDT
Baseline-LSTM
Baseline-RF
Baseline-SVM
Baseline-Transformer
Baseline-XGBoost
EVMP-GBDT
EVMP-LSTM
EVMP-RF
EVMP-SVM
EVMP-Transformer
EVMP-XGBoost
```
* `Illustration of model/` folder contains the images that show the results of all the models.

## Code Structure
* `data/` This folder contains synthetic promoters like `synthetic_promoter.csv`. 

*New!* This folder is needed to be mannually created, and the sample data can be found from `Dataset/`.
* `model/` The code for EVMP Framework.
* `output/` The output of the experiments, including visualized results.
* `save/` The saved models and logs.

** This folder is needed to be mannually created. **
* `utils/` The utility functions, including training, loading data, plot and learning rate adjustment.
* `config.py` The configuration file. Change the parameters in this file to change the experiment settings.
* `main.py` The main file. Run this file to run the experiments.
## Dependencies
Required dependencies (using `python 3.7` ):

```
pytorch==1.10.1
torchvision==0.11.2
cudatoolkit==11.3.1
joblib==0.17.0
matplotlib==3.5.3
scipy==1.7.3
tqdm==4.64.1
xgboost==1.5.1
scikit-learn==1.0.2
numpy==1.21.5
pandas==1.3.5
```


# Usage
## Data Preparation
ALL the data is in `csv` format.

Prepare the training data at `synthetic_data_path` and  `wild_data_path` in the format of:
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
Make sure that in `config.py`:
```
`if_test` = False
```
in `signal.txt`:
```
{'train': True}
```
You may set `pretrain = True` to use saved model specified at `pretrain_model_path` in `config.py`. Run the following command to train the model:
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
[001/10000] Train MAE Loss: 0.504992 | Val MAE loss: 0.350093
>>>>> Saving model with loss 0.350093
Learning rate -> 1.000000e-04
[002/10000] Train MAE Loss: 0.375110 | Val MAE loss: 0.377652
Learning rate -> 1.000000e-04
[003/10000] Train MAE Loss: 0.364994 | Val MAE loss: 0.345428
>>>>> Saving model with loss 0.345428
Learning rate -> 1.000000e-04
[004/10000] Train MAE Loss: 0.349199 | Val MAE loss: 0.346701
Learning rate -> 1.000000e-04
[005/10000] Train MAE Loss: 0.332358 | Val MAE loss: 0.288683
>>>>> Saving model with loss 0.288683
Learning rate -> 1.000000e-04
[006/10000] Train MAE Loss: 0.327650 | Val MAE loss: 0.289182
Learning rate -> 1.000000e-04
[007/10000] Train MAE Loss: 0.314388 | Val MAE loss: 0.288350
>>>>> Saving model with loss 0.288350
Learning rate -> 1.000000e-04
[008/10000] Train MAE Loss: 0.308514 | Val MAE loss: 0.359426
Learning rate -> 1.000000e-04
[009/10000] Train MAE Loss: 0.317545 | Val MAE loss: 0.273503
>>>>> Saving model with loss 0.273503
Learning rate -> 1.000000e-04
[010/10000] Train MAE Loss: 0.312983 | Val MAE loss: 0.265739
>>>>> Saving model with loss 0.265739
Learning rate -> 1.000000e-05
......
[319/10000] Train MAE Loss: 0.159014 | Val MAE loss: 0.180608
>>>>> Saving model with loss 0.180608
Learning rate -> 1.000000e-05
......
>>>>> End with signal!
>>>>> Training Complete! Start Testing...
Final Model: 
    train MAE: 0.126674, train R2: 0.83
    valid MAE: 0.180608, valid R2: 0.76
    test  MAE: 0.184852, test  R2: 0.77
```
To use the pretrain model, you need to change the `pretrain` parameter.

Note: The training time does not exceed 12 hours when training for no more than 500 epochs on a dataset of thousands of promoters.

## Early Stop
Instead of the Ctrl+C canceling, we use `signal.txt` to control the saving process. During the training process, the program automatically saves the best models according to the Val MAE loss. You may manually set the `train` in `signal.txt` to `false` to make it stop at the end of the current epoch and save the model parameters.


## Testing
Make sure that in `config.py`:
```
`if_test` = False
```
in `signal.txt`:
```
train = False
```
Set `pretrain = True` to use saved model specified at `pretrain_model_path` in `config.py`.
Run the following command to test the model:
```
python main.py
```
The pretrained model will be used to predict the strength.

Also, you may set `if_predict = True` in `config.py` to run prediction. The output will be saved in `predict_result_path`, which can be modified in the `config.py` file.

# Issues

This part is used to collect known issues. 

