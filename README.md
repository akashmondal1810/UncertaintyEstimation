# UncertaintyEst
Uncertainty estimation and effective descision making tool in uncertain dataset segments

## Features and corner cases
* Easy Preprocessing
* Prints basic informations about the dataset
* Missing value imputation strategy, which includes filling using Mean, Mode, Median, zero
* Can detect how many columns are to labelencode by finding all the non numeric columns
* Suppots autometic hyperparameter autotuning
* Can detect the problem type, 'Regression', 'Classification' or 'Multi class'

## Getting Started
Copy the repo to your local machine by
* running `git clone https://github.com/akashmondal1810/UncertaintyEst.git`
* run `cd UncertaintyEst`

#### Prerequisites
* Module used in this repo are, `tensorflow`, `numpy`, `pandas`, `sklearn`, `keras`, `json`, `multiprocessing` and `matplotlib`. These can be installed using `pip install -r dependencies.txt`

#### For Preprocessing
* First fill up the different preprocessing strategies in the `preprocessing/preprocessing.json`
* run `python process_data.py --dataPath DATA_PATH --dataSaveDir OUTPUT_PATH` for preprocessing the Dataset and replace the `DATA_PATH` with the  dataset path

#### Training
* Training arguments
    * simply run `python train_data.py -h` for the help with diffrent traing arguments
    * `--algo` the model you want to train, `MCD` for MC Dropout, `DeepEnsmb` for Deep Ensemble, `MultiXGB` for multiple XGBoost and `RandomXGB` for random XGBoost
    * `-dr` the devlopement(Training) dataset path
    * `-tc` target column name
    * It will train using the default value, you can train using other parameters mentioned in the training_strategy files inside training folder

#### Evaluation
* Arguments used for the evaluation process:
    * simply run `python Evaluation.py -h` for the help with diffrent arguments needed
    * `--algo` the model you want to evaluate, `MCD` for MC Dropout, `DeepEnsmb` for Deep Ensemble, `RandomXGB` for random XGBoost and `RandomXGB` for random XGBoost
    * `-edr` the dataset you want to evaluate
    * `-esd` csv path to save the results
    
#### Prediction
* Arguments used for prediction:
    * simply run `python Predict.py -h` for the help with diffrent arguments needed
    * `--algo` the model you want to evaluate, `MCD` for MC Dropout, `DeepEnsmb` for Deep Ensemble, `RandomXGB` for random XGBoost and `RandomXGB` for random XGBoost
    * `-pdr` Dataset path for predictions
    * `-psd` csv path to save the results

## Results
We tested our system in two Uncertain segments, the AUC Score in listed below. Here the probabilistic models scores are by suppressing the 'uncertain' points. Hence the uncertainty estimation is very useful to avoid misclassification, relaxing our neural network to make a prediction when there’s not so much confidence.

Models | XGBoost | MC Dropout | Deep Ensemble | Multi XGB | Random XGB
--- | --- | --- | --- | --- | --- 
Uncertain Seg1(From 20Q1) | 60.2 | 68.4 | 77.2 | 81.8 | 72.3
Uncertain Seg1(Low FICO<500) | 81.3 | 83.8 | 91.9 | 86.5 | 79.3
