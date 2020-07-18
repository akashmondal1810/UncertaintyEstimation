# UncertaintyEst
Uncertainty estimation tool, and effective descision making tool in uncertain dataset segments

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
* run `cd UncertaintyEst

#### Prerequisites
* Module used in this repo are, `tensorflow`, `numpy`, `pandas`, `sklearn`, `keras`, `json`, `multiprocessing` and `matplotlib`. These can be installed using `pip install -r dependencies.txt`
* Download the datasets in the `data` folder OR use your own data

#### Training
* Training arguments
    * simply run `python train_data.py -h` for the help with diffrent traing arguments
    * `-dr` the devlopement(Training) dataset path, in this case `data/dev1.csv`
    * `-tc` target column name, in this case `class_type`
    * `-nh` nodes in each hidden layers, here `500 1000 800` means three hidden layers with nodes 500, 1000 and 800
    * `-e` no of training epochs
    * `-bts` batch size
    * `-dp` the dropout rate
    * `-tu` the tau value
* Train the downloaded data by running `python train_data.py -dr data/dev1.csv -tc class_type -nh 500 1000 800 -e 4 -bts 1024 -dp 0.5 -tu 0.5`

#### Finding the Cutoff value
* Arguments used for finding the uncertainty threshold:
    * simply run `python findCutoff.py -h` for the help with diffrent arguments neede
    * `-vdr` the validation data path, in this case `data/val.csv`
* Find the cutoff value in the downloaded data by running `python findCutoff.py -vdr data/val1.csv`

#### Evaluation
* Arguments used for the evaluation process:
    * simply run `python Evaluation.py -h` for the help with diffrent arguments neede
    * `-edr` the dataset you want to evaluate, in this case `data/uncetainSeg1.csv`
    * `-esd` csv path to save the results
* Evaluate downloaded data by running `python Evaluation.py -edr data/uncetainSeg1.csv -esd results.csv`


## Features and corner cases
* Easy Preprocessing
* Prints basic informations about the dataset
* Missing value imputation strategy, which includes filling using Mean, Mode, Median, zero
* Can detect how many columns are to labelencode by finding all the non numeric columns
* Suppots autometic hyperparameter autotuning
* Can detect the problem type, 'Regression', 'Classification' or 'Multi class'

## Setup and Usage
* run `git clone https://github.com/akashmondal1810/UncertaintyEst.git`
* run `cd UncertaintyEst`

#### For Preprocessing
* First fill up the different preprocessing strategies in the `preprocessing/preprocessing.json`
* run `python process_data.py LOAN_DATA_PATH OUTPUT_PATH` for preprocessing the Dataset and replace the `LOAN_DATA_PATH` with the  dataset path
* On running `python process_data.py data/boston/datasets_Boston.csv data/boston` it will first output the basic information about the dataset in the scrren and after preprocessing it will split and save the data in the given destination as shown below
![Selection_069](https://user-images.githubusercontent.com/28530297/85549816-fc471a80-b63d-11ea-8530-4cd407a52aae.png)

#### For Training
* First fill up the different taining strategies in the `autotuning/training_strategy.json`
* run `python train_data.py DATASET_NAME` for training and aututuning the Dataset and replace the `DATASET_NAME` with the  dataset name like MNIST

#### For Testing
* run `python test_data.py DATASET_NAME SAVED_MODEL_PATH` for Evaluation of the model. Here replace the `SAVED_MODEL_PATH` with the path where the model have been saved after training and autotuning.
* On running it will print out different results as shown
![Selection_071](https://user-images.githubusercontent.com/28530297/85608451-b2c4f280-b672-11ea-8a24-267fdaf6e738.png)

## Code walkthrough
The [preprocessing](https://github.com/akashmondal1810/UncertaintyEst/tree/master/preprocessing), getting the data ready for the machine learning algorithms, deals with missing data, handling categorical data, bringing features on the same scale, and selecting meaningful features. Different classes have been created for the operations, e.g. ‘Information’, ‘Preprocess’, ‘runPreprocess’, etc. The 'Information' class prints summary information about the data set on the screen like no of rows and features, feature name, data type, size of the dataset, no of missing values in each feature, the memory size of the dataset,  ten samples of each feature, etc. In the 'Preprocess' class The preprocessing of the data set is done using this class. Here the missing values can be filled with None, Zero, column Mode, Mean, and Median values. The drop strategies can be defined to drop both the columns(axis=1) and the rows(axis=0) containing missing values. It also includes label encoders, getting dummy values, etc. And the last class 'runPreprocess', this class created to develop different preprocessing strategies, to run the ‘Preprocessor’ class.

The [Training Folder](https://github.com/akashmondal1810/UncertaintyEst/tree/master/training) contains a generators to generate the dataset on multiple cores in real-time which we can feed it right away to our model. It also contains Classes like ‘NNDropout’(Which creates the neural network architecture using the inputs like hidden layer structure, activation function, etc.), ‘MCDropout’(a constructor for the class implementing the MC Dropout neural network model taking the training data, tau value used for regularization, the Dropout rate for all the dropout layers in the network, etc. as input), etc. have been created for this process.

### Default directory structure

```
UncertaintyEst
├── preprocessing
|   ├── preprocessing.json (Stores the strategies required for the preprocessing)
|   ├── preprocessing.py (Script to deals with the preprocessing of data)
|   └── pp_runner.py (Helper script to run the proprocessing functions)
├── Training 
│   ├── generator.py (Generators script to generate the dataset)
│   ├── train_bin.py (Script containg constructor for the class implementing of the network architecture)
│   └── mc_dropout.py (Script for generating the MC Dropout model and making predictions)
.
.
    
```
