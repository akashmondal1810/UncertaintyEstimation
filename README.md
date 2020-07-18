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
* run `cd UncertaintyEst`
* Download the data by clicking in the [link](https://docs.google.com/u/0/uc?export=download&confirm=KW-f&id=16lUGsUKRU2EJNKYpo6Qrvjzm7g2-UrfZ) or by running 
`curl -O https://docs.google.com/u/0/uc?export=download&confirm=KW-f&id=16lUGsUKRU2EJNKYpo6Qrvjzm7g2-UrfZ` and extract it in the `data` folder

#### Prerequisites
* Module used in this repo are, `tensorflow`, `numpy`, `pandas`, `sklearn`, `keras`, `json`, `multiprocessing` and `matplotlib`. These can be installed using `pip install -r dependencies.txt`
* Download the datasets in the `data` folder OR use your own data

#### Training
* Training arguments
    * simply run `python train_data.py -h` for the help with diffrent traing arguments
    * `-dr` the devlopement(Training) dataset path, in this case `data/dev.csv`
    * `-tc` target column name, in this case `class_type`
    * `-nh` nodes in each hidden layers, here `500 1000 800` means three hidden layers with nodes 500, 1000 and 800
    * `-e` no of training epochs
    * `-bts` batch size
    * `-dp` the dropout rate
    * `-tu` the tau value
* Train the downloaded data by running `python train_data.py -dr data/dev.csv -tc class_type -nh 500 1000 800 -e 4 -bts 1024 -dp 0.5 -tu 0.5`

#### Finding the Cutoff value
* Arguments used for finding the uncertainty threshold:
    * simply run `python findCutoff.py -h` for the help with diffrent arguments neede
    * `-vdr` the validation data path, in this case `data/va.csv`
* Find the cutoff value in the downloaded data by running `python findCutoff.py -vdr data/val.csv`

#### Evaluation
* Arguments used for the evaluation process:
    * simply run `python Evaluation.py -h` for the help with diffrent arguments neede
    * `-edr` the dataset you want to evaluate, in this case `data/uncetainSeg1.csv`
    * `-esd` csv path to save the results
* Evaluate downloaded data by running `python Evaluation.py -edr data/uncetainSeg1.csv -esd results.csv`
