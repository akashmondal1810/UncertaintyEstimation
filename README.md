# UncertaintyEstimation
Deterministic point prediction models like Deep neural networks, Gradient Boosting algorithm have achieved state-of-the-art performance on a wide variety of machine learning tasks, and are becoming increasingly popular in domains such as computer vision, speech recognition, natural language processing, and in Credit Risk Analysis related decisions. Despite impressive accuracies in supervised learning benchmarks, these are poor at quantifying predictive
uncertainty, and tend to produce overconfident predictions. Overconfident incorrect predictions can be harmful, hence proper uncertainty quantification is crucial for practical applications. But, while these deterministic models often achieve high accuracy, their predictions have been shown to be miscalibrated and shown to yield arbitrarily high confidence far away from the training data. This means these models are susceptible to out-of-distribution (OOD or Uncertain)examples. So, when these classifiers are employed in real-world tasks, they tend to fail when the training and test distributions differ. Worse, these classifiers often fail silently by providing high confidence predictions while being woefully incorrect. These high confidence predictions are frequently produced by softmax
because softmax probabilities are computed with the fast-growing exponential function. Thus minor additions to the softmax inputs, i.e. the logits, can lead to substantial changes in the output distribution.  Meanwhile, probabilistic methods (in particular Bayesian methods) have long been known to improve predictive uncertainty estimates and have been empirically demonstrated to capture these falsely overconfident points by using the predictive uncertainty. Some orks like MacKay et. al. demonstrated experimentally that the predictive uncertainty of Bayesian neural networks will naturally be high in regions not covered by training data. Although the theoretical analysis of such Bayesian approximations are limited and currently a reviving field, we attempted to show that uncertainty is
not only something that places obstacles in front of good predictions, we can indeed benefit from it, if we are able to estimate it.
In this context, our contributions are two-fold. The first is uncertainty quantification: we calculate the predictive
uncertainty involved while estimating the model prediction; The second is to utilize the predictive uncertainty: to use this predictive uncertainty value in
decision making when the test example is from a different distribution (Uncertain Data Segment) from the training data;
We also introduced two novel methods to mine the model uncertainty using Gradient Boosting methods. To experimentally validate our methods, we evaluate it on the Lending Club loan dataset. More about the methodology can be found [here](https://drive.google.com/file/d/1jsuLeRvI9GkT-hxrV6cGqGgGVLuY6Bf1/view?usp=sharing).

## Features
* Predictive uncertainty estimation using Monte Carlo Dropout(`MCD`)
* Predictive uncertainty estimation using Deep Ensemble(`DeepEnsmb`)
* Predictive uncertainty estimation using multiple XGBoost(Novel Method, `MultiXGB`)
* Predictive uncertainty estimation using Random XGBoost(Novel Method, `RandomXGB`)
* We also utilized the predictive uncertainty in the decision making process

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

## Dataset
To experimentally validate our methods, we have used the [Lending Club loan](https://www.lendingclub.com/statistics/additional-statistics) dataset. Lending Club is a peer-to-peer lending company, the largest of its kind in the world. It is an online lending platform where borrowers are able to obtain loans and investors
can purchase notes backed by payments based on loans. The dataset contains complete loan data for all loans issued by lendingclub.com from 2007-2020, including the current loan status (Current, Charged Off, Fully Paid, etc.) and latest payment information.
Also we define a profit column in the existing data, containing the profit rate of each loan by taking the percentage difference between total payment (payments received to date for total amount funded) and the loan amount(total amount funded). As we will be exploring both the regression and classification, so we define
a target variable for the classification setup class type which is the positive/negative profit in each class, i.e. if the profit rate is positive, means the overall loan is a good loan and if the profit rate is negative, means the overall loan is a bad loan from the conservative investor’s standpoint. We will use this
field as the target label in classification setting and the profit rate for regression setting.

For our segment of intreset, the out of distribution or the uncertain segment, we have the data from 2020 quarter1 which include new types of loans like issued which is not present in the development data and also the shift in the distribution due to the global pandemic, COVID-19 situtation. The other uncertain segment is the loans with low FICO(FICO<500) values are considered. As the FICO score are below 500(which are in the range of below 579), hence these scores fall in the poor Score Ranges, and bad credit means higher risk hence higher uncertainty.

## Results
### Regression

#### Standard Datasets

We have used the identical setup as used by Gal and Ghahramani to evaluate MC-dropout. Each dataset is split into 20 train-test folds,
except for the protein dataset which uses 5 folds. A summary of the results is reported below (higher test log likelihood (LL) is better), using some standard dataset.
To Re-generate the results use the scripts from the `Experiment` directory.

Dataset | Deep Ensemble | MultiXGB | RandomXGB | MC Dropout | Catboost
--- | :---: | :---: | :---: | :---: | :---:
Boston Housing      | -2.41 ± 0.25 | -2.68 ± 0.12 | -2.90 ± 0.32 | -2.40 ± 0.04 | -2.67 ± 0.21
Concrete Strength   | -3.06 ± 0.18 | -3.33 ± 0.19 | -3.61 ± 0.50 | -2.93 ± 0.02 | -3.12 ± 0.11
Energy Efficiency   | -1.38 ± 0.22 | -1.60 ± 0.77 | -2.47 ± 0.28 | -1.21 ± 0.01 | -0.63 ± 0.18
Kin8nm              | 1.20 ± 0.02 | 0.41 ± 0.02 | 0.24 ± 0.11 | 1.14 ± 0.01 | 0.35 ± 0.02
Naval Propulsion    | 5.63 ± 0.05 | 3.18 ± 0.28 | 0.76 ± 0.01 | 4.45 ± 0.00 | 4.54 ± 0.03
Power Plant         | -2.79 ± 0.04 | -3.2 ± 0.12 | -5.4 ± 0.01 | -2.80 ± 0.01 | -2.82 ± 0.04
Protein Structure   | -2.83 ± 0.02 | -3.66 ± 0.09 | -5.89 ± 0.38 | -2.87 ± 0.00 | -2.96 ± 0.013
Wine Quality Red    | -0.94 ± 0.12 | -0.97 ± 0.05 | -1.31 ± 0.05 | -0.93 ± 0.01 | -0.99 ± 0.098
Yacht Hydrodynamics | -1.18 ± 0.21 | -3.1 ± 0.7 | -2.83 ± 1.87 | -1.25 ± 0.01 | -0.55 ± 0.24

Time Consumption in second(Train time/Test time):

Dataset | Deep Ensemble | MultiXGB | RandomXGB | MC Dropout | Catboost
--- | :---: | :---: | :---: | :---: | :---:
Boston Housing      | - | 3.4938 ± 0.5507/0.1448 ± 0.0501 | - | - | 0.3346 ± 0.0658/0.0032 ± 0.0006
Concrete Strength   | - | 2.3788 ± 0.3227/0.1158 ± 0.0318 | - | - | 0.3765 ± 0.1302/0.0042 ± 0.0041
Energy Efficiency   | - | 3.0059 ± 0.9319/0.1449 ± 0.04210 | - | - | 0.277 ± 0.0252/0.0025 ± 0.0010
Kin8nm              | - | 23.1881 ± 1.5528/0.2369 ± 0.0299 | - | - | 0.4678 ± 0.0697/0.01356 ± 0.0024
Naval Propulsion    | - | 33.3201 ± 2.6609/0.2247 ± 0.0270 | - | - | 0.5532 ± 0.0589/0.0218 ± 0.0027
Power Plant         | - | 21.5622 ± 4.0954/0.2290 ± 0.0687 | - | - | 0.4550 ± 0.0262/0.0100 ± 0.0024
Protein Structure   | - | 41.9296 ± 1.5878/0.2977 ± 0.01126 | - | - | 1.0143 ± 0.0357/0.0361 ± 0.0024
Wine Quality Red    | - | 6.7160 ± 1.1953/0.1356 ± 0.01548 | - | - | 0.3936 ± 0.137/0.0054 ± 0.0025
Yacht Hydrodynamics | - | 1.8915 ± 0.5307/0.1613 ± 0.0435 | - | - | 0.2852 ± 0.0371/0.003 ± 0.001

We observe that our method is very close (or is competitive with) existing methods in terms of LL.
On some datasets, we observe that our method is slightly worse in terms of LL. We believe that
this is due to the fact that our method is not optimizes for LL (which captures predictive uncertainty)
instead of other optimization function like MSE, MAE etc.

### Classification

#### Standard Datasets

AUC score in some standard datasets given below:
Dataset | NN | Laplace BNN | XGBoost 
--- | :---: | :---: | :---: 
Breast Cancer      | 0.992 | 0.991 | 0.993
Sonare   | 0.899 | 0.928 | 0.931

Time Consumption in second(Train time/ Test time):
Dataset | NN | Laplace BNN | XGBoost 
--- | :---: | :---: | :---: 
Breast Cancer      | 9.0821/0.0002 | 9.2532/0.0234 | 0.0350/0.0056
Sonare   | 8.8087/0.0002 | 8.8175/0.0352 | 0.0295/0.0090


#### The Loan Datasets
We tested our system in two Uncertain segments, segment#1(data from 2020Q1) and segment#2(data with Low FICO<500). The resulting AUC Score is listed below.

Models | XGBoost | MC Dropout | Deep Ensemble | Multi XGB | Random XGB
--- | --- | --- | --- | --- | --- 
segment#1 | - | - | - | - | 59.75
segment#2 | - | - | - | - | 89.07

## Future Works
* Making the tool scaleable
* To utilize the predictive uncertainty in other better way while decision making

## References
* Y. Gal and Z. Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In ICML, 2016
* J. M. Hernandez-Lobato and R. Adams, “Probabilistic backpropagation for scalable learning of bayesian neural networks,” in International Conference on Machine Learning, 2015, pp. 1861–1869
* https://www.lendingclub.com/statistics/additional-statistics
* https://xgboost.readthedocs.io/en/latest/python/index.html
* Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell ”Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles”
* Dan Hendrycks, Kevin Gimpel, ”A Baeline for detecting misclassified and out-of-distribution examples in neural network”
* Kendall, A. and Gal, Y. (2017). “What uncertainties do we need in Bayesian deep learning for computer vision?”
* https://humboldt-wi.github.io/blog/research/information_systems_1819/uncertainty-and-credit-scoring/
