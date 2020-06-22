# UncertaintyEst(Work in Progress)
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

### Setup and Uses
* run `python train_data.py MNIST` for training and autotuning in the MNIST Data
