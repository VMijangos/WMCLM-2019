# IWMLC 2019

This repository provide a reference for runnning the complexity estimation method proposed for IWMLC 2019. 

This program runs in python 3. The program uses the next libraries:

* Standard pyhton libraries (numpy, collections, itertools, random, re)
* nltk (Natural Language Toolkit) https://www.nltk.org/ 

## Basic Usage

To run the model, execute the following command:<br/>

	``python3 main.py --input corpora --output results/results.csv``

### Input/Output directories
The model requires specify the next directories:

* input : input directory of data. Default is corpora.
* output : output directory. Default is results/results.csv

### Parameters

The model uses the next parameters:

##### For the n-phone extractor:

* n : the size of n-phones. Default is 3

##### For the neural probabilistic language model:

* iter : number of iterations to train the neural network. Default is 50
* subsample_siz : Number of examples for epoch in SGD. Default is 100
* emb_dim : Number of dimensions in embedding vectors. Default is 300
* hid_dim : Number of dimensions in hidden layer. Default is 100

To run the model with different parameters, execute the program like the next example:<br/>

	``python3 main.py --input corpora --output results/results.csv --n 1 --iter 100``
