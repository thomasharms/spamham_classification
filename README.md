# Spam Classifiers

This tool is providing functionality of classification of spam out of sets of mails. The mails are imported directly in mail format and processed by sanitizing html, removing stop words, lemmatizing and transforming the message into lower case characters. The data is being classified by several classifiers in two different ways. On one hand spam classification is being done by using sklearn libraries directly and classifying by transforming text with the TF-IDF algorithm. On the other hand the classification with the same classifiers is repeated by using the NLTK - SklearnClassifier import functions. The data is being transformed in a different for that purpose.

## Getting Started

The process of set up requires a couple of prerequsites. The tool is being run and tested using Python v3.7. NLTK needs to be installed as well as the usual machine learning libraries. Details can be found by following the instructions at the Requirements section and Install section as well.

### Requirements

First of a properly installed Python v3.6+ is necessary. The content of the scripts as well as the data in this repository is mandatory as well. Furthermore the PyData libraries Pandas, sklearn, Numpy, NLTK are required. The process of geting these up and running is described in the Install section.

### Installing

The files are available at [the projects git repository](https://github.com/thomasharms/1a1testtool.git). Feel free to fork or clone ahead. Provided you have a Python Environment or at least an installation running, copy the files in a folder of your choice.
In case you need to install the PyData libraries, you can do so by using pip3 and a shell of your choice and execute:
```
sudo pip3 install numpy, Pandas, sklearn, scipy
```
Installing NLTK requires a couple of additional steps, such as described [here](https://www.nltk.org/install.html):
```
sudo pip3 install nltk
```
Depending on your installation, install any additional packages and dependencies you might encounter by running the scripts. None are known so far.

## Running Classifiers

There are two completely different ways of solving the classification problem: using machine learning algorithms by utilizing sparse matrices out of TF-IDF Vectorizers and using NLP approach by importing machine learning classifiers in a different fashion described in the following section.

### ML Classifiers

The script to run is spam_classification_ml.py . In order to do that you can run the script in two ways.
* First off you could open a shell and navigate towards the location of the folder, you put the data from the pulled/cloned git repository. Afterwards you excecute in a shell:
```
sudo python3 ./spam_classification_ml.py
```
* Alternatively you can run the script by executing:
```
sudo python3 /location/of/your/git/folder/spam_classification_ml.py
```
The classifiers will present an accuracy score, precision, recall and F1 Score value to you.

### NLP Classification

The NLP approach has been implemented within the spam_classification_nltk.py file. You can run it in a similar way as described in the previous section. The classifiers will present an accuracy score.
The algorithm is done as follows:
* build a list of tuples per message in the following fashion [(list_of_all_words, label), (,),...]
    the list of words has been pre_processed by removing stop words, lemmatized, lowered
* build a set of a list of all_words in all documents (set is removing duplicates)
* build a Frequency Distribution of the set of all words
* choose the n most common words as features (skiping the first couple of hundreds since they do not provide increase in entropy)
* the feature_set is built by building a list of tuples, containing (hashmap of features, label)
* the hashmap of features is looking for each word feature, if it is present in each document (word feature, True/False)
* training set and test set are build by spliting the feature_set
* each Classifier is import using nltk.SklearnClassifier as a wrapper for the machine learning classifiers

## Evaluation

The scores for accuracy, precision, recall and F1 score are very reasonable and constant between 93% and 98%. AdaBoost and RandomForest -classifiers are usually a bit better since they are ensembling multiple classifiers and a voting system as well as weighted boosting of better performing classifiers respectively. However, the problem of overfitting is being countered by AdaBoost, the SVM classifiers and RandomForest as well. AdaBoost and RandomForest are Bootstraping of the data before training and SVM are bound towards the amount of support vectors within their treshold. Furthermore there is a shuffle process of the training and test sets being implemented. While the outcome is constant and the score does not vary, the underfitting and overfitting concerns are not reasonable, hence the outcome can be trusted.
In case of the ML approach, the TF-IDF Vectorizer counters the bias coming along by longer or shorter messages and counters the repitition of several words within longer texts. TF-IDF boosts words of average frequency distributed within the corpus as well as the split performed in the NLP approach by cuting out the first 300 most common distributed words and slicing the feature set accordingly.

## Versioning

For the versions available, see the [tags on this repository](https://github.com/thomasharms/1a1testtool/tags). 

## Authors

* **Thomas Harms** - *Initial work* - [Thomas Harms](https://github.com/thomasharms)
