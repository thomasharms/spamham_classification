import email, os, sys, nltk, re

import pandas as pd
import numpy as np

from collections import defaultdict

from nltk.corpus import wordnet as wn
from nltk.classify.scikitlearn import SklearnClassifier
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier




class Import_Data():

    '''
    _spam_files
    _ham_files
    _spamnham_files

    _cwd

    # content focused including headers and preserving information
    _ham_word_list
    _spam_word_list

    # text focused only representing body, not working on HTML mail though
    _spam_body_word_list
    _ham_body_word_list

    # matrix of message, label columns mixed spam and ham
    _spam_ham_matrix

    _error
    # 0 everything is okay
    # 1 no file to read, doesnt exist
    # 2 spam folder doesnt exist
    # 3 ham folder doesnt exist
    # 4 spamnhamfolder doesnt exist
    # 7 content is not readable
    # 9 generic error

    self.error_messages = {}
    '''

    __data_set_folders =    {'spam': "/Data_Sets/spam/",
                            'ham': "/Data_Sets/ham/",
                            'hamnspam': "/Data_Sets/hamnspam/"
                            }

    def __init__(self):

        # sets the path of the working directory
        self.set_cwd()
        
        # sets the list of file_names of each category
        # _spam_files, _ham_files, _spamnham_files
        self.set_file_list()

        self.set_errors()

        self.build_ham_word_list()
        self.build_spam_word_list()

        self.import_as_data_frame()

    def set_errors(self):
        self._error = dict()
        self._error_messages = dict()
        
    def set_cwd(self):
        self._cwd = os.path.dirname(sys.argv[0])

    def set_file_list(self):
        self._spam_files = [self._cwd+self.__data_set_folders['spam']+file_name for file_name in os.listdir(self._cwd+self.__data_set_folders['spam'])]
        self._ham_files = [self._cwd+self.__data_set_folders['ham']+file_name for file_name in os.listdir(self._cwd+self.__data_set_folders['ham'])]
        self._spamnham_files = [self._cwd+self.__data_set_folders['hamnspam']+file_name for file_name in os.listdir(self._cwd+self.__data_set_folders['hamnspam'])]

    def get_fp(self, file_path):
        try:
            fp = open(file_path, 'r')
        
        except IOError as i:
            self._error[file_path] = 1
            self._error_messages[file_path] = str(i)

        except Exception as e:
            self._error[file_path] = 9
            self._error_messages[file_path] = str(e)

        return fp

    def get_ham_words_list(self):
        return self._ham_word_list

    def get_spam_words_list(self):
        return self._spam_word_list

    def get_spam_ham_matrix(self):
        return self._spam_ham_matrix

    def get_msg_from_file_pointer(self, fp, file_path):
        msg = ""
        try:
            msg = email.message_from_file(fp)
            
        except IOError as i:
            self._error[file_path] = 1
            self._error_messages[file_path] = i

        except Exception as e:
            self._error[file_path] = 9
            self._error_messages[file_path] = str(e)

        '''
        if isinstance(msg, email.message.Message): return msg
        else:
            return False
        '''
        return msg

    def read_mail_body(self, msg):

        body = ""
        if msg:
            
            # check if msg is subpart of msgs
            if msg.is_multipart():
                for part in msg.walk():
                    # check content type
                    ctype = part.get_content_type()
                    # check disposition
                    cdispo = str(part.get('Content-Disposition'))

                    # skip text/plain attachments
                    if ctype == 'text/plain' and 'attachment' not in cdispo:

                        # get body out of part for each subpart
                        # decode might not work so good/buggy, will be repeated
                        body = part.get_payload(decode=True)
                        break
            
            # not multipart - i.e. plain text, no attachments
            # decode might not work so good/buggy, will be repeated
            else:
                body = msg.get_payload(decode=True)
        return body

    # !!! main function to use to extract mail content !!!
    # content will everything within the msg
    def get_email_msg_content(self, file_path):

        fp = self.get_fp(file_path)
        
        msg = self.get_msg_from_file_pointer(fp, file_path)
        
        content = ""
        if msg:
            content = self.decode_content(msg)
        return content

    # !!! main function to use to extract mail BODY !!!
    # content will not just be the body of the mail
    # the subject, from, to will be extracted as well in a decoded fashion 
    # if mail not in HTML, e.g. plain text
    def get_email_msg_body(self, file_path):

        fp = self.get_fp(file_path)
        
        msg = self.get_msg_from_file_pointer(fp, file_path)
        
        body = self.read_mail_body(msg)
        
        return self.decode_content(body)

    def decode_content(self, content):
        if type(content) is bytes: return content.decode()

        # not elegant, but the easiest way to get rid of format symbols \r,\n etc.
        elif isinstance(content, email.message.Message): 
            trans_con = str(content)
            trans_con = trans_con.encode()
            return trans_con.decode() 

        elif type(content) is str: return content

        else: return content

    # switching the input lists to body focused lists self._ham_body_word_list+self._ham_body_word_list 
    # might provide better results since header has a lot of content not increasing entropy
    def import_as_data_frame(self):
        self._spam_ham_matrix = pd.DataFrame(self._spam_word_list+self._ham_word_list, columns=["Message", "Label"]).dropna()

    # builds a list of tuples representing a message 
    # [(list of words in message, label),...]
    def build_spam_word_list(self):
        self._spam_word_list = list()
        #self._spam_body_word_list = list()
    
        for document in self._spam_files:
            self._spam_word_list.append((self.get_email_msg_content(document), 'spam'))
            #self._spam_body_word_list.append((self.get_email_msg_body(document), 'spam'))

    # builds a list of tuples representing a message 
    # [(list of words in message, label),...]
    def build_ham_word_list(self):
        self._ham_word_list = list()
        #self._ham_body_word_list = list()
    
        for document in self._ham_files:
            self._ham_word_list.append((self.get_email_msg_content(document), 'ham'))
            #self._ham_body_word_list.append((self.get_email_msg_body(document), 'ham'))


class Text_Processor():

    '''
    _message

    _lematized_word_list
    # list will be taged, consists of tuples (word, pos_tag)
    _taged_word_list
    '''

    __stop_word_symbols= ['.',',','’',"“",';','´','\'','\"','`','?','!','$','%','*','&','=',':','(',')']
    
    def __init__(self, message, stop_words = True, lower = True, lemmatize=False, delete_markups=True, min_word_length=3, delete_duplicates=False):
        
        self.set_message(message)
        
        if delete_markups: self.remove_html_code()
        if stop_words: self.remove_stop_words()
        if lower: self.lower_message()
        if delete_duplicates: self.delete_duplicates()
        if lemmatize: self.lemmatize_message()

        self.set_min_word_length(min_word_length)
        
    def get_message(self):
        return self._message
            
    # returns a DataFrame of message and label columns
    def set_message(self, message):
        self._message = message

    # tokenize the message by words
    def tokenize_message_by_words(self):
        self._message = nltk.word_tokenize(self._message)

    # apply pos_tags
    def pos_tag_word_list(self):
        self._taged_word_list = nltk.pos_tag(self._message.split())

    # remove stop words
    def remove_stop_words(self):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        stop_words.update(self.__stop_word_symbols)
        filtered = [w for w in self._message.split() if not w in stop_words and w.isalpha()]
        self._message = " ".join(filtered)

    # lower case the message
    def lower_message(self):
        self._message = self._message.lower()

    # remove html or other markup tags
    def remove_html_code(self):
        pattern = r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});"
        prog = re.compile(pattern, re.MULTILINE)
        self._message = re.sub(prog, "", self._message)

    def delete_duplicates(self):
        self._message = " ".join(set(self._message.split()))

    def lemmatize_message(self):
        # WordNetLemmatizer requires Pos tags to classify word 
        # default is Noun
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        lemmatizer = nltk.stem.WordNetLemmatizer()
        self.pos_tag_word_list()
        self._lemmatized_word_list = list()

        for word, tag in self._taged_word_list:
            self._lemmatized_word_list.append(lemmatizer.lemmatize(word, tag_map[tag[0]]))

        self._message = " ".join(self._lemmatized_word_list)

    def set_min_word_length(self, min_word_length):

        words = self._message.split()
        self._message = " ".join([w for w in words if len(w)>=min_word_length])

# TODO build in precision, recall and f-score by nltk.metrics.scores -> (precision, recall)

class SpamHam_NLP_Classification():

    '''
    # pandas DataFrame of messages and labels
    _corpus_label_matrix

    # list of tuples list[(words_in_message, label)]
    _corpus_label_list

    # Frequency Distribution of words per document
    _word_freq_distribution

    # list of all words in messages
    _all_words

    # n (3000 default) most common words
    _features
    # list of tuples (feature_dict, label)
    # feature_dict is a hash map of all feature words and if they are within the message or not
    _feature_set

    _training_set, _testing_set

    #NLTK NB:
    NaiveBayesClassifier

    # ML Classifier:
    MultiNomialNB_Classifier 
    GaussianNB_Classifier
    BernoulliNB_Classifier 
    SVC_Classifier
    LinearSVC_Classifier
    SGD_Classifier 
    LogReg_Classifier
    
    # technically those are comitees of classifiers
    RandomForest_Classifier
    AdaBoost_Classifier 
    '''

    def __init__(   self, 
                    corpus_label_matrix,
                    NLTK_NB=True,
                    MultiNomialNB=True, 
                    GaussianNB=True, 
                    BernoulliNB=True, 
                    SVC=True,
                    LinearSVC=True,
                    SGD=True, 
                    LogReg=True, 
                    RandomForest=True,
                    AdaBoost = True):

        self.set_corpus_label_matrix(corpus_label_matrix)

        self.set_corpus_label_list()
        np.random.shuffle(self._corpus_label_list)
        
        
        self.set_all_words()
        self.set_freq_distribution()
        
        # get the n most common words as features out of self._unique_words
        self.set_features()
        self.build_feature_set()

        self.split_train_test()

        if NLTK_NB: self.NB_NLP_classify()
        if MultiNomialNB: self.MultiNomial_NB_classify()
        #!!! DO NOT USE FOR NOW Gaussian NB !!!
        #if GaussianNB: self.Gaussian_NB_classify()
        if BernoulliNB: self.Bernoulli_NB_classify()
        if SVC: self.SVC_classify()
        if LinearSVC: self.LinearSVC_classify()
        if SGD: self.SGD_classify()
        if LogReg: self.LG_classify()
        if RandomForest: self.randomforest_classify()
        if AdaBoost: self.adaboost_classify()
        
    def set_corpus_label_matrix(self, corpus_label_matrix):
        self._corpus_label_matrix = corpus_label_matrix

    def set_all_words(self):
        self._all_words = list()

        for message in self._corpus_label_list:
            self._all_words += message[0]

    def set_freq_distribution(self):
        self._word_freq_distribution = nltk.FreqDist(self._all_words)

    def set_features(self, feature_count=3000):
        #skiping the first few words since the most common words do not provide any entropy
        self._features = list(self._word_freq_distribution.keys())[300:feature_count]

    def get_corpus_label_list(self):
        return self._corpus_label_list

    def get_feature_set(self):
        return self._feature_set

    def set_corpus_label_list(self):
        self._corpus_label_list = list()

        for (index, message) in enumerate(self._corpus_label_matrix['Message']):
            doc = (self.pre_process_message(message).split(), self._corpus_label_matrix.loc[index]['Label'])
            self._corpus_label_list.append(doc)

    def pre_process_message(self, message):

        # remove stop words and html, lower text, lemmatize
        text_processor = Text_Processor(message, lemmatize=True)
        return text_processor.get_message()

    def find_features(self, message):
        words = set(message)
        features = dict()

        #build a hash map of features and boolean if feature is in message or not
        for w in self._features:
            features[w] = (w in words)

        return features
    
    def build_feature_set(self):

        self._feature_set = [(self.find_features(document), label) for (document, label) in self._corpus_label_list]

    # ratio is training = len(self._feature_set)*ratio
    def split_train_test(self, ratio=0.8):
        
        split_amount = int(len(self._feature_set)*ratio)
        self._training_set = self._feature_set[:split_amount]
        self._testing_set = self._feature_set[split_amount:]

    # accuracy is highly volatile
    def NB_NLP_classify(self):

        self.NaiveBayesClassifier = nltk.NaiveBayesClassifier.train(self._training_set)
        print("Naive Bayes Accuracy Score ::: ", (nltk.classify.accuracy(self.NaiveBayesClassifier, self._testing_set))*100)
        self.NaiveBayesClassifier.show_most_informative_features(20)

    def MultiNomial_NB_classify(self):
        self.MultiNomialNB_Classifier = SklearnClassifier(MultinomialNB())
        self.MultiNomialNB_Classifier.train(self._training_set)
        print("MultiNomial Naive Bayes Accuracy Score ::: ", (nltk.classify.accuracy(self.MultiNomialNB_Classifier, self._testing_set))*100)
    
    # !!! conversion of training set for Gaussian NB does not work !!!
    # !!! training set of word_feature is recognized as sparse matrix !!!
    # !!! error for dense conversion is thrown, definitely a bug !!!
    def Gaussian_NB_classify(self):
        self.GaussianNB_Classifier = SklearnClassifier(GaussianNB())
        self.GaussianNB_Classifier.train(self._training_set)
        print("Gaussian Naive Bayes Accuracy Score ::: ", (nltk.classify.accuracy(self.GaussianNB_Classifier, self._testing_set))*100)
        
    def Bernoulli_NB_classify(self):
        self.BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
        self.BernoulliNB_Classifier.train(self._training_set)
        print("Bernoulli Naive Bayes Accuracy Score ::: ", (nltk.classify.accuracy(self.BernoulliNB_Classifier, self._testing_set))*100)

    def SVC_classify(self):
        self.SVC_Classifier = SklearnClassifier(SVC(gamma='auto'))
        self.SVC_Classifier.train(self._training_set)
        print("C Support Vector Accuracy Score ::: ", (nltk.classify.accuracy(self.SVC_Classifier, self._testing_set))*100)
        
    def LinearSVC_classify(self):
        self.LinearSVC_Classifier = SklearnClassifier(LinearSVC())
        self.LinearSVC_Classifier.train(self._training_set)
        print("Linear Support Vector Classifier Accuracy Score ::: ", (nltk.classify.accuracy(self.LinearSVC_Classifier, self._testing_set))*100)

    def SGD_classify(self):
        self.SGD_Classifier = SklearnClassifier(SGDClassifier())
        self.SGD_Classifier.train(self._training_set)
        print("Stochastic Gradient Descent Classifier Accuracy Score ::: ", (nltk.classify.accuracy(self.SGD_Classifier, self._testing_set))*100)

    def LG_classify(self):
        self.LogReg_Classifier = SklearnClassifier(LogisticRegression(solver='lbfgs'))
        self.LogReg_Classifier.train(self._training_set)
        print("Logistic Regression Descent Classifier Accuracy Score ::: ", (nltk.classify.accuracy(self.LogReg_Classifier, self._testing_set))*100)

    # leaving metric to gini for now, could be tested with entropy as well
    # estimator set to 100 trees
    # no max depth, more precise at min_samples_split()
    # bootstrap is True to provide decrease in bias
    def randomforest_classify(self):
        self.RandomForest_Classifier = SklearnClassifier(RandomForestClassifier(n_estimators=100))
        self.RandomForest_Classifier.train(self._training_set)
        print("Random Forest Classifier Accuracy Score ::: ", (nltk.classify.accuracy(self.RandomForest_Classifier, self._testing_set))*100)

    # default alpha and weight are fairly reasonable
    def adaboost_classify(self):
        self.AdaBoost_Classifier = SklearnClassifier(AdaBoostClassifier())
        self.AdaBoost_Classifier.train(self._training_set)
        print("AdaBoost Classifier Accuracy Score ::: ", (nltk.classify.accuracy(self.AdaBoost_Classifier, self._testing_set))*100)
        

d = Import_Data()
cl = SpamHam_NLP_Classification(d.get_spam_ham_matrix())

#print(d.get_spam_ham_matrix().head())

