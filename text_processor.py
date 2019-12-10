import email, os, sys, nltk, re

import pandas as pd
import numpy as np

from collections import defaultdict

from nltk.corpus import wordnet as wn

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score


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
    
    def __init__(self, message, stop_words = True, lower = True, lemmatize=False, delete_markups=True, delete_duplicates=False):
        
        self.set_message(message)
        
        if delete_markups: self.remove_html_code()
        if stop_words: self.remove_stop_words()
        if lower: self.lower_message()
        if delete_duplicates: self.delete_duplicates()
        if lemmatize: self.lemmatize_message()
        
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

class Data_Transform():

    '''
    Data_Frame of imported Data
    # input corpus consists of ['Message', 'Label']
    # transforms into ['Message', 'Label', 'processed_message']
    _corpus
    
    _max_features

    # SciPy sparse matrix of data
    # BOW or TFIDF
    _vectorized_data

    _Train_X
    _Train_Y

    # encoded target vectors
    _Test_X
    _Test_Y

    _vectorized_Train_X
    _vectorized_Test_X
    '''

    def __init__(self, corpus, max_features=5000):
        self._corpus = corpus
        self._max_features = max_features

        # remove stop words and html, lemmatize, lower the message 
        # delete duplicates 
        self.prepare_corpus()

        # split text into train and test data by ratio
        self.split_data(ratio=0.3)

        # digitize raw labels into numbers
        self.encode_labels()

        # one of those two algorithms could be used to vectorize
        # tfidf counters word frequency in large document though
        
        #self.vectorize_tf()
        self.vectorize_tfidf()

    def get_data_vectors(self):
        return self._vectorized_Train_X, self._vectorized_Test_X, self._Train_Y, self._Test_Y

    def prepare_corpus(self):
        for index, message in enumerate(self._corpus['Message']):

            #this configuration removes stop words and html, lemmatizes and lowers the message
            text_processor = Text_Processor(message, lemmatize=True)
            self._corpus.loc[index, 'processed_message'] = text_processor.get_message()

    def encode_labels(self):
        encoder = LabelEncoder()
        self._Train_Y = encoder.fit_transform(self._Train_Y)
        self._Test_Y = encoder.fit_transform(self._Test_Y)

    def split_data(self, ratio=0.2):
        self._Train_X, self._Test_X, self._Train_Y, self._Test_Y = model_selection.train_test_split(self._corpus['processed_message'], self._corpus['Label'], test_size=ratio)

    # vectorize into bow model using CountVectorizer
    def vectorize_tf(self):
        vectorizer = CountVectorizer(max_features=self._max_features)
        vectorizer.fit(self._corpus['processed_message'])
        self._vectorized_Train_X = vectorizer.transform(self._Train_X)
        self._vectorized_Test_X = vectorizer.transform(self._Test_X)

    # vectorize into bow model using CountVectorizer
    def vectorize_tfidf(self):
        vectorizer = TfidfVectorizer(max_features=self._max_features)
        vectorizer.fit(self._corpus['processed_message'])
        self._vectorized_Train_X = vectorizer.transform(self._Train_X)
        self._vectorized_Test_X = vectorizer.transform(self._Test_X)

    def save_data_set(self):
        np.savetxt("Train_X.txt", self._vectorized_Train_X.toarray())
        np.savetxt("Test_X.txt", self._vectorized_Test_X.toarray())
        np.savetxt("Train_Y.txt", self._Train_Y.toarray())
        np.savetxt("Test_Y.txt", self._Test_Y.toarray())

class SpamHam_ML_Classification():

    '''
    _vectorized_Train_X, _vectorized_Test_X,
    _Train_Y, _Test_Y
    '''

    def __init__(self,
                vectorized_Train_X,
                vectorized_Test_X,
                Train_Y, Test_Y,
                MultiNomial_NB=True, 
                GaussianNB=True, 
                BernoulliNB=True, 
                SVC=True,
                SGD=True, 
                LogReg=True, 
                RandomForest=True,
                AdaBoost = True):

        self._vectorized_Train_X = vectorized_Train_X
        self._vectorized_Test_X = vectorized_Test_X
        self._Train_Y = Train_Y 
        self._Test_Y = Test_Y

        if MultiNomial_NB: self.MultiNomial_NB_classify()

    def MultiNomial_NB_classify(self):
        classifier = MultinomialNB()
        classifier.fit(self._vectorized_Train_X, self._Train_Y)
        predictions_NB = classifier.predict(self._vectorized_Test_X)
        print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, self._Test_Y)*100)

    def Gaussian_NB_classify(self):
        classifier = GaussianNB()


d = Import_Data()
dt = Data_Transform(d.get_spam_ham_matrix())
trx, tex, tr_y, te_y = dt.get_data_vectors()
cl = SpamHam_ML_Classification(trx, tex, tr_y, te_y)

#print(d.get_spam_ham_matrix().head())

