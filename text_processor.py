import email, os, sys, nltk
import pandas as pd
import numpy as np


class Import_Data():

    '''
    _spam_files
    _ham_files
    _spamnham_files

    _cwd

    _ham_word_list
    _spam_word_list

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

    def import_as_data_frame(self):
        self._spam_ham_matrix = pd.DataFrame(self._spam_word_list+self._ham_word_list, columns=["Message", "Label"])

    # builds a list of tuples representing a message 
    # [(list of words in message, label),...]
    def build_spam_word_list(self):
        self._spam_word_list = list()
    
        for document in self._spam_files:
            self._spam_word_list.append((self.get_email_msg_content(document), 'spam'))

    # builds a list of tuples representing a message 
    # [(list of words in message, label),...]
    def build_ham_word_list(self):
        self._ham_word_list = list()
    
        for document in self._ham_files:
            self._ham_word_list.append((self.get_email_msg_content(document), 'ham'))


class Text_Processor():

    
    def __init__(self):
        pass
            
    # documents is list of tuples (words, label)
    def set_all_words(self, documents):
        pass


d = Import_Data()
print(d.get_spam_ham_matrix().head())


'''
a = os.path.dirname(sys.argv[0])+"/Data_Sets/ham/"
documents = os.listdir(a)

with open('/Users/t/projects/spamham_classification/Data_Sets/spam/0488.6d41f6d7222978a3ee2b6cfbfce55a02', 'r') as f:
    b = email.message_from_file(f)
body = ""

if b.is_multipart():
    for part in b.walk():
        ctype = part.get_content_type()
        cdispo = str(part.get('Content-Disposition'))

        # skip any text/plain (txt) attachments
        if ctype == 'text/plain' and 'attachment' not in cdispo:
            body = part.get_payload(decode=True)  # decode
            break
# not multipart - i.e. plain text, no attachments, keeping fingers crossed
else:
    body = b.get_payload(decode=True)


if type(body) is bytes:
    print(type(body.decode()))
'''