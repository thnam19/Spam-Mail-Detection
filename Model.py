
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words('english'))

Spam = 1
Ham = 0

class TextProgress:
    def __init__(self,text):
        self.text = text
    
    def lower(self):
        return self.text.lower()

    def words(self):
        lower = self.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(lower)
        return words

    def remove_StopWords(self):
        clean_text = self.words()
        message_after = []
        for i in clean_text:
            if i not in stop_words:
                message_after.append(i)
        return " ".join(message_after)
    
    def convert_message(self):
        stemmer = PorterStemmer()
        text_convert = self.remove_StopWords().split()
        stem = [stemmer.stem(word) for word in text_convert]
        return " ".join(stem)


class NaiveBayes:
    def __init__(self):
        self.PriorHam = 1
        self.PriorSpam = 1
        self.dict_vocab_ham = {}
        self.dict_vocab_spam = {} 
        self.spam_true_positives = 0
        self.spam_false_negatives = 0
        self.ham_true_positives = 0
        self.ham_false_negatives = 0
        self.total_values_ham = 0
        self.total_values_spam = 0 
    
    """
    2 functions to build dictionary of spam and ham
    """
    def dictionary_ham(self,all_text_ham):
        all_text = all_text_ham.split()
        for word_ham in all_text:
            if word_ham in self.dict_vocab_ham:
                self.dict_vocab_ham[word_ham] += 1
            else:
                self.dict_vocab_ham[word_ham] = 1
    
    def dictionary_spam(self,all_text_spam):
        all_text_new = all_text_spam.split()
        for word_spam in all_text_new:
            if word_spam in self.dict_vocab_spam:
                self.dict_vocab_spam[word_spam] += 1
            else:
                self.dict_vocab_spam[word_spam] = 1
    
    """
    fuction to caculate each value of all spam and all ham text
    """
    def total(self):
        for spam in self.dict_vocab_spam.values():
            self.total_values_spam += spam
        for ham in self.dict_vocab_ham.values():
            self.total_values_ham += ham

    """
    function in oder to add more data for training 
    if test dataset has some words which don't exist in train dataset
    """
    def check_test_word(self,col_df_test):
        for message in col_df_test:
            for word in message.split():
                if word not in self.dict_vocab_spam.keys():
                    if word in self.dict_vocab_ham.keys():
                        for word_spam in self.dict_vocab_spam.keys():
                            self.dict_vocab_spam[word_spam] += 1
                        for word_ham in self.dict_vocab_ham.keys():
                            self.dict_vocab_ham[word_ham] += 1
                        self.dict_vocab_spam[word] = 1
                    else:
                        for word_spam in self.dict_vocab_spam.keys():
                            self.dict_vocab_spam[word_spam] += 2
                        for word_ham in self.dict_vocab_ham.keys():
                            self.dict_vocab_ham[word_ham] += 2
                        self.dict_vocab_spam[word] = 1
                        self.dict_vocab_ham[word] = 1
                else:
                    if word not in self.dict_vocab_ham.keys():
                        for word_spam in self.dict_vocab_spam.keys():
                            self.dict_vocab_spam[word_spam] += 1
                        for word_ham in self.dict_vocab_ham.keys():
                            self.dict_vocab_ham[word_ham] += 1
                        self.dict_vocab_ham[word] = 1        
                    else:
                        pass

    """
    caculate probability of each word in Spam data and Ham data
    """    
    def Calculate_PriorHam(self,word):
        PriorHam = (self.dict_vocab_ham[word]/self.total_values_ham)
        return PriorHam
    
    def Calculate_PriorSpam(self,word):
        PriorSpam = (self.dict_vocab_spam[word]/self.total_values_spam)
        return PriorSpam
    
    def add_Result_predict(self,predict,true_label,score_ham,score_spam,last_result):
        result = str([predict,true_label,score_ham,score_spam,last_result])
        return result
    
    """
    caculate confusion matrix's variable
    """
    def confusion_matrix(self,label,predict):
        if label == Spam and predict == Spam:
            self.spam_true_positives += 1 #tp
        elif label == Ham and predict == Ham:
            self.ham_true_positives += 1.#tn
        elif label == Spam and predict == Ham:
            self.spam_false_negatives += 1 #fn
        else:
            self.ham_false_negatives += 1 #fp
    
    """ 
    add result as 1:[ham,ham,0.2,0.1,right]
                  2:[spam,ham,0.01,0.2,wrong]
    """
    def predict(self,message,label):
        score_ham = self.PriorHam
        score_spam = self.PriorSpam
        predict = ""
        result = ""
        ham = (self.total_values_ham/(self.total_values_ham + self.total_values_spam))
        spam = (self.total_values_spam/(self.total_values_ham + self.total_values_spam))
        for word in message.split():
            score_ham *= ham
            score_spam *= spam
            score_ham *= self.Calculate_PriorHam(word)
            score_spam *= self.Calculate_PriorSpam(word)
            
        if score_ham > score_spam:
            predict = Ham
        else:
            predict = Spam

        if predict == label:
            result = f"Right"
        else:
            result = f"Wrong"

        self.confusion_matrix(label,predict)
        return self.add_Result_predict(predict,label,score_ham,score_spam,result)

    """
    4 funcitons to caculate model's parameter
    """
    def accuracy(self):
        return (self.ham_true_positives + self.spam_true_positives)/(
                self.spam_true_positives + self.ham_true_positives + self.spam_false_negatives + self.ham_false_negatives)
    
    def Precision(self):
        precision = self.spam_true_positives/(self.ham_false_negatives+self.spam_true_positives)
        return precision
    
    def recall(self):
        return (self.spam_true_positives/(self.spam_true_positives+self.spam_false_negatives))
    
    def f1_matrix(self):
        precision = self.Precision()
        recall = self.recall()
        return 2 * (precision*recall)/(precision+recall)

    def Print_confusionMatrix(self):
        TP = self.spam_true_positives
        TN = self.ham_true_positives
        FP = self.ham_false_negatives
        FN = self.spam_false_negatives
        confusion = (
                            "----------|-----Spam-predicted--------Ham-predicted----------" + "\n"
                            "Spam-True |           {TP}        |        {FP}             |" + "\n"
                            "-------------------------------------------------------------" + "\n"
                            "Ham-True  |            {FN}       |        {TN}             |" + "\n"
                            "-------------------------------------------------------------"
        )
        return confusion.format(TP=TP, FP=FP, FN=FN, TN=TN)


