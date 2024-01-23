
from Model import TextProgress
from Model import NaiveBayes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
result_doc = f'last_result.txt'

df = pd.read_csv("Spam_ham.csv",encoding='latin',engine='python')
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
df.rename(columns={'v1':'label','v2':'message'},inplace=True)
df.drop_duplicates(keep='first',inplace=True) 
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])
ham = df[df['label']==0]
spam = df[df['label']==1]
ham_convert = ham.sample(n=len(spam),random_state=44)
new_df = pd.concat([ham_convert,spam],ignore_index=True)
new_df = new_df.sample(frac=1).reset_index(drop=True)
X = new_df.iloc[:,0]
y = new_df.iloc[:,1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
df_train = pd.concat([X_train,y_train],axis=1).reset_index(drop=True)
df_test = pd.concat([X_test,y_test],axis=1).reset_index(drop=True)

df_train.to_csv('Train_data.txt',sep='\t',index=False)
df_test.to_csv('Test_data.txt',sep='\t',index=False)

def convert_col_text(df_col):
    df_col = df_col.apply(lambda x: TextProgress(x).convert_message())
    return df_col

def run():
    df_train['message'] = convert_col_text(df_train['message'])
    df_test['message'] = convert_col_text(df_test['message'])
    all_text_spam = " ".join(df_train[df_train['label']==1]['message'])
    all_text_ham = " ".join(df_train[df_train['label']==0]['message'])
    dict_message_label_test = dict(zip(df_test['message'],df_test['label']))
    
    naive_bayes = NaiveBayes()
    naive_bayes.dictionary_ham(all_text_ham)
    naive_bayes.dictionary_spam(all_text_spam)
    naive_bayes.total()
    naive_bayes.check_test_word(df_test['message'])
    f = open(result_doc,"w")
    for message,label in dict_message_label_test.items():
        result = naive_bayes.predict(message,label)
        f.write(str(result)+ "\n")
    f.close()

    print("Accuracy:  "+ str(naive_bayes.accuracy())+ "\n" )
    print("Precision: "+ str(naive_bayes.Precision())+ "\n"  )
    print("recall:    "+ str(naive_bayes.recall())+ "\n"  )
    print("F1:        "+ str(naive_bayes.f1_matrix())+ "\n"  )
    print(naive_bayes.Print_confusionMatrix())

run()