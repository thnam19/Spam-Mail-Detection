# Spam-Mail-Detection
A spam Detector using Naive Bayes Algorithm.

### File structure
* Model.py:
    * Text Progress following 4 steps are Lower case, Tokenization, removing Stop Words, and Stemming.
    * Using NaiveBayes Algorithm, Calculating parameters Accuracy, Precision, Recall, F1.
    * This file also contains code for building data from Train_data, predicting class of emails as Spam or Ham
      
* Run.py: this file predicts on test_data and displays the performance Measures.

### Confusion Matrix

|                  | SPAM (Predicted)   | HAM (Predicted)   |
|------------------|--------------------|-------------------|
| SPAM-True        |       TP = 119     |       FN = 27     |
| HAM-True         |       FP =   3     |       TN = 113    |

Accuracy:    0.8854961832061069

Precision:   0.815068493150685

Recall:      0.9754098360655737

f1-measure:  0.8880597014925373
  
