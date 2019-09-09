import pandas as pd 
import numpy as np
from preprocessing import Preprocess
from vectorizing import Vectors
from datasetBalance import DatasetBalance
from classifier import Classifier
import joblib 

# readin the training and testing files in the dataframe
train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

# converting the columns into list with UTF-8 encoding 
train_text = train_data['Review Text'].values.astype('U').tolist()
train_title = train_data['Review Title'].values.astype('U').tolist()
train_rating = train_data['Star Rating'].values.astype('U').tolist()
test_text = test_data['Review Text'].values.astype('U').tolist()

# converting the rating into integer
for i in range(len(train_rating)):
    train_rating[i]=int(train_rating[i])

# instanciating the Preprocessrt,Vectorizer,DatasetBalancer and Classifier
pre=Preprocess()
v=Vectors()
balance=DatasetBalance()

# preprocess the train data
x=[]
for i in train_text:
  x.append(" ".join(pre.preprocessing(i)))

print("traing data ready")

print("now making count vectors")
# making count vectors
x_cv=v.count_vectors(x,train=True)

y=train_rating

print("saving the vectorizer")
# saving the vectorizer
joblib.dump(v,"models/cv_vectorizer.sav")


print("oversampling the Count Vector")
# oversampling the Count Vectorizer 
x_balanced,y_balanced=balance.oversample1(x_cv,y)

clf1=Classifier()
print("Training using the Count Vectors")
# traing usnig he count vectors 
clf1.classify_all(x_balanced,y_balanced)
joblib.dump(clf1,"models/cv_classifier.sav")


