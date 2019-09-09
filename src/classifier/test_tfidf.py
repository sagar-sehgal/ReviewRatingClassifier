import pandas as pd 
import numpy as np
import joblib
from preprocessing import Preprocess
from vectorizing import Vectors
from datasetBalance import DatasetBalance
from classifier import Classifier


print("getting the test data")
# reading the test data and converig it to list with URTF-8 encoding
test_data = pd.read_csv("./test.csv")
test_id = test_data['id'].values.astype('U').tolist()
test_text = test_data['Review Text'].values.astype('U').tolist()

# loading the vectorizer
print("loading the vectorizer")
v=joblib.load("models/tfidf_vectorizer.sav")

# preprocessig the text data
print("preprocessig the text data")
pre=Preprocess()
x=[]
for i in test_text:
	x.append(" ".join(pre.preprocessing(i)))

# converting the text to vectors
print("converting the text to vectors")
x_tfidf=v.tf_idf_vectors(x,train=False)

clf2=joblib.load("models/tfidf_classifier.sav")
pred2=clf2.predict_all(x_tfidf)

# the best model while crossvalidation was that of DecisionTreeClassifier
f1=open("../submission_tfidf.csv","w")
f1.write("id , Star Rating \n")
for i,j in zip(test_id,pred2["DecisionTreeClassifier"]):
	f1.write(i+","+str(j)+"\n")
f1.close()
