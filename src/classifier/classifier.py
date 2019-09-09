from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib

class Classifier:
    '''
    This class keeps the various classifiers
    Classifier.classify_all(x,y)=> Trains the classifer on the complete dataset
    Classifier.predict_all(x,y)=> Predicts using all the classifiers
    '''
    def __init__(self):
        RAN_STATE=1
        self.mnb = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=False) 
        self.lrc = LogisticRegression(C=3.730229437354635, penalty='l2',solver='liblinear',multi_class='auto', random_state = RAN_STATE)
        self.dtc = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=2,random_state = RAN_STATE)

    def classify_all(self,x,y):
        self.clf_train_score={}
        
        print("training MultinomialNB")
        self.mnb.fit(x.toarray(),y)
        scores_mnb=cross_val_score(self.mnb,x.toarray(),y,cv=10,scoring="f1_weighted")
        
        print("training LogisticRegression")
        self.lrc.fit(x.toarray(),y)
        scores_lrc=cross_val_score(self.lrc,x.toarray(),y,cv=10,scoring="f1_weighted")
        
        print("training DecisionTreeClassifier")
        self.dtc.fit(x.toarray(),y)
        scores_dtc=cross_val_score(self.dtc,x.toarray(),y,cv=10,scoring="f1_weighted")
        
        # print("saving MultinomialNB")
        # joblib.dump(self.mnb,"models/mnb.sav")
        
        # print("saving LogisticRegression")
        # joblib.dump(self.lrc,"models/lrc.sav")
        
        # print("saving DecisionTreeClassifier")
        # joblib.dump(self.dtc,"models/dtc.sav")
        
        
        self.clf_train_score["MultinomialNB"]=scores_mnb.mean()
        self.clf_train_score["LogisticRegression"]=scores_lrc.mean()        
        self.clf_train_score["DecisionTreeClassifier"]=scores_dtc.mean()
        
    def predict_all(self,x):
        self.clf_test_pred={}
        
        # print("loading MultinomialNB")
        # self.mnb = joblib.load("models/mnb.sav")
        
        # print("loading LogisticRegression")
        # self.lrc = joblib.load("models/lrc.sav")
        
        # print("loading DecisionTreeClassifier")
        # self.dtc = joblib.load("models/dtc.sav")
        
        
        print("predicting MultinomialNB")
        pred_mnb = self.mnb.predict(x)
        
        print("predicting LogisticRegression")
        pred_lrc = self.lrc.predict(x)
        
        print("predicting DecisionTreeClassifier")
        pred_dtc = self.dtc.predict(x)
        
        self.clf_test_pred["MultinomialNB"]=pred_mnb
        self.clf_test_pred["LogisticRegression"]=pred_lrc
        self.clf_test_pred["DecisionTreeClassifier"]=pred_dtc
        
        return self.clf_test_pred