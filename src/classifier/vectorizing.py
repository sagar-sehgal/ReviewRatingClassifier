from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Vectors:
    '''
    This class converts the data into vectors. 
    It can both train as well as build the vectors from the existing vocabulary.
    How to use
    >>> v=Vectors()
    >>> x=["Hello world"," hello", "hi hello"," hi everyone"]
    >>> v.count_vectors(x)

    '''
    def __init__(self):
        # definign the basic hyper parameters of count vectorizer and TF-IDF vectorizer
        self.cv=CountVectorizer(ngram_range=(1,2))
        self.tfidf=TfidfVectorizer(min_df=0.001,ngram_range=(1,2))
    
    def tf_idf_vectors(self,data,train=False):

        # if train=True fit the data to build up TF-IF vectors
        if(train):
            vector=self.tfidf.fit_transform(data)
        # if train=False make the TF-IDF vectors from the previous vocabulary
        else:
            vector=self.tfidf.transform(data)
        
        return vector
    
    def count_vectors(self,data,train=False):
        # if train=True fit the data to build up Count vectors
        if(train):
            vector=self.cv.fit_transform(data)
        # if train=False make the Count vectors from the previous vocabulary
        else:
            vector=self.cv.transform(data)
        
        return vector
