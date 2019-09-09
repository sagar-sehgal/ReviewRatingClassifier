import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string 
import re

class Preprocess:
    ''' 
    This class is used to preprocess the text using various techniques. It preprocesses the data sentence. 
    How to use
    >>> pre = Preprocess()
    >>> pre.preprocessing("Your sentence goes here")
    ['sentence', 'go']
    Will get the preprocessed sentence in form f a list
    '''
    
    def __init__(self):
        ''' 
        Initializer function. Intitializes various classes.
        '''
        
        # Initializing the wordnet Lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Initializing the Tweet Tokenizer
        self.tweettoken=TweetTokenizer()
    
        # getting the list of all stop words from nltk
        self.stop_words = set(stopwords.words('english'))

        # now some of the words are removed from th stopword list, becuae these word specifically indicate a negative sentiment 
        self.stop_words.discard("not")
        self.stop_words.discard("didn't")
        self.stop_words.discard("doesn't")
        self.stop_words.discard("wasn't")
        self.stop_words.discard("shouldn't")
        self.stop_words.discard("needn't")
        self.stop_words.discard("hasn't")
        self.stop_words.discard("haven't")
        self.stop_words.discard("hadn't")
        self.stop_words.discard("don't")
        
        # some words that have to be removed ... unnecessary words
        self.remove_words=[".","..","..."]
        
    def sent_tokenize(self,data):
        '''
        Used to break the review into sentences. But we don't need to break the review into sentences
        '''
        
        # joining all the sentnces in the review by emoving '.' 
        data=data.replace("."," ")
        return data

    def word_tokenize(self,sent):
        '''
        Breaks the sentence into tokens. 
        Tweet Tokenizer has been specifically used for thsi task 
        since the reviews were a type of free text and contained emojis 
        and various other non-inmportant information which was non-relevant
        '''
        return self.tweettoken.tokenize(sent)
    
    def lemmatize(self,word):
        '''
        This function lemmatizes the word and brings it to a common word format.
        Also # was removed from the hastag words ... so that they can be processed further
        '''
        if("#" in word):
            word=word.replace("#","")
        return self.lemmatizer.lemmatize(word)
    
    def is_stop_word(self,word):
        '''
        Function used to check if a given word lies in our stopword list
        '''
        if(word.lower() in self.stop_words):
            return True
        return False
    
    def has_number(self,word):
        '''
        Used to check if a word has a number
        '''
        return any(char.isdigit() for char in word)
    
    def deEmojify(self, word):
        '''
        used to remove emoji from a word
        '''
        return word.encode('ascii', 'ignore').decode('ascii')

    def is_extra_word(self,word):
        '''
        Removes all extra words by calling various functions defined above
        like removing punctuation words, words having numbers, @ words, stop words
        or any special words which has been marked in remove_words list
        '''
        if(self.has_number(word.lower())):
            return True
        if('@' in word.lower()):
            return True
        if(self.is_stop_word(word.lower())):
            return True
        if(word.lower() in string.punctuation):
            return True
        if(word.lower() in self.remove_words):
            return True
        if(len(word)==0):
            return True
        return False
    
    def reduce_lengthening(self,word):
        '''
        Shortens the words, if the characters in a word repeats.
        It would be redcued to having at max of 2 repeating chars.
        For eg:- aweeeesome=> aweesome
        '''
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", word)
        
    def preprocessing(self,data):
        '''
        This method uses all the above methods to preprocess a complete sentence.
        The argument given is the review which we want to process.
        '''
        
        # joining all the sentences in a review
        sents=self.sent_tokenize(data)
        # spliiting a sentence into words
        word_tokenized=self.word_tokenize(sents)
        
        # processing all the words and applying varios functions
        words=[]
        for j in word_tokenized:
            w=j.lower()
            w=self.deEmojify(w)
            w=self.reduce_lengthening(w)
            w=self.lemmatize(w)
            if(self.is_extra_word(w)==False):
                words.append(w)
        return words
