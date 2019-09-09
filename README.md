
# REVIEW RATING PREDICTION

### How to install

1. Make the viratual enviornment
```
virtualenv venv --python=python3
```

2. Activate the virtual enviornment
```
source venv/bin/activate
```

3. Install the dependencies
```
pip install -r rquirements.txt
```

### See the data analysis
 The data analysis can be seen in the file `src/data_analysis.ipynb` . This is a Jupyter Notebook which would be showing the data analysis of the dataset.

### The Predicitons
2 types of vectors with 3 types of classifiers are used. i used Cross Validation over these models to get the results. Results can be seen as follows:-
-   With Count Vectors
	-   MultinomialNB- 0.8257    
	-   Logistic Regression- 0.8630  
	-   Decision Tree Classifier- 0.8005
-   With TF-IDF Vectors
	-   MultinomialNB- 0.6942
	-   Logistic Regression- 0.7756    
	-   Decision Tree Classifier- 0.7916	

### Files in the `src/classifier` folder

### Reproduce the results
The model can be retrained. For that go to `src/classifier` folder.
To train the model using the Count vectors run:-
```
python3 train_cv.py
```
This will train the model and save them in `src/classifier/models` folder.
To test the classifier, run the following command:-
```
python3 test_cv.py
``` 
A `src/submission_tfidf.csv`  would be produced as a result of this.

To train the model using the TF-IDF vectors run:-
```
python3 train_tfidf.py
```
This will train the model and save them in `src/classifier/models` folder.
To test the classifier, run the following command:-
```
python3 test_tfidf.py
``` 
A `src/submission_tfidf.csv`  would be produced as a result of this.
