Thoughts:
Variety could be a problem since there are 105 different categories.
Label Encoding might lead to misleading interpretations of the categorical data
One shot leads to massive tables
Target Encoding is suggested, but I'm not sure

It seems that variety plays a big factor into where the wine is from.

country
US        622
Italy     174
France    133
Spain      71
Name: count, dtype: int64

Imbalance of data

For metric, I will use weighted F1-score as it (takes the class imbalance into account by weighting each class's
F1-score by its proportion of samples in the dataset.)

Bag of words
72 % without stop words
78 % with stop words

TF-IDF Vectorisation
79 % with stop words

Sentiment analysis

79 % - had no effect
152 neg
837 pos
11 neu

"From a site near Annapolis, this wine shows a preponderance of dark grape and cherry flavor interwoven with cinnamon
and black peppercorn. Light-bodied and ethereally layered, it has touches of rustic earthiness and leather."
Is considered negative


LLM Usage
1: Asking LLM to predict based on desc + details without local model
2: LLM-guided Text Vectorisation (to reduce the features to context relevant ones)
3: Extract specific features (could be missing) and performing RF

1: LMM without local model:
Overall Accuracy: 72.80% (728/1000 correct)

2: LLM-guided Text Vectorisation
When testing with wine_quality_10
It added 108 new features, compared to 180 (without LLM filtering)
Overall Accuracy 79% - no difference compared to without LLM

3: LLM Feature Extraction
There are two ways of doing it: (1) Predefined Categories and (2) Dynamic Category Building.
Chose to do (1) because every row is compared against the same categories, reducing noise and overfitting.
Takes a long time to run ~20s per row.
percentage not calcualted yet TODO


C:\Users\Alexa\AppData\Local\Programs\Python\Python310\python.exe C:\Programming\Python\ArtificialLabs\src\main.py 
Data loaded successfully with 500 rows.
Data preprocessing completed.
Best Hyperparameters: {'max_depth': 30, 'n_estimators': 100}
Best Cross-Validation Accuracy: 0.55
Test Accuracy: 0.56

Classification Report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00         7
           1       0.20      0.14      0.17         7
           2       0.00      0.00      0.00         5
           3       0.61      0.87      0.72        31

    accuracy                           0.56        50
   macro avg       0.20      0.25      0.22        50
weighted avg       0.41      0.56      0.47        50


Confusion Matrix:
 [[ 0  0  0  7]
 [ 0  1  0  6]
 [ 0  1  0  4]
 [ 1  3  0 27]]

Process finished with exit code 0




C:\Users\Alexa\AppData\Local\Programs\Python\Python310\python.exe C:\Programming\Python\ArtificialLabs\src\main.py 
Data loaded successfully with 500 rows.
Data preprocessing completed.
Test Accuracy: 0.66

Classification Report:
               precision    recall  f1-score   support

           0       0.50      0.14      0.22         7
           1       0.60      0.43      0.50         7
           2       0.00      0.00      0.00         5
           3       0.67      0.94      0.78        31

    accuracy                           0.66        50
   macro avg       0.44      0.38      0.38        50
weighted avg       0.57      0.66      0.59        50


Confusion Matrix:
 [[ 1  0  0  6]
 [ 0  3  0  4]
 [ 0  1  0  4]
 [ 1  1  0 29]]

Process finished with exit code 0



