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
It added 108 new features


