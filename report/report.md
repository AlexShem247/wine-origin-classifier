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

