# ML Assigment Process Description
The assignment was done on my own as I did not find any groups in time.
The following describes how the process was done, from exploring the data to the implementation
of a model along with justifications.

## Data exploration
First, I explored the data by looking at the attributes and doing research about the dataset. I did not find any description
of the attributes on Kaggle, so I had to search for it seperately. Once I had access to the descriptions, I tried to get
an idea of what features would be useful for the project by looking at correlation coefficients for each attribute with
the sale price and looking at scatter plots.

## Data preprocessing
Preprocessing was necessary to allow the machine learning models to be as effective as possible. Alot of the data was
either categorical, had scewed distributions or had large variety in scale.

### Dropping irrelevant features
Both the "Order" and "PID" features seemed irrelevant for predicting housing prices, and were therefore dropped from the
dataset. The "Pool Area" feature had a large zero-inflated distribution, and was also dropped for that reason. I did not
know enough about real estate to drop any other housing related feature from the dataset.

### Imputing missing values
As most machine learning algorithms cannot work with missing values, I had to take care of the missing values in the 
dataset. This was done seperately for numerical and categorical features: numerical missing values were replaced by
the median for that feature, while the categorical features were replaced by the constant "No". 

### Encoding categorical features
As most machine learning algorithms can only work with numbers, categorical features had to be encoded to numbers.
This was done differently for different features: those with a natural order of categories were encoded in an ordinal
manner, while the rest was encoded using "one hot" encoding.

### Transforming scewed distributions
By looking at the histogram for the numerical features, I found some of them to be scewed, having long tails. This is
usually unwanted for the algorithms, and was therefore handled by computing either the logarithm or square root of the 
features. The resulting distributions were much more symmetrical.

### Standardizing
The numerical values in the dataset varied alot in scale. By standardizing them, the algorithm was less prone to be biased
towards features with greater scale. Standardization is relatively little affected by outliers, which made it a natural
choise over min-max scaling.

## Implementing machine learning algorithms
Once the preprocessing was done, the data was ready to be fed to a machine learning algorithm. Three algorithms were
tested: __linear regression__, __polynomial regression__ and __ensemble regression__. RMSE was chosen as the performance
measure, which is a cost function (lower values are better).

### Linear regression
The linear regression algorithm performed decent on the training data, but had a huge cost during cross-validation, which
indicated that it was overfitting to a big extent.

### Polynomial regression
The polynomial regression algorithm performed even better at the training data, while having a much higher cost during
cross-validation, which also indicated overfitting.

### Ensemble regression
The ensemble regression algorithm (random forest in particular) had a much smaller ratio between the cost for the training
data and cross-validation, indicating a less severe overfitting the the others. It also gave the best (lowest) performance
measure on during cross-validation, which made it a good choice to go with. 

