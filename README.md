# aiap15-Ng-Guangren-Ryan-680H  
  
Ng Guangren, Ryan  
ryan.ng@protonmail.com  
  
Use Github actions to install the requirements from `requirements.txt` and run `run.sh`.

# In `main.py`

Using configuration files so that if misconfigured the codes will be rombust enough to throw errors and give instruction on how to configure the file.

`sqlite3` connections to extract the two tables into two dataframes. 
  
Afterward are feature engineering or feature cleaning that is explained in `eda.ipynb`

There is then the pipelines for `HistGradientBoostingClassifier`, `adaBoostClasifier` and `RandomForestClassifier` and the bayesian optimisations using `ax_platform`.

# In `PipelinesAndOptimisation.py` and the class

## Pipeline

Most of the `init` pipeline have been explained in `eda.ipynb`.
Both `StandardScaler` and `MinMaxScaler` are very sensitive to the presence of outliers and cannot guarantee balanced feature scales in the presence of outliers.
This dataset have enough outliers so carefulness is required about them.Thus `RobustScaler` is used instead

This is a imbalanced set so class weights are to balance all classifiers.
I did not use SMOTE from imblearn because it crashes visual studio code and python.
[And according to this user's take](https://datascience.stackexchange.com/a/52676), as imbalanced classes is still a large area of study and there may not be a superior method for theparticular case yet

## Optimisation

There are configured parameter scopes in a wrapper function for each type of classifier. These functions can be easily followed and used with other classifiers.

Then such parameter_scopes are pass to `bayesian_optimisation()` for the optimisation.  
`Bayesian optimisation` is used in hopes of finding the right hyper-parameters faster than `Grid Search`. However either of them are just heuristics where it is harder to choose between them if both have large amounts of hyperparameters.

Metrics are also saved after the analysis for analysis later with proper metrics names, classifiers names and datetime all in the name for easier distinction between them. `optimization_trace_single_method_plotly()` can plot the accuracy over iterations to see said trends.

`ax_optimise()` is central to bayesian optimisation.`StratifiedKFold` is used because as seen in a bar plot of labels in `eda.ipynb`, the dataset is highly imbalanced. `StratifiedKFold` helps to prevent data leakage.

`base()` does the fittin and predicting after hyperparameter tuning is completed. More metrics are captured at this point such as confusion table metirces


# Explaination of Algorithms choosen:

`Random Forest`,  `ada boost` and `Histogram Gradient Boosting Tree` are used and commonly used in analysis because ensembles helps to improve generalisability and rombustness over single estimator.

`Histogram Gradient Boosting Trees` is able to handle large amount of data better than `Gradient Boosting Trees` as it uses bins to reduce amount of number or to quantify the conintues values to a descrete set.

`ada boost` is a generalisation of `Gradient Boosting Tree`. `SVM` is not used as that is for binary classification while this dataset have 3 classes.

# Explaination of Metrics used:

Do we care about true positive more than both true conditions?

As explained about, the dataset is imbalanced, so `balance_accuracy_score` is used. It also because  true positive and true negatives are important. For example, some people would enjoy more if they are in `Luxuary`or `Deluxe` then in `standard` as the former two types will be more taylored to their preferences, they should not be misplaced.

`auc_roc_score` looks at actual conditions of True positive rate and False positive rate and doesn't require optimisation of a threshold for each label unklike f1.
  
These other metrics below will be remained for cross references or some other inquiries about positive centric statistics.
   
`f1`,`average_precision_score` or aka "area under the precision and recall curve" are more secondary in as these metrics looks at positive `f1`the weighted harmonic mean between pecision and recall. `average_precision_score` is used as it supports multiclass and multilables and calcuate the data in a one_vs_the_rest fashion and averages. These two metrics may be less useful.

final thoughs: balanced accuracy + AUCROC > f1, average_precision_score


# Selecting prefered pipeline

the pipeline objects after bayesian optimisation has the best parameters within. you may train it with `self.df_x` and `self.df_y` and then test it with `self.df_x2` and `df_y2`