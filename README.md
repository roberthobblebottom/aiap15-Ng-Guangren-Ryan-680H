# aiap15-Ng-Guangren-Ryan-680H  
  
Ng Guangren, Ryan  
ryan.ng@protonmail.com  
  
Use Github actions to install the requirements from `requirements.txt` and run `run.sh`.

# In `main.py``

Using configuration files so that if misconfigured the codes will be rombust enough to throw errors and give instruction on how to configure the file.

`sqlite3` Connection to extract the two tables into two dataframes. 
  
Afterward are feature engineering or feature cleaning that is explained in `eda.ipynb`

There is then the pipelines for `HistGradientBoostingClassifier` and `RandomForestClassifier`and the bayesian optimisations using `ax_platform`.

# In `PipelinesAndOptimisation.py` and the class

There are configured parameter scopes in a wrapper function for each type of classifier. These functions can be easily followed and used with other classifiers.

Then such parameter_scopes are pass to `bayesian_optimisation()` for the optimisation.  
`Bayesian optimisation` is used in hopes of finding the right hyper-parameters faster than `Grid Search`. However either of them are just heuristics where it is harder to choose between them if both have large amounts of hyperparameters.

Matrics are also saved after the analysis for analysis later

`ax_optimise()` is central to bayesian optimisation.`StratifiedKFold` is used as seen in `eda.ipynb`, the dataset is highly imbalanced.

`base()`


# Explaination of Algorithms choosen:

# Explaination of Metrics used:

`Accuracy` is used because the positive and negatives are very important.As some people would enjoy more if they are in `Luxuary`or `Deluxe`, they should not be misplaced due to low power or 