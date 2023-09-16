import pandas as pd
import numpy as np
import sqlite3
import sys
import configparser
from datetime import datetime
from copy import deepcopy
from sklearn.impute import  SimpleImputer
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,label_binarize,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt
from ax import optimize
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import init_notebook_plotting, render

rng =0
def histogram_gradient_boosting_classifier_bayes_optimisation(df_x,df_y): 
    parameterScope = [
    {"name": "learning_rate", "type": "range", "value_type": "float",
     "bounds": [0.01,0.5], "is_ordered":False, },
    {"name": "l2_regularization", "type": "range", "value_type": "float",
     "bounds": [0.01,0.5], "is_ordered":True, },
    {"name": "max_leaf_nodes", "type": "range", "value_type": "int",
     "bounds": [ 5,  25], "is_ordered":True},
    {"name": "max_depth", "type": "range", "value_type": "int",
     "bounds": [3,  25], "is_ordered":True},
    {"name": "max_iter", "type": "range","log_scale":True, "value_type": "int",
     "bounds": [5,  25], "is_ordered":True},
    ]

    best_parameters_1, values_1, experiment_1, model_1 = optimize(
        parameters=parameterScope,
        evaluation_function=ax_optimise,
        objective_name='accuracy',
        total_trials=6,
        minimize=True,random_seed=rng
    )

    # Getting all the matrices out from pipeline 1
    datetime_now = datetime.now()
    objectives_1 = np.array([[trial.objective_mean for trial in experiment_1.trials.values()]])
    best_objective_plot = optimization_trace_single_method(y=objectives_1,
                                                            title="",
                                                              ylabel="mean absoulute error")
    fig = best_objective_plot    # TODO 
    matrices = str(best_parameters_1) +"\n"+str(values_1)
    kfold = StratifiedKFold(5,shuffle=True)
    for rest, pick in kfold.split(df_x,df_y):
        x_train = df_x.iloc[rest, :]
        y_train = df_y[rest]
        x_test = df_x.iloc[pick, :]
        y_test = df_y[pick]
        break
    hist_gradiant_boosting_classifer_pipeline_func(best_parameters_1, x_train, y_train, x_test,y_test,
                                               show_stats=True,matrices=matrices,datetime_now=datetime_now) 

    return model_1
def hist_gradiant_boosting_classifer_pipeline_func(parameters,x_train,y_train,
                                              x_test,y_test,show_stats,matrices = None,datetime_now=None):
    pipeline = deepcopy(base_pipeline)
    pipeline.steps.append(
                          ("histogram_gradient_boosting_classifier",HistGradientBoostingClassifier(
                              warm_start =False,
                              random_state = rng
                          )))
    pipeline.set_params(**{
        "histogram_gradient_boosting_classifier__learning_rate":parameters["learning_rate"],
        "histogram_gradient_boosting_classifier__max_depth":parameters["max_depth"],
        "histogram_gradient_boosting_classifier__max_leaf_nodes":parameters["max_leaf_nodes"],
        "histogram_gradient_boosting_classifier__l2_regularization":parameters["l2_regularization"],
        "histogram_gradient_boosting_classifier__max_iter":parameters["max_iter"],
    })# TODO something is wrong here
    return base("Histogram Gradient Boosting Classifier",pipeline,x_train,y_train,x_test,y_test,
                show_stats,matrices=matrices,datetime_now=datetime_now)

    

def base(title,estimator,x_train,y_train,x_test,y_test,show_stats,matrices=None,datetime_now=None):
    pipeline = estimator.fit(x_train, y=y_train)
    # print()
    # print(x_train.iloc[:10,:3])
    # print()
    y_predictions = pipeline.predict(x_test)
    if not show_stats:
        return y_predictions
    matrices = ""
    y_probabilities = pipeline.predict_proba(x_test)
    cm = confusion_matrix(y_test, y_predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=estimator.classes_)
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)
    plt.title(title)
    # display.figure__.savefig("classifiers_performances/"+title+" "+datetime_now)  # TODO: does this line and the previous 3 lines work together correctly? 
    
    matrices += "\n" + classification_report(y_test, y_predictions)
    matrices += "\n AUROC score:"+str(roc_auc_score(y_test,    
          y_probabilities, multi_class='ovo')) 
    with open("classifiers_performances/"+title+"_"+str(datetime_now),'w') as f:
        for line in matrices:
            f.write(line)
            # f.write('\n')
    return y_predictions
    
def ax_optimise(parameters):
    accuracies =[]    
    kfold = StratifiedKFold(5,shuffle=True)
    for rest, pick in kfold.split(df_x,df_y):
        x_train = df_x.iloc[rest, :]
        y_train = df_y[rest]
        x_test = df_x.iloc[pick, :]
        y_test = df_y[pick]
        # y_test_binarized = label_binarize(y_test, classes=[
        #                                             'functional', 'functional needs repair', 'non functional'])
        predictions = train_predict_pipeline(parameters, x_train, y_train, x_test,y_test, False)
        # mae =  mean_absolute_error(y_test_binarized,
                                # label_binarize(predictions, classes=['functional', 'functional needs repair', 'non functional'])
                                # )
        print()
        print(y_test)
        print(predictions)
        accuracy = accuracy_score(y_test,predictions)
        accuracies.append(accuracy)       
    print(accuracies)
    sum = 0
    for accuracy in accuracies:
        sum += accuracy 
    avg = sum/len(accuracies)
    print("average accuracy: "+format(avg))
    return avg




if __name__ == "__main__":
    # config and check correctness of config
    config_file_name = sys.argv[1]
    pre_post_db_text = "pre_post_db"
    list_of_keys = ["pre_purchase_survey_db_name","post_purchase_survey_db_name",
                    "pre_purchase_survey_table_name","post_purchase_survey_table_name"]

    config = configparser.ConfigParser()
    read_file = config.read(config_file_name)
    if len(read_file) == 0:
        raise Exception(
            config_file_name+''' configuration file is not found, 
            please create said file or use the original file''')
    if pre_post_db_text not in config.sections():
        raise Exception(
            "[pre_post_db] section is not in "+config_file_name+
            " create a section for it in the config ")
    if not set(list_of_keys).issubset(set(config[pre_post_db_text].keys())):
        raise Exception(
            "some of the keys from "+ list_of_keys+
            " are not found in pre_post_db section")

    #connection
    config = config[pre_post_db_text]
    connection_post = sqlite3.connect(config["post_purchase_survey_db_name"])
    connection_pre = sqlite3.connect(config["pre_purchase_survey_db_name"])
    df_pre = pd.read_sql_query("SELECT * FROM "+config["pre_purchase_survey_table_name"],
                               connection_pre,index_col="index")
    df_post = pd.read_sql_query("SELECT * FROM "+config["post_purchase_survey_table_name"],
                                connection_post,index_col="index")
    connection_post.close()
    connection_pre.close()
        

    passanger_code_text = "internal_passanger_code"

    #renaming
    df_pre.rename({"Ext_Intcode":passanger_code_text},axis=1,inplace=True)
    df_post.rename({"Ext_Intcode":passanger_code_text},axis=1,inplace=True)
    def replace_spaces(x):
        return x.replace(" ","_").strip().lower()
    df_pre.rename(replace_spaces,axis=1,inplace=True)
    df_post.rename(replace_spaces,axis=1,inplace=True)

    #splitting dupes = duplications 
    df_pre_2_dupes_indexes = df_pre.groupby(passanger_code_text).size()[lambda x: x>1].index
    df_post_2_dupes_indexes = df_post.groupby(passanger_code_text).size()[lambda x: x>1].index
    df_pre_2_dupes = df_pre[lambda x: x.internal_passanger_code.isin(df_pre_2_dupes_indexes)]
    df_post_2_dupes = df_post[lambda x: x.internal_passanger_code.isin(df_post_2_dupes_indexes)]\
        .drop([passanger_code_text],axis=1)
    df_2_dupes = pd.concat([df_pre_2_dupes,df_post_2_dupes],axis=1)

    df_split_1 = df_2_dupes.drop_duplicates(passanger_code_text,keep='first',ignore_index=True)
    df_split_2 = df_2_dupes.drop_duplicates(passanger_code_text,keep='last',ignore_index=True)

    #feature engineering to get non None ticket_typefor df_split_2
    comparison =df_split_1.compare(df_split_2)
    comparison.ticket_type = comparison.ticket_type.fillna("")
    df_split_2.loc[:,"ticket_type"] = comparison.ticket_type.self + comparison.ticket_type.other

    #Single coded rows 
    select_single_coded = lambda x: ~x.internal_passanger_code.isin(df_split_2.internal_passanger_code)
    post_single_code= df_post[select_single_coded].drop(passanger_code_text,axis=True)
    pre_single_code= df_pre[select_single_coded]
    df_single_code = pd.concat([pre_single_code,post_single_code],axis=1)  

    #combining the dfs
    df = pd.concat([df_split_2,df_single_code],ignore_index=True)
    
    #
    df_with_y_nans = df[df.ticket_type.isna() | df.ticket_type == ""]
    
    # Splitting into x, y
    df = df[~df.ticket_type.isna() & ~(df.ticket_type == "")]
    df_x = df.drop(["ticket_type"],axis =1).set_index(passanger_code_text,drop=True)
    df_y = df.ticket_type 

    # cruise_distance feature engineering
    df_x.cruise_distance = df_x.cruise_distance.str.split(" ").str[0]

    #feature engineering of `date-of_birth` to `age` of type int
    df_x["temp_1"] = pd.to_datetime(df_x.date_of_birth,format="%Y-%m-%d",errors='coerce')
    df_x["temp_2"] = pd.to_datetime(df_x.date_of_birth,format="%d/%m/%Y",errors='coerce')
    df_x["date_of_birth"] = df_x.temp_1.fillna(df_x.temp_2)
    df_x.drop(["temp_1","temp_2"],axis=1,inplace=True)

    def age(born):
        if str(born) == "NaT": 
            return born
        born = datetime.strptime(str(born), "%Y-%m-%d %H:%M:%S").date()
        today = datetime.today()
        return today.year - born.year - ((today.month, 
                                        today.day) < (born.month, 
                                                        born.day))
    df_x["age"] = df_x.date_of_birth.apply(age).fillna(0).astype(str).astype(int)
    df_x.drop("date_of_birth",axis=1,inplace=True)

    #Logging from str to int
    df_x["logging"] = pd.to_datetime(df_x.logging).astype(int)

    #imputation of wifi and entertainemnt
    df_x[["wifi","entertainment"]]=df_x[["wifi","entertainment"]].fillna(-1) 

    label_encoder = LabelEncoder()
    df_y = label_encoder.fit_transform(df_y)

    ordinal_encoder_categories = [ 'onboard_wifi_service',
        'embarkation/disembarkation_time_convenient', 'ease_of_online_booking',
        'gate_location', 'onboard_dining_service', 'online_check-in',
        'cabin_comfort', 'onboard_entertainment', 'cabin_service',
        'baggage_handling', 'port_check-in_service', 'onboard_service',
        'cleanliness']
    one_hot_encoder_categories = ["gender","source_of_traffic","cruise_name"]

    #pipeline operations
    base_pipeline = Pipeline([("column_transformer",
                               ColumnTransformer(
                               [
                                #    ('wifi_entertainment_simple_imputer',SimpleImputer(missing_values=-1,strategy='constant'),["wifi","entertainment"]), 
                                # SimpleImputer does not impute Nan natively, I have to do it outside, I don't think there will be much leakage
                                    ('ordinal_encoder',
                                     OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan),
                                     ordinal_encoder_categories),
                                    ('one_hot_encoder',OneHotEncoder(sparse=False),one_hot_encoder_categories),
                                ], remainder='passthrough',
                               )
                               ),
                         ("simple_imputer",SimpleImputer(strategy="median")),
                         ("select_percentile",SelectPercentile(percentile=50)),
                         ])
    
    train_predict_pipeline = hist_gradiant_boosting_classifer_pipeline_func
    
    model_1 = histogram_gradient_boosting_classifier_bayes_optimisation(df_x,df_y)
    