import pandas as pd
import sqlite3
import sys
import configparser
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder,RobustScaler,label_binarize
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
# from imblearn.over_sampling import SMOTEN

from sklearn.linear_model import SGDClassifier
from PipelinesAndBayesianOptimisations import PipelinesAndOptimisations

rng =0
classes = []
b=None

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
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    df_x["age"] = df_x.date_of_birth.apply(age).fillna(0).astype(str).astype(int)
    df_x.drop("date_of_birth",axis=1,inplace=True)

    #Logging from str to int
    df_x["logging"] = pd.to_datetime(df_x.logging).astype(int)

    #imputation of wifi and entertainemnt
    df_x[["wifi","entertainment"]]=df_x[["wifi","entertainment"]].fillna(-1) 

    label_encoder = LabelEncoder()
    df_y = label_encoder.fit_transform(df_y,)
    classes = label_encoder.classes_
    print("Histogram Gradient Boosting Classifier Bayesian Optimisation")
    print("Histogram Gradient Boosting Classifier Bayesian Optimisation")
    print("Histogram Gradient Boosting Classifier Bayesian Optimisation")
    params = [
                "histogram_gradient_boosting_classifier__learning_rate",
                "histogram_gradient_boosting_classifier__max_depth",
                "histogram_gradient_boosting_classifier__max_leaf_nodes",
                "histogram_gradient_boosting_classifier__l2_regularization",
                "histogram_gradient_boosting_classifier__max_iter"
    ]
    b = PipelinesAndOptimisations(df_x,df_y,classes,
             classifier=("histogram_gradient_boosting_classifier",HistGradientBoostingClassifier(
                              warm_start =False,random_state = rng,
                              class_weight = 'balanced'
                        )),
            params_to_be_set=params
            )
    b.hist_gbc_bo()

    print("Random Forest Classifier Bayesian Optimisation")
    print("Random Forest Classifier Bayesian Optimisation")
    print("Random Forest Classifier Bayesian Optimisation")
    params = [
           'random_forest_classifier__max_depth',
            'random_forest_classifier__criterion',
            'random_forest_classifier__max_features' 
    ]
    b = PipelinesAndOptimisations(df_x,df_y,classes,
             classifier=("random_forest_classifier",RandomForestClassifier(
                    random_state=rng,
                    n_jobs =5,
                    oob_score=True, 
                    n_estimators = 500,
                    class_weight="balanced"
             )), params_to_be_set=params)
    b.rf_bo()
    
    