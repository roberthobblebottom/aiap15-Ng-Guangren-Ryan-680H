import pandas as pd
import numpy as np
import sqlite3
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import datetime,timedelta
import sys

if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config_file_name ="src/pipeline.ini" 
    read_file = config.read(config_file_name)
    if len(read_file) == 0:
        raise Exception(config_file_name+" configuration file is not found, please create said file or use the original file")
    if "post_pre_db" not in config.sections():
        raise Exception("[post_pre_db] section is not in "+config_file_name+" create a section for it in the config ")
    list_of_keys = ["pre_purchase_survey_db_name","post_purchase_survey_db_name","pre_purchase_survey_table_name","post_purchase_survey_table_name"]
    if not set(list_of_keys).issubset(set(config["post_pre_db"].keys())):
        raise Exception("some of the keys "+ list_of_keys+" is not in post_pre_db section")
    config = config['post_pre_db']
    connection_post = sqlite3.connect(config["post_purchase_survey_db_name"])
    connection_pre = sqlite3.connect(config["pre_purchase_survey_db_name"])
    df_pre = pd.read_sql_query("SELECT * FROM "+config["pre_purchase_survey_table_name"],connection_pre,index_col="index")
    df_post = pd.read_sql_query("SELECT * FROM "+config["post_purchase_survey_table_name"],connection_post,index_col="index")
    connection_post.close()
    connection_pre.close()