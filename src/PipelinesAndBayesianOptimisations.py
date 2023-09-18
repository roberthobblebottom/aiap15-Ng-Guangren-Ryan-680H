
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import  SimpleImputer
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from ax import optimize
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method_plotly
from ax.utils.notebook.plotting import init_notebook_plotting, render
rng = 0
class PipelinesAndOptimisations:
    rng =0
    # train_predict_pipeline,df_x,df_y,classes=None,None,None,None

    def __init__(self,df_x,df_y,classes,classifier,params_to_be_set):
        self.df_x = df_x
        self.df_y = df_y
        self.x_train,self.y_train,self.x_test,self.y_test = \
            pd.DataFrame(),pd.DataFrame(),pd.DataFrame,pd.DataFrame()
        self.classes = classes
        ordinal_encoder_categories = [ 'onboard_wifi_service',
        'embarkation/disembarkation_time_convenient', 'ease_of_online_booking',
        'gate_location', 'onboard_dining_service', 'online_check-in',
        'cabin_comfort', 'onboard_entertainment', 'cabin_service',
        'baggage_handling', 'port_check-in_service', 'onboard_service',
        'cleanliness']
        one_hot_encoder_categories = ["gender","source_of_traffic","cruise_name"]

        self.pipeline = Pipeline([("column_transformer",
                               ColumnTransformer(
                               [
                                #('wifi_entertainment_simple_imputer',SimpleImputer(missing_values=-1,strategy='constant'),["wifi","entertainment"]), 
                                # SimpleImputer does not impute Nan natively, I have to do it outside, I don't think there will be much leakage
                                    ('ordinal_encoder',
                                     OrdinalEncoder(handle_unknown='use_encoded_value',
                                                    unknown_value=np.nan),
                                     ordinal_encoder_categories),
                                    ('one_hot_encoder',OneHotEncoder(sparse_output=False),
                                     one_hot_encoder_categories),
                                ], remainder='passthrough', verbose_feature_names_out=False
                               )),
                            ("robust_scaler",RobustScaler()),
                            ("simple_imputer",SimpleImputer(strategy="median")),
                            ("select_percentile",SelectPercentile(percentile=51)),
                        #  ("smoten",SMOTEN(random_state=rng)), # crashes my computer and may not be most useful, class weights are better?
                        #  ("standard_scaler",StandardScaler()),
                         ])
        
                        # TODO reduce this
                        # StandardScaler removes the mean and scales the data to unit variance.  
                        # The scaling shrinks the range of the feature values as shown in the left figure below.  
                        # However, the outliers have an influence when computing the empirical mean and standard deviation.  
                        # Note in particular that because the outliers on each feature have different magnitudes,  
                        # the spread of the transformed data on each feature is very different:  
                        # most of the data lie in the [-2, 4] range for the transformed median income feature 
                        # while the same data is squeezed in the smaller [-0.2, 0.2] range for the transformed average house occupancy.
                        # StandardScaler therefore cannot guarantee balanced feature scales in the presence of outliers.       
                        # Both StandardScaler and MinMaxScaler are very sensitive to the presence of outliers.     
                        # TODO should I use RobustScaler?     
                        # There are outliers but few enoght
        
        self.pipeline.steps.append(classifier)
        self.params_to_be_set = params_to_be_set
        self.matrices = ""
        self.datetime_now = datetime.now()

    def hist_gbc_bo(self):
        name = "Histogram Gradient Boosting Classifier"
        parameters = [
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
        self.bayesian_optimization(parameters=parameters,name=name)
    def rf_bo(self):
        name = "Random Forest Classifier"
        parameters=[{"name": "max_depth", 
            "type": "range", 
            "value_type": "int",
            "log_scale":True,
            "bounds": [1,200],"is_ordered":True},

            {"name":"criterion",
            "type":"choice",
            "value_type":"str",
            "values":["gini","entropy"],#No "log_loss"
            "is_ordered":True},

            {"name":"max_features",
            "type":"choice",
            "value_type":"str",
            "values":["sqrt","log2",None],
            "is_ordered":True},
        ]
        self.bayesian_optimization(parameters=parameters,name=name)

    def bayesian_optimization(self,parameters,name):
        best_parameters, values, experiment, model = optimize(
            parameters=parameters,
            evaluation_function=self.ax_optimise,
            objective_name='accuracy',
            total_trials=1,# TODO set it to a higher number later
            minimize=True,random_seed=rng
        )
        
        objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
        optimization_trace_single_method_plotly(y=objectives, title="",ylabel="accuracy")\
                    .write_image("classifiers_performances/optimization_trace_single_method_ploty_"+\
                                name+"_"+str(self.datetime_now)+".png")
        self.matrices = str(best_parameters) +"\n"+str(values)
        # kfold = StratifiedKFold(5,shuffle=True)
        # for rest, pick in kfold.split(self.df_x,self.df_y):
        #     self.x_train = self.df_x.iloc[rest, :]
        #     self.y_train = self.df_y[rest]
        #     self.x_test = self.df_x.iloc[pick, :]
        #     self.y_test = self.df_y[pick]
        #     break
        self.paramters = best_parameters
        self.base(name,show_stats=True) 
        return model
    
    def base(self,name,show_stats):
        print("pipeline fitting")
        pipeline = self.pipeline.fit(self.x_train, y=self.y_train)
        y_predictions = pipeline.predict(self.x_test)
        if not show_stats:
            return y_predictions
        print("Showing more stats")
        y_probabilities = pipeline.predict_proba(self.x_test)
        cm = confusion_matrix(self.y_test, y_predictions)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes).plot()
        plt.xticks(rotation=30)
        plt.yticks(rotation=30)
        plt.title(name)
        plt.savefig("classifiers_performances/confusion_matrix_"+name+
                    "_"+str(self.datetime_now),format="png")
        
        self.matrices += "\n" + classification_report(self.y_test, y_predictions)
        self.matrices += "\n AUROC score:"+str(roc_auc_score(self.y_test,    
            y_probabilities, multi_class='ovo')) 
        with open("classifiers_performances/"+name+"_"+str(self.datetime_now),'w') as f:
            for line in self.matrices:
                f.write(line)
                # f.write('\n')
        return y_predictions
        
    def ax_optimise(self,parameters):
        self.paramters = parameters
        accuracies =[]    
        kfold = StratifiedKFold(5) 
        # Useful for hugely imbalanced class count and arbitrary data order. No need shuffle
        classifier_name = self.params_to_be_set[0].split("__")[0]
        params = {param: parameters[param.replace(classifier_name+"__","")] for param in self.params_to_be_set}
        self.pipeline.set_params(**params)
        for rest, pick in kfold.split(self.df_x,self.df_y):
            self.x_train = self.df_x.iloc[rest, :]
            self.y_train = self.df_y[rest]
            self.x_test = self.df_x.iloc[pick, :]
            self.y_test = self.df_y[pick]
            predictions = self.base("",False)
            # print()
            # print(y_test)
            # print(predictions)
            accuracy = accuracy_score(self.y_test,predictions)
            accuracies.append(accuracy)       
        print(accuracies)
        sum = 0
        for accuracy in accuracies:
            sum += accuracy 
        avg = sum/len(accuracies)
        print("average accuracy: "+format(avg))
        return avg

