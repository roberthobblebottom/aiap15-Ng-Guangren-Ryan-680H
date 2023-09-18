params1 = [
            "histogram_gradient_boosting_classifier__learning_rate",
            "histogram_gradient_boosting_classifier__max_depth",
            "histogram_gradient_boosting_classifier__max_leaf_nodes",
            "histogram_gradient_boosting_classifier__l2_regularization",
            "histogram_gradient_boosting_classifier__max_iter"
]
parameters = {
        }
classifier_name = params1[0].split("__")[0]
params = {param: parameters[param.replace(classifier_name,"")] for param in params1}
print(params)
