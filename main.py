from preprocessing import DataPreprocessor
from utils import *
from sklearn.model_selection import train_test_split
from model_training import  *
from evaluation import evaluate_model

filename = 'diabetes_prediction_dataset.csv'

# load up the data into the pre processor
pre_processor = DataPreprocessor(filename)
pre_processor.load_dataset()

#check to see what the data looks like initially:
pre_processor.head()
"""
HEAD
-------------------
   gender   age  hypertension  heart_disease smoking_history    bmi  HbA1c_level  blood_glucose_level  diabetes
0  Female  80.0             0              1           never  25.19          6.6                  140         0
1  Female  54.0             0              0         No Info  27.32          6.6                   80         0
2    Male  28.0             0              0           never  27.32          5.7                  158         0
3  Female  36.0             0              0         current  23.45          5.0                  155         0
4    Male  76.0             1              1         current  20.14          4.8                  155         0
"""

# looking at numerican properties of the dataset
pre_processor.describe()
"""
                 age  hypertension  heart_disease            bmi    HbA1c_level  blood_glucose_level       diabetes
count  100000.000000  100000.00000  100000.000000  100000.000000  100000.000000        100000.000000  100000.000000
mean       41.885856       0.07485       0.039420      27.320767       5.527507           138.058060       0.085000
std        22.516840       0.26315       0.194593       6.636783       1.070672            40.708136       0.278883
min         0.080000       0.00000       0.000000      10.010000       3.500000            80.000000       0.000000
25%        24.000000       0.00000       0.000000      23.630000       4.800000           100.000000       0.000000
50%        43.000000       0.00000       0.000000      27.320000       5.800000           140.000000       0.000000
75%        60.000000       0.00000       0.000000      29.580000       6.200000           159.000000       0.000000
max        80.000000       1.00000       1.000000      95.690000       9.000000           300.000000       1.000000

# min age of 0.08 might outlier.
# sample is highly imbalanced. 8.5% of the sample has diabetes while 91.5% does not have diabetes. This will need to be addressed later.

"""

# See data type and if there are any nulls in the data
pre_processor.info()
"""
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 9 columns):
 #   Column               Non-Null Count   Dtype
---  ------               --------------   -----
 0   gender               100000 non-null  object
 1   age                  100000 non-null  float64
 2   hypertension         100000 non-null  int64
 3   heart_disease        100000 non-null  int64
 4   smoking_history      100000 non-null  object
 5   bmi                  100000 non-null  float64
 6   HbA1c_level          100000 non-null  float64
 7   blood_glucose_level  100000 non-null  int64
 8   diabetes             100000 non-null  int64

 # no column is has null
 # types for each column might be necessary later
"""

# check the shape of the dataset
pre_processor.shape()
"""
Dataset contains 100000 rows and 9 columns.
"""

#encode categorical features (gender, smoking_history)
pre_processor.encode_categorical()
"""
ENCODED HEAD:
-------------------
HEAD
-------------------
   gender   age  hypertension  heart_disease  smoking_history    bmi  HbA1c_level  blood_glucose_level  diabetes
0       0  80.0             0              1                4  25.19          6.6                  140         0
1       0  54.0             0              0                0  27.32          6.6                   80         0
2       1  28.0             0              0                4  27.32          5.7                  158         0
3       0  36.0             0              0                1  23.45          5.0                  155         0
4       1  76.0             1              1                1  20.14          4.8                  155         0
"""

# show graph of class distrubution
#pre_processor.class_distribution('diabetes')
"""
A graph opens
"""

# show histograms to identify outliers
#pre_processor.histogram_outliers()
"""
A graph opens
"""

# show box-plots to identify outliers
#pre_processor.boxplot_outliers()
"""
A graph opens
"""

# show a correlation analysis
#pre_processor.correlation_analysis()
"""
A graph opens
"""




# TODO choose a standarization model. This will depend on the model we train
# Standardization VS Normalization
# Standardize numerical features
pre_processor.standardize()
"""
Continuous numerical features standardized.
   gender       age  hypertension  heart_disease  smoking_history       bmi  HbA1c_level  blood_glucose_level  diabetes
0       0  1.692704             0              1                4 -0.321056     1.001706             0.047704         0
1       0  0.538006             0              0                0 -0.000116     1.001706            -1.426210         0
2       1 -0.616691             0              0                4 -0.000116     0.161108             0.489878         0
3       0 -0.261399             0              0                1 -0.583232    -0.492690             0.416183         0
4       1  1.515058             1              1                1 -1.081970    -0.679490             0.416183         0
-------------------
"""








# retrieve DF
df = pre_processor.getDf()



#examine df
printLine()
print("Showing dataframe after pre_processor...")
print("HEAD")
printLine()
print(df.head())
"""
HEAD
-------------------
   gender   age  hypertension  heart_disease  smoking_history    bmi  HbA1c_level  blood_glucose_level  diabetes
0       0  80.0             0              1                4  25.19          6.6                  140         0
1       0  54.0             0              0                0  27.32          6.6                   80         0
2       1  28.0             0              0                4  27.32          5.7                  158         0
3       0  36.0             0              0                1  23.45          5.0                  155         0
4       1  76.0             1              1                1  20.14          4.8                  155         0
"""


"""
Things to answer:
    1. Are there missing values? Which columns are affected?
    2. Do numerical features have outliers? Are they normally distributed or skewed?
    3. Are there correlations between features or with the target variable?
    4. Is the target variable balanced, or is there a class imbalance?
"""



# Split the dataset into training and testing sets
X = df.drop(columns=['diabetes'])
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

printLine()
print("Dataset split into training and testing sets.")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
"""
Dataset split into training and testing sets.
Training set: 80000 samples
Testing set: 20000 samples
"""




# Train models
logistic_regression_model = train_logistic_regression(X_train, y_train)
decision_tree_model = train_decision_tree(X_train, y_train)
random_forest_model = train_random_forest(X_train, y_train)

"""
# Evaluate models
printLine()
print("Logistic Regression Model Evaluation")
printLine()
lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluate_model(logistic_regression_model, X_test, y_test)
print(f"Accuracy: {lr_accuracy}")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")
print(f"F1 Score: {lr_f1}")
"""
"""
-------------------
Logistic Regression Model Evaluation
-------------------
Accuracy: 0.95875
Precision: 0.8645747316267548
Recall: 0.6129976580796253
F1 Score: 0.7173689619732785
"""
"""
printLine()
print("Decision Tree Model Evaluation")
printLine()
dt_accuracy, dt_precision, dt_recall, dt_f1 = evaluate_model(decision_tree_model, X_test, y_test)
print(f"Accuracy: {dt_accuracy}")
print(f"Precision: {dt_precision}")
print(f"Recall: {dt_recall}")
print(f"F1 Score: {dt_f1}")
"""
"""
-------------------
Decision Tree Model Evaluation
-------------------
Accuracy: 0.9519
Precision: 0.7090807174887892
Recall: 0.740632318501171
F1 Score: 0.7245131729667812
"""
"""
printLine()
print("Random Forest Model Evaluation")
printLine()
rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(random_forest_model, X_test, y_test)
print(f"Accuracy: {rf_accuracy}")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")
print(f"F1 Score: {rf_f1}")
"""
"""
-------------------
Random Forest Model Evaluation
-------------------
Accuracy: 0.9703
Precision: 0.9484702093397746
Recall: 0.689695550351288
F1 Score: 0.7986440677966101
"""

# Hyperparameter tuning
"""
printLine()
print("Hyperparameter Tuning for Logistic Regression")
printLine()
hyper_logistic_regression_model = hyper_train_logistic_regression(X_train, y_train)
hlr_accuracy, hlr_precision, hlr_recall, hlr_f1 = evaluate_model(hyper_logistic_regression_model, X_test, y_test)
print(f"Accuracy: {hlr_accuracy}")
print(f"Precision: {hlr_precision}")
print(f"Recall: {hlr_recall}")
print(f"F1 Score: {hlr_f1}")
"""
"""
-------------------
Hyperparameter Tuning for Logistic Regression
-------------------
Best parameters for Logistic Regression: {'C': 0.1, 'solver': 'lbfgs'}
Accuracy: 0.9589
Precision: 0.8685524126455907
Recall: 0.6112412177985949
F1 Score: 0.7175257731958763
"""
"""
printLine()
print("Hyperparameter Tuning for Decision Tree")
printLine()
hyper_descision_tree_model = hyper_train_decision_tree(X_train, y_train)
hdt_accuracy, hdt_precision, hdt_recall, hdt_f1 = evaluate_model(hyper_descision_tree_model, X_test, y_test)
print(f"Accuracy: {hdt_accuracy}")
print(f"Precision: {hdt_precision}")
print(f"Recall: {hdt_recall}")
print(f"F1 Score: {hdt_f1}")
"""
"""
-------------------
Hyperparameter Tuning for Decision Tree
-------------------
Best parameters for Decision Tree: {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 3, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Accuracy: 0.97215
Precision: 1.0
Recall: 0.6738875878220141
F1 Score: 0.8051766351871283
"""


"""
This takes a long time, commenting it out for now
"""
"""
printLine()
print("Hyperparameter Tuning for Random Forest")
printLine()
hyper_random_forest_model = hyper_train_random_forest(X_train, y_train)
hrf_accuracy, hrf_precision, hrf_recall, hrf_f1 = evaluate_model(hyper_random_forest_model, X_test, y_test)
print(f"Accuracy: {hrf_accuracy}")
print(f"Precision: {hrf_precision}")
print(f"Recall: {hrf_recall}")
print(f"F1 Score: {hrf_f1}")
"""

"""
Hyperparameter Tuning for Random Forest
-------------------
Best parameters for Random Forest: {'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
Accuracy: 0.9722
Precision: 1.0
Recall: 0.6744730679156908
F1 Score: 0.8055944055944056
"""
"""
# feature selection
printLine()
print("Feature Selection for Logistic Regression")
printLine()
rfe_logistic_regression_model, selected_features, X_test_selected = feature_selection_rfe_logistic(X_train, y_train, X_test)
rfe_lr_accuracy, rfe_lr_precision, rfe_lr_recall, rfe_lr_f1 = evaluate_model(rfe_logistic_regression_model, X_test_selected, y_test)
print(f"Accuracy: {rfe_lr_accuracy}")
print(f"Precision: {rfe_lr_precision}")
print(f"Recall: {rfe_lr_recall}")
print(f"F1 Score: {rfe_lr_f1}")
"""
"""
-------------------
Feature Selection for Logistic Regression
-------------------
Selected Features by RFE: ['age', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level']
Accuracy: 0.95665
Precision: 0.8584825234441603
Recall: 0.5895784543325527
F1 Score: 0.6990628254078445
"""

printLine()
print("Feature Selection For Random Forest")
printLine()

# Perform feature selection with Random Forest
rf_model, selected_features_rf, X_test_selected_rf = feature_selection_random_forest(X_train, y_train, X_test, n_features=5)

# Evaluate the model
rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(rf_model, X_test_selected_rf, y_test)
print(f"Accuracy: {rf_accuracy}")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")
print(f"F1 Score: {rf_f1}")


printLine()
print("Feature Selection for Decision Tree")
printLine()

# Perform feature selection with Decision Tree
dt_model, selected_features, X_test_selected = feature_selection_decision_tree(X_train, y_train, X_test, n_features=5)
# Evaluate the model
dt_accuracy, dt_precision, dt_recall, dt_f1 = evaluate_model(dt_model, X_test_selected, y_test)
print(f"Accuracy: {dt_accuracy}")
print(f"Precision: {dt_precision}")
print(f"Recall: {dt_recall}")
print(f"F1 Score: {dt_f1}")
"""

-------------------
Decision Tree Model Evaluation
-------------------
Accuracy: 0.9519
Precision: 0.7090807174887892
Recall: 0.740632318501171
F1 Score: 0.7245131729667812


-------------------
Feature Selection for Decision Tree
-------------------
Selected Features by Decision Tree: ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'smoking_history']
Accuracy: 0.95275
Precision: 0.7168845935190449
Recall: 0.7382903981264637
F1 Score: 0.7274300548024228


-------------------
Feature Selection for Decision Tree
-------------------
Selected Features by Decision Tree: ['HbA1c_level', 'blood_glucose_level']
Accuracy: 0.97215
Precision: 1.0
Recall: 0.6738875878220141
F1 Score: 0.8051766351871283

-------------------
Hyperparameter Tuning for Decision Tree
-------------------
Best parameters for Decision Tree: {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 3, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Accuracy: 0.97215
Precision: 1.0
Recall: 0.6738875878220141
F1 Score: 0.8051766351871283


Hyperparameter Tuning for Random Forest
-------------------
Best parameters for Random Forest: {'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
Accuracy: 0.9722
Precision: 1.0
Recall: 0.6744730679156908
F1 Score: 0.8055944055944056

-------------------
Feature Selection For Random Forest
-------------------
Selected Features by Random Forest: ['HbA1c_level', 'blood_glucose_level']
Accuracy: 0.97215
Precision: 1.0
Recall: 0.6738875878220141
F1 Score: 0.8051766351871283

-------------------
Feature Selection + Threshold Tuning for Decision Tree
-------------------
Selected Features by Decision Tree: ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'smoking_history']
Threshold: 0.6
Accuracy: 0.95275
Precision: 0.7168845935190449
Recall: 0.7382903981264637
F1 Score: 0.7274300548024228

-------------------
Feature Selection + Threshold Tuning for Decision Tree
-------------------
Selected Features by Decision Tree: ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'smoking_history']
Threshold: 0.06
Accuracy: 0.949
Precision: 0.6865509761388287
Recall: 0.7412177985948478
F1 Score: 0.7128378378378378

-------------------
Feature Selection + Threshold Tuning for Decision Tree
-------------------
Selected Features by Decision Tree: ['HbA1c_level', 'blood_glucose_level']
Threshold: 0.06
Accuracy: 0.6977
Precision: 0.21550367261280168
Recall: 0.961943793911007
F1 Score: 0.35212173167595373
"""
"""
printLine()
print("Feature Selection + Hyperparameter Tuning for Decision Tree")
printLine()

"""
"""
# Combine feature selection and hyperparameter tuning
combined_decision_tree_model, X_test_selected = combined_train_decision_tree(X_train, y_train, X_test, n_features=5)

# Evaluate the model
cdt_accuracy, cdt_precision, cdt_recall, cdt_f1 = evaluate_model(combined_decision_tree_model, X_test_selected, y_test)
print(f"Accuracy: {cdt_accuracy}")
print(f"Precision: {cdt_precision}")
print(f"Recall: {cdt_recall}")
print(f"F1 Score: {cdt_f1}")
"""
"""
-------------------
Feature Selection + Hyperparameter Tuning for Decision Tree
-------------------
-------------------
Feature Selection
-------------------
Selected Features: ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'smoking_history']
-------------------
Hyperparameter Tuning
-------------------
Best Parameters: {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2}
Accuracy: 0.97185
Precision: 0.9897348160821214
Recall: 0.677400468384075
F1 Score: 0.8043100451859576
"""


plot_confusion_matrix(dt_model, X_test_selected, y_test)
plot_roc_curve(dt_model, X_test_selected, y_test)
feature_results = evaluate_n_features(X_train, y_train, X_test, y_test, max_features=9)
plot_feature_selection_results(feature_results)

#plot roc curve, feature results, plot feature selection results for random forest


#plot_roc_curve(rf_model, X_test_selected_rf, y_test)
#feature_results_rf = evaluate_n_features_random_forest(X_train, y_train, X_test, y_test, max_features=9)
#plot_feature_selection_results(feature_results_rf)

#analyze thresholds for dt_model
#analyze_thresholds(dt_model, X_test_selected, y_test)

printLine()
print("Feature Selection + Threshold Tuning for Decision Tree")
printLine()

# Evaluate for a specific threshold
model, selected_features, accuracy, precision, recall, f1 = feature_selection_with_threshold(
    X_train, y_train, X_test, y_test, n_features=5, threshold=0.06
)

plot_confusion_matrix(model, X_test[selected_features], y_test)
plot_roc_curve(model, X_test[selected_features], y_test)






#############################################################################
# NOTES
#############################################################################
"""
Hyperparamter random forest and descision tree models produce similar reuslts. Random forest takes much longer to train however.
descision tree is best candidate to move forward with.

Feature selection made logistic regression worse

Feature selection seems to have the best results with descision tree so far

BEST RESULTS SO FAR:
-------------------
Feature Selection for Decision Tree
-------------------
Selected Features by Decision Tree: ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'smoking_history']
Accuracy: 0.95275
Precision: 0.7168845935190449
Recall: 0.7382903981264637
F1 Score: 0.7274300548024228

Explanation:

Accuracy: Set is very highly imbalance with 91.5% of the data not having diabetes. 
This means that a model that predicts no diabetes for all samples will have an accuracy of 91.5%. 
This is not the best metric to evaluate the model.

Precision: Precision is the ratio of correctly predicted positive observations to the total predicted positives.
That is to say there seems to be about 30% false positives in the model. This is not ideal.

Recall: Recall is the ratio of correctly predicted positive observations to the all observations in actual class.
This is probably the most important metric. Considering that we are trying to predict diabetes, we want to make sure that we are not missing any cases of diabetes.

F1 Score: F1 Score is the weighted average of Precision and Recall.

Hyperparameter tuning gives this result:
Accuracy: 0.9722
Precision: 1.0
Recall: 0.6744730679156908
F1 Score: 0.8055944055944056

The precision is 1.0 which is perfect. This means that there are no false positives in the model.
While combined with the accuracy of 0.9722, this is tempting to say that it's the best model, however 
the recall is only 0.67, which means we are missing a lot of the cases of diabetes.

Things not tried yet:
- Model Fusion
- Regularization Techniques
"""





