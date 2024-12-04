from preprocessing import DataPreprocessor
from utils import printLine

filename = 'diabetes_prediction_dataset.csv'

# load up the data into the pre processor
pre_processor = DataPreprocessor(filename)
pre_processor.load_dataset()

#check to see what the data looks like initially:
pre_processor.head()

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
pre_processor.class_distribution('diabetes')
"""
A graph opens
"""

# show histograms to identify outliers
pre_processor.histogram_outliers()
"""
A graph opens
"""

# show box-plots to identify outliers
pre_processor.boxplot_outliers()
"""
A graph opens
"""

# show a correlation analysis
pre_processor.correlation_analysis()
"""
A graph opens
"""




# TODO choose a standarization model. This will depend on the model we train
# Standardization VS Normalization









# retrieve DF
df = pre_processor.getDf()



#examine df
printLine()
print("Showing dataframe after pre_processor...")
print("HEAD")
printLine()
print(df.head())
"""
Things to answer:
    1. Are there missing values? Which columns are affected?
    2. Do numerical features have outliers? Are they normally distributed or skewed?
    3. Are there correlations between features or with the target variable?
    4. Is the target variable balanced, or is there a class imbalance?
"""