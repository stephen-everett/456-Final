# This file contains the code related to Step 1. Data preprocessing

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from utils import printLine

# EDA
class DataPreprocessor:
    def __init__(self, dataset_path):
        """
        Initialize the DataPreprocessor with the dataset path.
        """
        self.dataset_path = dataset_path
        self.df = None

    def load_dataset(self):
        """
        Load the dataset from the specified path.
        """
        self.df = pd.read_csv(self.dataset_path)
        print("Dataset loaded successfully.")

    def head(self, rows=5):
        """
        Display the first few rows of the dataset.
        """
        print("HEAD")
        printLine()
        print(self.df.head(rows))
        print()

    def describe(self):
        """
        Display summary statistics for numerical features.
        """
        print("DESCRIBE")
        printLine()
        print(self.df.describe())
        print()

    def info(self):
        """
        Display information about the dataset, including column types and non-null counts.
        """
        print("INFO")
        printLine()
        print(self.df.info())
        print()

    def shape(self):
        """
        Display the number of rows and columns in the dataset.
        """
        print("SHAPE")
        printLine()
        print(f"Dataset contains {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        print()

    def class_distribution(self, target_column):
        """
        Display the distribution of the target variable.
        """
        print("Class Distribution: ")
        printLine()
        target_counts = self.df[target_column].value_counts()
        print(target_counts)
        print("\nOpening graph...")
        target_counts.plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.show()
        print('Done.')
    
    def encode_categorical(self):
        """
        Encode categorical features using LabelEncoder.
        """
        print("\nEncoding categorical features...")
        label_encoder_gender = LabelEncoder()
        self.df['gender'] = label_encoder_gender.fit_transform(self.df['gender'])

        label_encoder_smoking = LabelEncoder()
        self.df['smoking_history'] = label_encoder_smoking.fit_transform(self.df['smoking_history'])

        print("\nENCODED HEAD: ")
        printLine()
        self.head()

    def histogram_outliers(self):
        """
        Detect outliers using histograms
        """
        self.df.hist(figsize=(12, 8), bins=20)
        plt.tight_layout()
        plt.show()

    def boxplot_outliers(self):
        """
        Detect outliers using boxplots.
        """
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns

        for col in numerical_cols:
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

    def correlation_analysis(self):
        """
        Display the correlation matrix and visualize it with a heatmap.
        """
        print("\nExamining correlations....")
        corr_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def getDf(self):
        return self.df
"""
#loading the dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Check to see what the data looks like (we can just open the csv manually)
print("\nTaking an initial look at the dataset: \n")
print("HEAD")
printLine()
print(df.head())
print()
"""
"""

gender   age  hypertension  heart_disease smoking_history    bmi  HbA1c_level  blood_glucose_level  diabetes
0  Female  80.0             0              1           never  25.19          6.6                  140         0
1  Female  54.0             0              0         No Info  27.32          6.6                   80         0
2    Male  28.0             0              0           never  27.32          5.7                  158         0
3  Female  36.0             0              0         current  23.45          5.0                  155         0
4    Male  76.0             1              1         current  20.14          4.8                  155         0

#smoking_history might need encoding

"""
"""
# Taking note of some of the numerical
print("DESCRIBE")
printLine() 
print(df.describe())
print()
"""
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
"""
print("INFO")
printLine()
print(df.info())
print()
"""
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
"""
print("SHAPE")
printLine()
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print()
"""
"""
####################
# target is diabetes
target_counts = df['diabetes'].value_counts()
print("Class Distribution: ")
printLine()
print(target_counts)
"""
"""
diabetes
0    91500
1     8500
Name: count, dtype: int64

# 8.5% positive
"""
"""
print("\nOpening graph...")
target_counts.plot(kind='bar')
plt.title('Class Distrubution')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.show()
print('Done.')
#########################
"""
# ENCODING
"""
 Using label encoding because the smoking history doesn't fit nicely into a binary classification
"""
"""
print('\nencoding....')
label_encoder_gender = LabelEncoder()
df['gender'] = label_encoder_gender.fit_transform(df['gender'])

# Label encode 'smoking_history'
label_encoder_smoking = LabelEncoder()
df['smoking_history'] = label_encoder_smoking.fit_transform(df['smoking_history'])

# Display the first few rows
print("ENCODED HEAD: ")
printLine()
print(df.head())
"""


##############################
"""
print("\nExamining columns for outliers...")
# Histograms for numerical features
df.hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

# Boxplots for potential outliers
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numerical_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


########
print("\nExamining correlations....")
# Correlation matrix
corr_matrix = df.corr()

# Visualize the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
"""
# Pairplot for numerical features
"""This takes a very long time"""
#sns.pairplot(df, hue='diabetes')  
#plt.show()



