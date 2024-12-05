# This file contains the code related to Step 1. Data preprocessing

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        self.scaler = StandardScaler()

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

    def standardize(self):
        # Identify continuous numerical columns (exclude binary/categorical columns)
        continuous_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        
        # Standardize only continuous numerical features
        self.df[continuous_cols] = self.scaler.fit_transform(self.df[continuous_cols])
        
        print("Continuous numerical features standardized.")
        print(self.df.head())

    def getDf(self):
        return self.df

