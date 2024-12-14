from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def hyper_train_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 0.15, 0.2, 0.3, 1, 5, 10],
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga'],
    }
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters for Logistic Regression:", grid_search.best_params_)
    return grid_search.best_estimator_

def hyper_train_decision_tree(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': [None, 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'ccp_alpha': [0.0, 0.01, 0.05],
        'class_weight': [None, 'balanced'] #added later
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    print("Best parameters for Decision Tree:", grid_search.best_params_)
    return grid_search.best_estimator_

def hyper_train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
        'bootstrap': [True]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    print("Best parameters for Random Forest:", grid_search.best_params_)
    return grid_search.best_estimator_

def feature_selection_rfe_logistic(X_train, y_train, X_test):
    # Instantiate a Logistic Regression model
    model = LogisticRegression(max_iter=200, random_state=42)
    
    # Recursive Feature Elimination
    rfe = RFE(estimator=model, n_features_to_select=5)  # Select top 5 features
    rfe.fit(X_train, y_train)
    
    # Get selected features
    selected_features = X_train.columns[rfe.support_]
    print("Selected Features by RFE:", list(selected_features))
    
    # Train a Logistic Regression model on the selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    model.fit(X_train_selected, y_train)
    return model, selected_features, X_test_selected

def feature_selection_decision_tree(X_train, y_train, X_test, n_features=5):
    # Instantiate a Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    
    # Train the Decision Tree model
    model.fit(X_train, y_train)
    
    # Compute feature importances
    importances = model.feature_importances_
    
    # Rank features by importance
    feature_ranking = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature, _ in feature_ranking[:n_features]]
    print("Selected Features by Decision Tree:", selected_features)
    
    # Reduce feature set to selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Retrain the model on the reduced feature set
    model.fit(X_train_selected, y_train)
    return model, selected_features, X_test_selected

def feature_selection_random_forest(X_train, y_train, X_test, n_features=5):
    # Instantiate a Random Forest model
    model = RandomForestClassifier(random_state=42)
    
    # Train the Random Forest model
    model.fit(X_train, y_train)
    
    # Compute feature importances
    importances = model.feature_importances_
    
    # Rank features by importance
    feature_ranking = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature, _ in feature_ranking[:n_features]]
    print("Selected Features by Random Forest:", selected_features)
    
    # Reduce feature set to selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Retrain the model on the reduced feature set
    model.fit(X_train_selected, y_train)
    return model, selected_features, X_test_selected

def combined_train_decision_tree(X_train, y_train, X_test, n_features=5):
    # Step 1: Feature Selection using Decision Tree
    print("-------------------")
    print("Feature Selection")
    print("-------------------")
    
    # Train a Decision Tree to rank features
    selection_model = DecisionTreeClassifier(random_state=42)
    selection_model.fit(X_train, y_train)
    
    # Compute feature importances
    importances = selection_model.feature_importances_
    feature_ranking = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature, _ in feature_ranking[:n_features]]
    print("Selected Features:", selected_features)
    
    # Reduce the dataset to selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Step 2: Hyperparameter Tuning with GridSearchCV
    print("-------------------")
    print("Hyperparameter Tuning")
    print("-------------------")
    
    param_grid = {
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': [None, 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'ccp_alpha': [0.0, 0.01, 0.05],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train_selected, y_train)
    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    return best_model, X_test_selected

def evaluate_n_features(X_train, y_train, X_test, y_test, max_features=10):
    """
    Evaluate model performance for different numbers of selected features.
    """
    results = []
    for n in range(1, max_features + 1):
        model, selected_features, X_test_selected = feature_selection_decision_tree(X_train, y_train, X_test, n_features=n)
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((n, accuracy, selected_features))
        print(f"n_features={n}, Accuracy={accuracy:.4f}, Features={selected_features}")
    
    return results

def plot_feature_selection_results(feature_results):
    """
    Plot the accuracy as a function of the number of features.
    """
    n_features = [res[0] for res in feature_results]
    accuracies = [res[1] for res in feature_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_features, accuracies, marker='o')
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Selected Features")
    plt.show()

def evaluate_n_features_random_forest(X_train, y_train, X_test, y_test, max_features=10):
    """
    Evaluate model performance for different numbers of selected features.
    """
    results = []
    for n in range(1, max_features + 1):
        model, selected_features, X_test_selected = feature_selection_random_forest(X_train, y_train, X_test, n_features=n)
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((n, accuracy, selected_features))
        print(f"n_features={n}, Accuracy={accuracy:.4f}, Features={selected_features}")
    
    return results


def analyze_thresholds(model, X_test, y_test):
    """
    Plot ROC curve and annotate thresholds to understand their impact.
    """
    # Get predicted probabilities for the positive class
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve with Thresholds")
    
    # Annotate thresholds
    for i in range(0, len(thresholds), len(thresholds) // 10):  # Add annotations for every ~10% of thresholds
        plt.annotate(f"{thresholds[i]:.2f}", (fpr[i], tpr[i]))
    
    plt.legend()
    plt.show()

def feature_selection_with_threshold(X_train, y_train, X_test, y_test, n_features=5, threshold=0.5):
    # Step 1: Feature Selection using Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Compute feature importances
    importances = model.feature_importances_
    feature_ranking = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature, _ in feature_ranking[:n_features]]
    print("Selected Features by Decision Tree:", selected_features)
    
    # Reduce dataset to selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Retrain the model on the reduced feature set
    model.fit(X_train_selected, y_train)
    
    # Step 2: Get Probabilities and Apply Threshold
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]  # Probabilities for the positive class
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)  # Apply custom threshold
    
    # Step 3: Evaluate the Model
    accuracy = accuracy_score(y_test, y_pred_threshold)
    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)
    
    # Print Metrics
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    return model, selected_features, accuracy, precision, recall, f1

