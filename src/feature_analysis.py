# feature_analysis.py

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import pointbiserialr, chi2_contingency
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE

from instantiate_models import get_rfe_models

def analyze_correlations(df, features):
    """
    Function to analyze correlations and perform Point-Biserial Correlation and Chi-Squared Tests for a given DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with features and target variable 'Y'.
        features (list): List of feature column names to analyze.
    
    Returns:
        tuple: DataFrames with Point-Biserial Correlation and Chi-Squared p-values.
    """
    # Calculate the correlation matrices for different methods
    correlation_matrix_pearson = df.corr(method='pearson')
    correlation_matrix_spearman = df.corr(method='spearman')
    correlation_matrix_kendall = df.corr(method='kendall')

    # Initialize dictionaries to store results for Point-Biserial and Chi-Squared
    point_biserial_results = {'Feature': [], 'Point-Biserial Correlation': []}
    chi_squared_results = {'Feature': [], 'Chi-Squared p-value': []}

    # Calculate Point-Biserial Correlation and Chi-Squared Test for each feature
    for feature in features:
        # Point-Biserial Correlation
        corr, _ = pointbiserialr(df['Y'], df[feature])
        point_biserial_results['Feature'].append(feature)
        point_biserial_results['Point-Biserial Correlation'].append(corr)

        # Chi-Squared Test
        contingency_table = pd.crosstab(df['Y'], df[feature])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi_squared_results['Feature'].append(feature)
        chi_squared_results['Chi-Squared p-value'].append(p)

    # Convert Point-Biserial and Chi-Squared results into DataFrames
    point_biserial_df = pd.DataFrame(point_biserial_results)
    chi_squared_df = pd.DataFrame(chi_squared_results)

    # Set up the figure and subplots for all the heatmaps and matrices
    plt.figure(figsize=(18, 10))  # Adjust size to fit all plots in one row

    # Pearson's Correlation Matrix
    plt.subplot(2, 3, 1)
    sns.heatmap(correlation_matrix_pearson, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.3)
    plt.title("Pearson Correlation Matrix")

    # Spearman's Rank Correlation Matrix
    plt.subplot(2, 3, 2)
    sns.heatmap(correlation_matrix_spearman, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.3)
    plt.title("Spearman Correlation Matrix")

    # Kendall's Tau Correlation Matrix
    plt.subplot(2, 3, 3)
    sns.heatmap(correlation_matrix_kendall, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.3)
    plt.title("Kendall Correlation Matrix")

    # Point-Biserial Correlation Matrix
    plt.subplot(2, 3, 4)
    sns.heatmap(point_biserial_df.set_index('Feature').T, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.3)
    plt.title("Point-Biserial Correlation Matrix")

    # Chi-Squared p-value Matrix
    plt.subplot(2, 3, 5)
    sns.heatmap(chi_squared_df.set_index('Feature').T, annot=True, cmap='coolwarm', fmt=".4f", linewidths=.3)
    plt.title("Chi-Squared Test p-values")

    # Adjust layout to ensure everything fits
    plt.tight_layout()

    # Show the plots
    plt.show()

    return point_biserial_df, chi_squared_df


# logistic_regression.py
def perform_logistic_regression_v1(df, features, target_column):
    """
    Function to perform logistic regression on a given DataFrame and print the summary.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with features and target variable.
        features (list): List of feature column names to include in the model.
        target_column (str): Name of the target column (default is 'Y').
    
    Returns:
        result (statsmodels.LogitResults): Fitted logistic regression model result.
    """
    # Prepare the feature columns
    X = df[features]
    X = sm.add_constant(X)  # Add constant (intercept) to the model

    # Prepare the target variable
    y = df[target_column]

    # Fit logistic regression model
    model = sm.Logit(y, X)
    result = model.fit()

    # Print the model summary
    print(result.summary())
    
    return result


def random_forest_feature_analysis(df, features, target, random_num):
    """
    Function to perform Random Forest and Decision Tree feature analysis, including accuracy and feature importance.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with features and target column.
        target_column (str): The name of the target column (default is 'Y').
        feature_columns (list): List of feature columns to be used in the analysis (default is None, uses all columns except the target).
        
    Returns:
        dict: A dictionary containing accuracy and feature importance for both Decision Tree and Random Forest.
    """
    
    # Use all columns except the target column if feature_columns is not provided
    if features is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[features]  # Feature columns
    y = df[target]    # Target column
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_num)
    
    # Train and evaluate Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=random_num)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    dt_feature_importance = {feature: importance for feature, importance in zip(X.columns, dt.feature_importances_)}
    
    # Train and evaluate Random Forest Classifier
    rf = RandomForestClassifier(random_state=random_num)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_feature_importance = {feature: importance for feature, importance in zip(X.columns, rf.feature_importances_)}
    
    # Permutation importance for Random Forest
    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=random_num, n_jobs=2)
    permutation_importance_rf = {feature: importance for feature, importance in zip(X.columns, result.importances_mean)}
    
    # Organize the results into a dictionary
    results = {
        'Decision Tree Accuracy': dt_accuracy,
        'Decision Tree Feature Importance': dt_feature_importance,
        'Random Forest Accuracy': rf_accuracy,
        'Random Forest Feature Importance': rf_feature_importance,
        'Permutation Importance (Random Forest)': permutation_importance_rf
    }
    
    # Print the results
    print(f"Decision Tree Accuracy: {dt_accuracy}")
    print("Decision Tree Feature Importance:")
    for feature, importance in dt_feature_importance.items():
        print(f"{feature}: {importance}")
    
    print(f"\nRandom Forest Accuracy: {rf_accuracy}")
    print("Random Forest Feature Importance:")
    for feature, importance in rf_feature_importance.items():
        print(f"{feature}: {importance}")
    
    print("\nPermutation Importance for Random Forest:")
    for feature, importance in permutation_importance_rf.items():
        print(f"{feature}: {importance}")
    
    # Print feature importances for Random Forest
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importances (Random Forest):")
    for i in indices:
        print(f"{X.columns[i]}: {importances[i]}")
    
    return results


import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

def feature_ranking_with_rfe(df, features, target, iterations=50, random_seed=42):
    """
    Perform feature ranking using RFE (Recursive Feature Elimination) for different classifiers.
    
    Parameters:
    - df: pandas DataFrame, contains features and target columns
    - iterations: int, number of iterations for RFE
    - random_seed: int, random seed for reproducibility
    
    Returns:
    - None
    """
    if random_seed:
        np.random.seed(random_seed)
    
    # Extract features and target columns
    X = df.drop(columns=target)  # Features
    y = df[target]  # Target column
    
    feature_names = X.columns.tolist()

    # Initialize feature scores for each model
    feature_scores_rf = np.zeros(len(feature_names))
    feature_scores_lr = np.zeros(len(feature_names))
    feature_scores_dt = np.zeros(len(feature_names))
    feature_scores_svc = np.zeros(len(feature_names))
    feature_scores_gb = np.zeros(len(feature_names))
    feature_scores_ada = np.zeros(len(feature_names))
    feature_scores_et = np.zeros(len(feature_names))
    feature_scores_rc = np.zeros(len(feature_names))
    feature_scores_lasso = np.zeros(len(feature_names))
    feature_scores_SGD = np.zeros(len(feature_names))

    # Iterate over models
    for model, feature_scores in zip(
        [RandomForestClassifier(random_state=random_seed),
         LogisticRegression(max_iter=5000),
         DecisionTreeClassifier(random_state=random_seed),
         SVC(kernel='linear'),
         GradientBoostingClassifier(),
         AdaBoostClassifier(),
         ExtraTreesClassifier(),
         RidgeClassifier(),
         Lasso(),
         SGDClassifier(max_iter=1000, random_state=random_seed)],
        [feature_scores_rf, feature_scores_lr, feature_scores_dt, feature_scores_svc,
         feature_scores_gb, feature_scores_ada, feature_scores_et, feature_scores_rc, feature_scores_lasso, feature_scores_SGD]
    ):
        print(f"\nRunning RFE for model: {model.__class__.__name__}")

        for _ in range(iterations):
            # Generate a random state for each iteration
            random_state = np.random.randint(1000, 9999)
            
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
            
            # Apply RFE
            rfe = RFE(estimator=model, n_features_to_select=1)
            rfe.fit(X_train, y_train)
            
            # Update feature scores (lower ranks are more important)
            feature_scores += rfe.ranking_

        # Average the scores over the number of iterations
        average_scores = feature_scores / iterations
        print(f"average_scores for {model.__class__.__name__}", average_scores)

        # Get the final feature importance ranking
        final_ranking = dict(zip(feature_names, average_scores))
        final_ranking_sorted = sorted(final_ranking.items(), key=lambda x: x[1])

        print(f"Final Feature Importance Ranking for {model.__class__.__name__} (lower scores are more important):")
        for feature, score in final_ranking_sorted:
            print(f"{feature}: {score:.2f}")
