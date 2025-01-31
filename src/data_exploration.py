# src/data_exploration.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from tabulate import tabulate
from IPython.display import display, Markdown
from utils import print_bold, color_value

def data_exploration(df):
    # Display the shape of the dataset
    print_bold("Shape of the dataset:")
    print(df.shape)
    print("\n")
    
    # Display the first 6 rows of the dataset
    print_bold("First 6 rows of the dataset:")
    print(df.head(6))
    print("\n")
    
    # Display information about the dataset
    print_bold("Dataset info:")
    df.info(verbose=True)
    print("\n")
    
    # Display unique values for all columns
    print_bold("Unique values in each column:")
    for column in df.columns:
        print(f'{column}: {df[column].unique()}')
    print("\n")
    
    # Count missing values in the dataset
    print_bold("Missing values in each feature:")
    print(df.isnull().sum())
    print("\n")
    
    # Display summary statistics
    print_bold("Summary statistics of the dataset:")
    print(df.describe())
    print("\n")
    
    # Check for columns with constant values
    print_bold("Columns with constant values:")
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print(constant_cols)
    else:
        print("No columns in the dataset contain the same value for every row.")
    print("\n")
    
    # Correlation matrix
    print_bold("Correlation matrix:")
    correlation_matrix = df.corr()
    print(correlation_matrix)
    print("\n")
    
    # Check for missing data and print value counts
    print_bold("Missing data value counts in each column:")
    missing_data_counts = df.isnull().sum()
    print(missing_data_counts)
    print("\n")
        
    # Distribution of data types
    print_bold("Distribution of data types in the dataset:")
    print(df.dtypes.value_counts())
    print("\n")
    
    # Check for negative values in specific columns (example 'X1')
    print_bold("Checking for negative values in 'X1' column:")
    print(df[df['X1'] < 0])
    print("\n")
    
    # Count non-null values in each column
    print_bold("Non-null value counts for each feature:")
    print(df.count())
    print("\n")
    
    # First and Last 5 Rows
    print_bold("First 5 rows of the dataset:")
    print(df.head())
    print("\n")
    
    print_bold("Last 5 rows of the dataset:")
    print(df.tail())
    print("\n")

        # Skewness and Kurtosis analysis
    print_bold("Skewness and Kurtosis of each feature:")
    # Preparing data for tabulation
    results = []
    for column in df.columns:
        # Get values for the column and compute skewness and kurtosis
        column_values = df[column].dropna().to_numpy()  # Ensure no NaNs are included
        skewness = skew(column_values, bias=False)
        kurt = kurtosis(column_values, bias=False)
        # Append the results with the formatted color values for skewness and kurtosis
        results.append([column, color_value(skewness), color_value(kurt)])
    # Printing the results as a table
    headers = ["Feature", "Skewness", "Kurtosis"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("\n")

    # Plot histograms for numeric features with count above each bar (if count > 0)
    print_bold("Histograms of Numeric Columns:")
    ax = df.hist(figsize=(12, 12), bins=20)
    plt.suptitle('Histograms of Numeric Columns')

    # Ensure we are working with a list of axes, even for a single subplot
    if isinstance(ax, np.ndarray):
        axes = ax.flatten()  # If multiple subplots, flatten the array
    else:
        axes = [ax]  # If only one subplot, make it a list

    # Get the maximum y-axis limit to avoid text overlapping with the top line
    y_max = axes[0].get_ylim()[1]  # Get the y-axis limit from the first subplot

    # Add count above each bar, but only if the count is greater than 0
    for axes in axes:  # Loop through each subplot
        for patch in axes.patches:  # Loop through each bar (patch)
            # Get the height of the bar
            height = patch.get_height()
            
            # Only print if the count is greater than 0
            if height > 0:
                # Get the x position of the bar
                x_position = patch.get_x() + patch.get_width() / 2
                # Adjust y position to avoid overlapping the top line
                y_position = height+ .001 * y_max  # A small buffer from the top of the bar
                # Add the count above the bar
                axes.text(x_position, y_position, str(int(height)), ha='center', va='bottom')

    plt.show()



    # Check for outliers using boxplot
    print_bold("Box plot of the dataset:")
    plt.figure(figsize=(12, 6))  # Set the figure size
    sns.boxplot(data=df, orient='v')  # Create box plots for each feature
    plt.title("Box Plots for All Features")  # Set plot title
    plt.xlabel("Features")  # Set x-label
    plt.ylabel("Value")  # Set y-label
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

