# week-6-python-assignment

Objective For this Assignment:

To load and analyze a dataset using the pandas library in Python.
To create simple plots and charts with the matplotlib library for visualizing the data.



Submission Requirements
Submit a Jupyter notebook (.ipynb file) or Python script (.py file) containing:
Data loading and exploration steps.
Basic data analysis results.
Visualizations.
Any findings or observations.
Objective For this Assignment:

To load and analyze a dataset using the pandas library in Python.
To create simple plots and charts with the matplotlib library for visualizing the data.



Submission Requirements
Submit a Jupyter notebook (.ipynb file) or Python script (.py file) containing:
Data loading and exploration steps.
Basic data analysis results.
Visualizations.
Any findings or observations.

****


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target (species) as a new column
df['species'] = iris.target

# Map the target numbers to species names for easier interpretation
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_map)

# Display the first few rows of the dataset
print(df.head())

# Explore the structure of the dataset
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

Explanation:
We loaded the dataset using load_iris() from sklearn.datasets.
The dataset is then converted into a pandas DataFrame for easier analysis.
We map the numerical species values to their corresponding names (setosa, versicolor, virginica).
We print the first few rows of the dataset with .head().
We use .info() to check the structure, including data types and missing values.
.isnull().sum() checks for any missing values in the dataset.


# Compute basic statistics for the numerical columns
print("\nBasic Statistics (Numerical Columns):")
print(df.describe())

# Group by species and compute the mean of numerical columns
print("\nGroup by species and compute mean:")
species_mean = df.groupby('species').mean()
print(species_mean)

Explanation:
We use .describe() to get a summary of the numerical columns (mean, standard deviation, min, max, etc.).
We group the dataset by the species column and calculate the mean for each species using .groupby() and .mean().

# Line plot for Sepal length across different species
plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x='species', y='sepal length (cm)', marker='o')
plt.title('Sepal Length Across Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

Explanation:
This line chart displays the trend of Sepal length across the three species. We use seaborn.lineplot() to visualize the data.


# Bar chart showing average petal length for each species
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length for Each Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

Explanation:
This bar chart compares the average Petal length for each species using seaborn.barplot().

# Histogram for Sepal Width
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

Explanation:
This histogram shows the distribution of the sepal width across all flowers in the dataset. We use sns.histplot() with kde=True to also show a Kernel Density Estimate (KDE) curve.

# Scatter plot of Sepal Length vs Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='Set1')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()



Explanation:
This scatter plot visualizes the relationship between sepal length and petal length. Different species are colored differently using hue='species'.