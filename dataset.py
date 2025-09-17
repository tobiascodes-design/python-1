#  Data Analysis & Visualization with Pandas, Matplotlib, and Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ------------------------------
# Task 1: Load and Explore Dataset
# ------------------------------

try:
    # Load iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame  # convert to pandas DataFrame

    print("✅ Dataset loaded successfully!\n")

    # Display first few rows
    print("First 5 rows of dataset:")
    print(df.head(), "\n")

    # Check data types and missing values
    print("Dataset Info:")
    print(df.info(), "\n")

    print("Missing values in each column:")
    print(df.isnull().sum(), "\n")

    # Clean dataset: (no missing values in Iris, but showing process)
    df = df.dropna()

except FileNotFoundError:
    print("❌ Dataset file not found. Please check the file path.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# ------------------------------
# Task 2: Basic Data Analysis
# ------------------------------

# Basic statistics
print("Basic statistics of numerical columns:")
print(df.describe(), "\n")

# Group by species and compute mean of numerical columns
grouped = df.groupby("target").mean()
print("Mean values grouped by species:")
print(grouped, "\n")

# Pattern finding (example observation)
print("Observation: Sepal length tends to increase across species.\n")

# ------------------------------
# Task 3: Data Visualization
# ------------------------------

sns.set(style="whitegrid")  # prettier plots

# 1. Line chart: simulate time-series using index
plt.figure(figsize=(8, 5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color="blue")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index (simulated time)")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart: average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x="target", y="petal length (cm)", data=df, estimator="mean")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species (0=setosa, 1=versicolor, 2=virginica)")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram: distribution of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df["sepal width (cm)"], bins=15, color="orange", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot: sepal length vs petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="Set1"
)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
