import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['species'] = target_names[y]

# Create and save the Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species', palette='viridis')
plt.title('Sepal Length vs. Sepal Width')
plt.savefig('scatter_plot.png')  # Save the figure as an image file
plt.close()  # Close the figure

# Create and save the Pair Plot
pair_plot = sns.pairplot(df, hue='species')
pair_plot.fig.suptitle('Pair Plot of Iris Dataset', y=1.02)  # Add a title
pair_plot.savefig('pair_plot.png')  # Save the figure as an image file
plt.close()  # Close the figure

