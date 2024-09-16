import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the Iris dataset
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names

# Apply PCA
def apply_pca(X, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

# Train and evaluate the classifier
def train_and_evaluate(X_train, X_test, y_train, y_test, target_names):
    classifier = LogisticRegression(max_iter=200)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    return accuracy, report

# Plotting PCA results
def plot_pca(X_pca, y, pca):
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Iris Dataset')
    plt.colorbar(scatter, label='Species')
    plt.show()

# Main function
def main():
    # Load data
    X, y, feature_names, target_names = load_data()
    
    # Apply PCA
    X_pca, pca = apply_pca(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    
    # Train and evaluate
    accuracy, report = train_and_evaluate(X_train, X_test, y_train, y_test, target_names)
    
    # Print results
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)
    
    # Plot PCA results
    plot_pca(X_pca, y, pca)

if __name__ == "__main__":
    main()

