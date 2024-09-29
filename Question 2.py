# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:32:06 2024
Assignment 1: Question 2
@author: Damien Liscio
"""

# Import required libraries
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Question 1: Generate Swiss roll dataset
# Use make_swiss_roll to generate dataset
X, y = make_swiss_roll(n_samples=1000,noise=0.05, random_state=0)

# Question 2: Plot the resulting generated Swiss roll dataset
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    X[:,0], X[:,1], X[:,2], c=y, s=50, alpha=0.8
)
ax.set_title("Swiss Roll Dataset Plot")  # Set plot title
ax.view_init(azim=-66, elev=12)
_ = ax.text2D(0.8, 0.05, s="n_samples=1000", transform=ax.transAxes)

# Question 3: Use Kernel PCA (kPCA) with linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points).
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_binned = np.digitize(y, bins=np.linspace(np.min(y), np.max(y), 10))
# Make list of required kernels and empty dictiornary
kernels = ['linear', 'rbf', 'sigmoid']
X_kpca_values = {}
# Loop through kernels and perform kPCA & add kernel and value to dictionary
for kernel in kernels:
    X_kpca = KernelPCA(n_components=2, kernel=kernel, gamma=0.04)
    X_kpca_values[kernel] = X_kpca.fit_transform(X_scaled)

# Question 4: Plot the kPCA results of applying the linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points) from (3). Explain and compare the results 
# Plot the kPCA results
plt.figure(figsize=(11, 4))
for i, kernel in enumerate(kernels):
    plt.subplot(1, 3, i+1)
    plt.scatter(X_kpca_values[kernel][:, 0], X_kpca_values[kernel][:, 1], c=y, cmap=plt.cm.hot, s=10)
    plt.title(f'kPCA with {kernel.capitalize()} Kernel')
    plt.xlabel('Z1')
    plt.ylabel('Z2')
plt.tight_layout()
plt.show()

# In comparing the results from plotting the kPCA results for the various kernels, it is clear to see 
# there was a discrepency in their quality. The rbf kernel was the best at unrolling the data and had
# the least amount of overlapping space and the most easily distinguishable data points. The sigmoid & 
# linear kernel led to just about the same result, both with hard a lot of hard to distinguish data points
# and an overlapping of the data.

# Question 5: Using kPCA and a kernel of your choice, apply Logistic Regression for classification. Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at the end of the pipeline. Print out best parameters found by GridSearchCV.
# Make pipeline with KernelPCA and logistic Regression
clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
    ])
# Make parameters
param_grid = [{
    "kpca__gamma" : np.linspace(0.03,0.05,10),
    "kpca__kernel" : ["rbf","sigmoid"]
    }]
# Use pipeline & parameters with grid search to find the best parameters
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_scaled, y_binned)
print(grid_search.best_params_) # Print best parameters

# Question 6: Plot the results from using GridSearchCV in (5).
# Get grid search results
results = grid_search.cv_results_
# Extract relevant values
mean_test_scores = results['mean_test_score']
params = results['params']
gammas = np.array([param['kpca__gamma'] for param in params])
kernels = np.array([param['kpca__kernel'] for param in params])

# Plot gamma vs mean test score results
plt.figure(figsize=(10, 6))
for kernel in np.unique(kernels):
    mask = kernels == kernel
    plt.plot(gammas[mask], mean_test_scores[mask], label=f'Kernel: {kernel}')
plt.title('Grid Search Results: Gamma vs Mean Test Score', fontsize=16)
plt.xlabel('Gamma', fontsize=14)
plt.ylabel('Mean Test Score', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()