# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:04:01 2024
Assignment 1: Question 1
@author: Damien Liscio
"""

# Import statements for necessary libraries
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

# Question 1: Retrieve and load the mnist_784 dataset of 70,000 instances.
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Split the data into training (60,000) and testing (10,000)
X_train, y_train = mnist.data[:60000], mnist.target[:60000]
X_test, y_test =  mnist.data[60000:], mnist.target[60000:]

# Print dataset shapes
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Question 2: Display each digit.
# Convert features into int type to match desired digit
y_train = y_train.astype(int)
# Create figure with 10 subplots
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
# Loop through features to find the indexes for the occurrences of each digit (0-9)
for i in range(10):
    index = np.where(y_train == i)[0][0]  # Store as array and get first element as index
    
    # Get the desired image and label and reshape image
    image = X_train[index].reshape(28, 28)
    label = y_train[index]
    
    # Plot given image in a subplot and give appropriate title
    ax = axes[i // 5, i % 5]
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label {label}')
    ax.axis('off')
    
# Display plot of each digit
plt.tight_layout()
plt.show()

# Question 3: Use PCA to retrieve the 1st and 2nd principal component and output their explained variance ratio.
# Create the instance of PCA with 2 principal components
pca = PCA(n_components=2)
# Fit the model and transform the data
X_pca = pca.fit_transform(X_train)
# Output explained variance ratio
ev1 = pca.explained_variance_ratio_[0]
ev2 = pca.explained_variance_ratio_[1]
print(f"Explained Variance Ratio for Principal Component 1: {ev1:.4f}")
print(f"Explained Variance Ratio for Principal Component 2: {ev2:.4f}")

# Question 4: Plot the projections of the 1st & 2nd principal component onto a 1D hyperplane.
pc1 = X_pca[:, 0]  # Projections on the first principal component
pc2 = X_pca[:, 1]  # Projections on the second principal component

# Create a figure for projections & plot
plt.figure(figsize=(10, 5))
plt.scatter(pc1, np.zeros_like(pc1), alpha=0.5, color='blue', label='PC1', marker='o')
plt.scatter(np.zeros_like(pc2), pc2, alpha=0.5, color='red', label='PC2', marker='x')
plt.title('Projections onto 1D Hyperplane')
plt.xlabel('Principal Component Value')
plt.legend()
plt.show()

# Question 5: Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions
# Set number of batches for increments
n_batches = 100
# Initiate incremental PCA with 154 components
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)
    
X_reduced = inc_pca.transform(X_train)

# Question 6: Display the original and compressed digits from (5)
# Inverse transform to reconstruct images
X_reconstructed = inc_pca.inverse_transform(X_reduced)

# Display original and compressed digits
num_images = 5  # Number of images to display
fig, axes = plt.subplots(2, num_images, figsize=(15, 5))

for i in range(num_images):
    # Original image
    original_image = X_train[i].reshape(28, 28)
    axes[0, i].imshow(original_image, cmap='gray')
    axes[0, i].set_title(f'Original {y_train[i]}')
    axes[0, i].axis('off')
    
    # Compressed image
    reconstructed_image = X_reconstructed[i].reshape(28, 28)
    axes[1, i].imshow(reconstructed_image, cmap='gray')
    axes[1, i].set_title(f'Reconstructed {y_train[i]}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()