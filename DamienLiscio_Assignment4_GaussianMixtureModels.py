# -*- coding: utf-8 -*-
"""
COMP257 Sec.402 Unsupervised & Reinforcement Learning
Assignment 4: Gaussian Mixture Models
Name: Damien Liscio
Student #: 301237966
Due: Monday, October 28th, 2024

"""

#Import statements
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter

# 1. Retrieve and load the dataset
olivetti = fetch_olivetti_faces() #Retrieve dataset
X = olivetti.data  #Assign image data
y = olivetti.target  #Assign image labels

# 2. Split the dataset into training, testing & validation using stratified sampling
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=66) #Initialize Stratified Shuffle Split
for train_index, test_index in sss.split(X,y):    #Split data & labels into training and testing data
    X_train, X_test = X[train_index], X[test_index]  #Assign testing and training data appropriately after split
    y_train, y_test = y[train_index], y[test_index]  #Assign testing  and training labels appropriatley after split
    
for train_index, val_index in sss.split(X_train, y_train): #Split training data & labels into training and validation data
    X_train, X_val = X[train_index], X[val_index]  #Assign validation and training data appropriately after split
    y_train, y_val = y[train_index], y[val_index]  #Assign validation and training labels appropriatley after split
    
#Print sizes of training, test, and validation sets
print(f"Training Set Size: {X_train.shape[0]}/400, {(X_train.shape[0]/400)*100}%  \nTest Set Size: {X_test.shape[0]}/400, {(X_test.shape[0]/400)*100}% \nValidation Test Size: {X_val.shape[0]}/400, {(X_val.shape[0]/400)*100}%\n")

# 3. Apply PCA on the training data, preserving 99% of the variance, to reduce the dataset's dimensionality
print(f"Training Data Dimensions: {X_train.shape[1]}")
pca = PCA(0.99, whiten=True) #Initialize PCA, preserving 99% of the datas variance
X_train = pca.fit_transform(X_train)  #Apply PCA to training data
print(f"Reduced Training Data Dimensions: {X_train.shape[1]}")  #Print reduced training data shape

# 4. Determine the most suitable covariance type for the dataset
covariance_types = ['diag', 'full', 'spherical', 'tied']; #List of covariance types to use with model
aic_scores_cv = {type: [] for type in covariance_types} #AIC scores gathered from model
for type in covariance_types:  #Loop through covariance types
    for n in range(1,81,5):  #Loop through number of components 
        model = GaussianMixture(n_components=n, covariance_type=type, random_state=66).fit(X_train)  #Initialize model
        aic_score = model.aic(X_train) #Get aic score for model 
        aic_scores_cv[type].append((n, aic_score))  #Add AIC score to dictionary
for type in covariance_types:  #Loop through each type
    print(f"\nCovariance Type: {type}")  #Print the covariance type
    print("Components | AIC Score")  #Headers for number of components and AIC score for given covariance type
    for n, aic_score in aic_scores_cv[type]:  #Get number of components and AIC scores for each type
        print(f"{n:>9} | {aic_score:2f}")  #Print the components and AIC score for given covariance type

"""

Based on the results, the most suitable covariance type is full. This is for a couple of reasons. The first being the 
lowest AIC score is achieved with full, somewhere between 0-20 components. This shows that the model is identifying 
similarities within the data in this range of components. The second reason being the other 3 models reach their lowest 
AIC scores at the maximum number of components. This indicates that with these covariance types, the model is not 
grouping the data points effectively into clusters and is having issues finding the underlying similarities within the data.
All 3 AIC scores are trending downward slightly up to the maximum number of components, and as such it is safe to assume
the trend would continue if the number of components were to be increased. For these reasons, the most suitable covariance
type is full.

"""

# 5. Determine the minimum number of clusters that best represent the dataset using either AIC or BIC
n_components = []; #Range of clusters to use with model
aic_scores = [] #AIC scores gathered from model
for n in np.arange(1, 20): #Use range of 1-20 because based off covariance results the minimum number of clusters occurs in this range
    model = GaussianMixture(n, covariance_type='full', random_state=66).fit(X_train)  #Initialize model
    aic_scores.append(model.aic(X_train))  #Add AIC score to list
    n_components.append(n)  #Add number of components for this iteration to list
plt.plot(n_components, aic_scores, label='AIC')  #Plot the results of AIC for each model
plt.legend(loc='best')  #Plot legend
plt.xlabel('n_components');  #Plot label
plt.show() #Show the plot
best_aic_index = np.array(aic_scores).argmin()  #Get the index of the lowest AIC score
print(f"\nMinimum number of clusters that best represents the dataset: {n_components[best_aic_index]}\n") #Use index to print number of components for lowest AIC score

"""

The minimum number of clusters which best represent the dataset given AIC is 5 clusters. This is because given the results
from the plot with the AIC results for the range of clusters, 5 yielded the lowest AIC value.

"""

# 6. Plot the results from 3 and 4
#Get explained variance for each component from pca above to plot #3
cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100  #Convert cumulative explained ratio to percentage
components = np.arange(1, len(cumulative_variance) + 1) #Get component index
#Make and display plot for #3
plt.plot(components, cumulative_variance, label="Explained Variance Ratio")
plt.axhline(y=99, color='red', linestyle='--', label="99% Variance Threshold") #Plot threshold line of 99%
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance As %")
plt.title("Explained Variance by Number of Principal Components")
plt.legend()
plt.show()

#Make and display plot for #4 AIC scores
for type in covariance_types: #Loop through each covariance type
    components, aic_scores = zip(*aic_scores_cv[type])  #Get components and AIC scores for covariance type
    plt.plot(components, aic_scores, label=f"{type} Covariance") #Plot the AIC scores compared to number of components for covariance type
#Add plot details and display
plt.xlabel("Number of Components")
plt.ylabel("AIC Score")
plt.title("AIC Score vs Number of Components for Various Covariance Types")
plt.legend()
plt.show()

# 7. Output the hard clustering assignments for each instance to identify which cluster each image belongs to
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=66).fit(X_train)  #Train model with found ideal parameters
print(f"Confirmation Model has Converged: {gmm.converged_}\n")
X_test = pca.transform(X_test)  #Apply PCA to X_test
hard_clusters = gmm.predict(X_train) #Predict to get clusters for each image
print("Hard Clustering Assignments for Each Image: \n", hard_clusters, "\n") #Print hard clustering assignments

# 8. Output the soft clustering probabilities for each instance to show the likelihood of each image belonging to each cluster
soft_clusters = gmm.predict_proba(X_test) #Get probabilities for each image belonging to each cluster
print("Soft Clustering Probabilities for Each Image: \n", soft_clusters, "\n")

# 9. Use the model to generate some new faces and visualize them
data_new, label_new = gmm.sample(5) #Generate 5 new images
faces_new = pca.inverse_transform(data_new) #Apply PCA inverse transform data back to its original space
for img in faces_new:  #Loop through each new image in new set of faces
    plt.imshow(img.reshape(64,64))  #Reshape image
    plt.axis('off') #Turn off plots axis for image
    plt.show() #Display new image

# 10. Modify some images
img_rotated = rotate(pca.inverse_transform(X_test[0]).reshape(64,64), angle=30) #Rotate the image by 15 degrees
img_flipped = np.flipud(pca.inverse_transform(X_test[1]).reshape(64,64)) #Flip the image horizontally
img_darkened = pca.inverse_transform(X_test[2]).reshape(64,64) * 0.01  #Reduce Image intensity
img_blurred = gaussian_filter(pca.inverse_transform(X_test[3]).reshape(64,64), sigma=1) #Blur image

#Rotated Imagw
plt.imshow(img_rotated) #Reshape image
plt.axis('off') #Turn off plots axis for image
plt.show() #Turn off plots axis for image

#Flipped Image
plt.imshow(img_flipped) #Reshape image
plt.axis('off') #Turn off plots axis for image
plt.show() #Display new image

#Darkened Image
plt.imshow(img_darkened) #Reshape image
plt.axis('off') #Turn off plots axis for image
plt.show() #Display new image

#Blurred Image
plt.imshow(img_blurred) #Reshape image
plt.axis('off') #Turn off plots axis for image
plt.show() #Display new image

# 11. Determine if the model can detect the anamolies produced in step 10 by comparing the output of the score_samples() method for normal images and for anomolies
test_scores = gmm.score_samples(X_test)  #Get scores for test images

img_rotated_flat = img_rotated.flatten() #Flatten image
img_rotated_flat = pca.transform(img_rotated_flat.reshape(1,-1)) #Transform and reshape
rotated_scores = gmm.score_samples(img_rotated_flat) #Get scores for rotated image

img_flipped_flat = img_flipped.flatten() #Flatten image
img_flipped_flat = pca.transform(img_flipped_flat.reshape(1,-1)) #Transform and reshape
flipped_scores = gmm.score_samples(img_flipped_flat) #Get scores for flipped image

img_darkened_flat = img_darkened.flatten() #Flatten image
img_darkened_flat = pca.transform(img_darkened_flat.reshape(1,-1)) #Transform and reshape
darkened_scores = gmm.score_samples(img_darkened_flat)  #Get scores for darkened image

img_blurred_flat = img_blurred.flatten() #Flatten image
img_blurred_flat = pca.transform(img_blurred_flat.reshape(1,-1)) #Transform and reshape
blurred_scores = gmm.score_samples(img_blurred_flat)  #Get scores for blurred image

#Print and compare mean scores for each image
print(f"Mean score for original images: {np.mean(test_scores):,.2f}")  #Print score for original images
print(f"Score for rotated images: {rotated_scores[0]:,.2f}") #Print score for rotated image
print(f"Score for flipped images: {flipped_scores[0]:,.2f}") #Print score for flipped image
print(f"Score for darkened images: {darkened_scores[0]:,.2f}") #Print score for darkened image
print(f"Score for blurred images: {blurred_scores[0]:,.2f}") #Print score for darkened image