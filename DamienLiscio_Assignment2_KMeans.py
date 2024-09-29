"""
Assignment 2: K-Means & DBSCAN Clustering
Name: Damien Liscio
Student #: 301237966

"""

#Import statements for required libraries
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

#1. Retrieve and load the Olivetti faces dataset
faces_dataset = fetch_olivetti_faces() #Load olivetti faces dataset
X = faces_dataset.data  #Assign image data
y = faces_dataset.target  #Assign image labels

#2. Split the data into a training set, a validation set, and a test set using stratified sampling to ensure that there are the same number of images per person in each set.
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=66) #Initialize Stratified Shuffle Split
for train_index, test_index in sss.split(X,y):    #Split data & labels into training and testing data
    X_train, X_test = X[train_index], X[test_index]  #Assign testing and training data appropriately after split
    y_train, y_test = y[train_index], y[test_index]  #Assign testing  and training labels appropriatley after split
    
for train_index, val_index in sss.split(X_train, y_train): #Split training data & labels into training and validation data
    X_train, X_val = X[train_index], X[val_index]  #Assign validation and training data appropriately after split
    y_train, y_val = y[train_index], y[val_index]  #Assign validation and training labels appropriatley after split
    
#Print sizes of training, test, and validation sets
print(f"Training Set Size: {X_train.shape[0]}/400, {(X_train.shape[0]/400)*100}%  \nTest Set Size: {X_test.shape[0]}/400, {(X_test.shape[0]/400)*100}% \nValidation Test Size: {X_val.shape[0]}/400, {(X_val.shape[0]/400)*100}%")

"""
I chose this split ratio for a couple of different reasons. The first one being an 80/20 split for the 
training and testing sets. This ensures a large enough training data for the algorithm to learn while 
also keeping a large enough portion (1/5) of the data to test with after training to see the learning success
of the algorithm. I then chose to do another 80/20 split for the training and validation set. I did this for
much the same reasons, with the training set being 64% of the original data sets size, giving it almost 2/3 of
the data to learn from. This leaves the validation set at a size of 16% the orignal data set size to fine
tune with.
"""

#3. Using k-fold cross validation, train a classifier to predict which person is represented in each picture, and evaluate it on the validation set.
#Initialize classifier
svm = SVC(kernel='linear', random_state=66)

#Do KFold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
cv_scores = cross_val_score(svm, X_train, y_train, cv=kfold)

#Fit classifier after validation
svm.fit(X_train, y_train)

#Predict with classifier using validation set
val_predict = svm.predict(X_val)

#Calculate accuracy of classifier on validation set
accuracy_val = accuracy_score(y_val, val_predict)*100
print(f"Validation Set Prediction Accuracy After KFold: {accuracy_val}%")

#4. Use K-Means to reduce the dimensionality of the set. Provide your rationale for the similarity measure used to perform the clustering.
#Find ideal number of clusters using silhouette method
#Range of possible clusters
range_n_clusters = [2,3,4,5,6]

#Loop through possible number of clusters
for n_clusters in range_n_clusters:
    
    #Initialize KMean with n_clusters value as loop value and predict
    kmeans = KMeans(n_clusters=n_clusters, random_state=66)
    y_pred = kmeans.fit_predict(X_train)
    
    #Get silhouette scores for training data
    silhouette_avg_score = silhouette_score(X_train, y_pred) 

    print(f"Number of Clusters: {n_clusters} \nAverage Silhouette Score:{silhouette_avg_score}")  #print results
    
    # Get silhouette samples
    sample_silhouette_values = silhouette_samples(X_train, y_pred)

    #Create a silhouette plot
    plt.figure(figsize=(10, 6))
    y_lower = 10  #Starting position for the first cluster

    for i in range(n_clusters):
        #Aggregate silhouette scores for samples belonging to cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[y_pred == i]

        #Sort the silhouette scores
        ith_cluster_silhouette_values.sort()

        #Determine the size of the cluster
        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        #Plot the silhouette scores for the cluster
        plt.fill_betweenx(np.arange(y_lower, y_lower + size_cluster_i),
                          0, ith_cluster_silhouette_values)

        #Label for the cluster
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

        #Increment the position for the next cluster
        y_lower += size_cluster_i

    #Make Silhouette Graph
    plt.title(f'Silhouette Plot for {n_clusters} Clusters')
    plt.xlabel('Silhouette Coefficient Values')
    plt.ylabel('Cluster Label')
    plt.axvline(x=silhouette_avg_score, color='red', linestyle='--')  # Average silhouette score line
    plt.yticks([])
    plt.xlim([-0.1, 1])
    plt.show()

"""
Based on the results of the silhouette method, the ideal number of clusters is 4.
"""
#Reduce the dimensionality of the set with n_clusters=4
kmeans_4c = KMeans(n_clusters=4, random_state=66)
X_train_clustered = kmeans_4c.fit_transform(X_train)

#5. Use the set from step (4) to train a classifier as in step (3)
#Create new classifier for clustered data
svm_clustered = SVC(kernel='linear', random_state=66)

#Get labels from clustered KMeans model & combine training data with cluster labels
y_pred_4c = kmeans_4c.labels_

#Do KFold cross validation
kfold_2 = KFold(n_splits=5, shuffle=True, random_state=66)
cv_scores_2 = cross_val_score(svm_clustered, X_train_clustered, y_pred_4c, cv=kfold_2)

#Fit classifier after validation
svm_clustered.fit(X_train_clustered, y_pred_4c)

#Prepare validation data by clustering 
kmeans_4c_val = KMeans(n_clusters=4, random_state=66)
X_val_clustered = kmeans_4c_val.fit_transform(X_val)
#Get labels from clustered validation
y_val_clustered = kmeans_4c_val.labels_

#Predict with classifier using validation set
X_val_predict_clustered = svm_clustered.predict(X_val_clustered)

#Calculate accuracy of classifier on validation set
accuracy_val_clustered = accuracy_score(y_val_clustered, X_val_predict_clustered)*100
print(f"Validation Set Prediction Accuracy After Clustering: {accuracy_val_clustered}%")

#6. Apply DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm to the Olivetti Faces dataset for clustering. Preprocess the images and convert them into feature vectors, then use DBSCAN to group similar images together based on their density. Provide your rationale for the similarity measure used to perform the clustering, considering the nature of facial image data.
#Preprocess data by converting into feature vectors
images = faces_dataset.images
n_samples, h, w = images.shape
X_db = images.reshape((n_samples, h*w)) 

#Apply DBSCAN
dbscan = DBSCAN(eps=7.3, min_samples=3, metric='euclidean')
dbscan_labels = dbscan.fit_predict(X_db)

#Print the clusters labels and indicies of core instances
print("Labels: ", dbscan.labels_)
print("Indices of the first 10 core instances: ", dbscan.core_sample_indices_[:10])
print("Core Samples", dbscan.components_)
    
"""
The rationale for the similarity measure used, in this case euclidean, was chosen because it handles
the high dimensional feature vectors of facial images well. Because the images are the same size and are
aligned, the euclidean distance is an effective measurement for the similarity between faces,
"""