"""
COMP257 Sec.402 Unsupervised & Reinforcement Learning
Assignment 3: Hierarchical Clustering
Name: Damien Liscio
Student #: 301237966
Due: Sunday, October 13th, 2024

"""

#Import statements for required libraries
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import minkowski
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
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
print(f"Training Set Size: {X_train.shape[0]}/400, {(X_train.shape[0]/400)*100}%  \nTest Set Size: {X_test.shape[0]}/400, {(X_test.shape[0]/400)*100}% \nValidation Test Size: {X_val.shape[0]}/400, {(X_val.shape[0]/400)*100}%\n")

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
print(f"Validation Set Prediction Accuracy After KFold: {accuracy_val}%\n")

#4. Using either Agglomerative Hierarchical Clustering (AHC) or Divisive Hierarchical Clustering (DHC) and using the centroid-based clustering rule, reduce the dimensionality of the set by using the given similarity measures.
#Reduce the dimensionality of the dataset using PCA
pca = PCA() #Create PCA Instance
pca.fit(X_train) #Fit the training data
#cumulative sum of explained variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train) #Transform training data
print(f"n_components: {pca.n_components_}\n") #Print the number of components

similarity_measures = ["euclidean", "minkowski", "cosine"] #List of similarity measures
clusters_range = range(90, 110) #List of n_clusters
#Define AHC for each similarity measure
for similarity in similarity_measures:
    #Create dictionary for silhouette scores
    silhouette_scores = {similarity: [] for similarity in similarity_measures}
    for n_cluster in clusters_range:
        if similarity == "minkowski":
            AHC_clf = AgglomerativeClustering(n_clusters=n_cluster, metric='precomputed', linkage='average')
            distance_matrix = metrics.pairwise_distances(X_train_reduced, metric=lambda u, v: minkowski(u, v, p=2)) #Compute Minkowski distance matrix
            AHC_clf.fit(distance_matrix) #Fit the data with the minkowski matrix
            data_labels = AHC_clf.labels_  #Get data labels
            silhouette_score = metrics.silhouette_score(distance_matrix, data_labels, metric='precomputed')
        else:
            AHC_clf = AgglomerativeClustering(n_clusters=n_cluster, metric=similarity, linkage='average')
            AHC_clf.fit(X_train_reduced) #Fit the reduced training data with the classifier
            data_labels = AHC_clf.labels_  #Get data labels
            silhouette_score = metrics.silhouette_score(X_train_reduced, data_labels)
    
        data_labels = AHC_clf.labels_  #Get data labels
        silhouette_score = metrics.silhouette_score(X_train_reduced, data_labels)  #get silhouette score
        silhouette_scores[similarity].append(silhouette_score) #Append silhouette score to dictionary
        print(f"Silhouette Score For {similarity} & {n_cluster} clusters: {silhouette_score}\n")
     
    #Make plot figure for silhouette scores
    plt.figure(figsize=(10,6))
    plt.plot(list(clusters_range), silhouette_scores[similarity], label=f'{similarity} similarity')
    #Set plot labels and title
    plt.title(f'Silhouette Scores for Different Cluster Sizes and {similarity} similarity')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.legend()
    #Show the plot
    plt.show()
    
    #Make the plot figure for dendrogram
    plt.figure(figsize=(10,6))
    #Set plot labels 
    plt.title(f"Hierarchical Clustering Dendrogram for {similarity}")
    plt.xlabel("Data Labels")
    plt.ylabel("Distance")
    dendro = sch.dendrogram(sch.linkage(X_train_reduced, "average", metric=similarity),
                            labels=data_labels,
                            leaf_rotation=90,
                            leaf_font_size=8,
                            show_contracted=True)
    #Show the plot
    plt.show()
    
#5. Use the silhouette score approach to choose the number of clusters for 4(a), 4(b), and 4(c).
#Use silhouette scores to make classifiers with ideal number of clusters and fit & predict 
#Given the silhouette scores, the ideal number of clusters is somewhere in the high 90's/low 100's. This is
#becuase this is where the silhouette scores are the highest and closest to +1 before dropping
#back down towards 0.

#Euclidean 
AHC_clf_e = AgglomerativeClustering(n_clusters=105, metric='euclidean', linkage='average')
euclidean_predict = AHC_clf_e.fit_predict(X_train_reduced)  #Fit with training data
data_labels_euclidean = AHC_clf_e.labels_ #Get euclidean labels

AHC_clf_m = AgglomerativeClustering(n_clusters=105, metric='precomputed', linkage='average')
distance_matrix = metrics.pairwise_distances(X_train_reduced, metric=lambda u, v: minkowski(u, v, p=2)) #Compute Minkowski distance matrix
minkowski_predict = AHC_clf_m.fit_predict(distance_matrix)  #Fit with minkowski matrix
data_labels_minkowski = AHC_clf_m.labels_ #Get minkowski labels

#Cosine 
AHC_clf_c = AgglomerativeClustering(n_clusters=97, metric='cosine', linkage='average')
cosine_predict = AHC_clf_c.fit_predict(X_train_reduced)  #Fit with training data
data_labels_cosine = AHC_clf_c.labels_ #Get cosine labels

#6. Use the set from (4(a), 4(b), or 4(c)) to train a classifier as in (3) using k-fold cross validation.
svm_clustered = SVC(kernel='linear', random_state=66) #Initialize classifier

#Do KFold with cross validation
kfold_clustered = KFold(n_splits=5, shuffle=True, random_state=66)
cv_scores_clustered = cross_val_score(svm_clustered, X_train_reduced, data_labels_euclidean)

#Train the classifier 
svm_clustered.fit(X_train_reduced, data_labels_euclidean)

#Reduce dimensionality of validation set
X_val_reduced = pca.transform(X_val)

#Predit using validation set
val_predict_clustered = svm_clustered.predict(X_val_reduced)

#Print cluster labels
print(f"Cluster Labels Predicted for Validation Set:{val_predict_clustered}")



