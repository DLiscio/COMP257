# -*- coding: utf-8 -*-
"""

COMP257 Sec.402 Unsupervised & Reinforcement Learning
Assignment 5: Autoencoders
Name: Damien Liscio
Student #: 301237966
Due: Monday, November 11th, 2024

"""

#Import statements for required libraries
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import ReLU, BatchNormalization
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#1. Retrieve data and split into training, validation and testing
faces_dataset = fetch_olivetti_faces() #Load olivetti faces dataset
X = faces_dataset.data  #Assign image data
y = faces_dataset.target  #Assign image labels

#Split the data into a training set, a validation set, and a test set using stratified sampling to ensure that there are the same number of images per person in each set.
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=66) #Initialize Stratified Shuffle Split
for train_index, test_index in sss.split(X,y):    #Split data & labels into training and testing data
    X_train, X_test = X[train_index], X[test_index]  #Assign testing and training data appropriately after split
    y_train, y_test = y[train_index], y[test_index]  #Assign testing  and training labels appropriatley after split
    
for train_index, val_index in sss.split(X_train, y_train): #Split training data & labels into training and validation data
    X_train, X_val = X[train_index], X[val_index]  #Assign validation and training data appropriately after split
    y_train, y_val = y[train_index], y[val_index]  #Assign validation and training labels appropriatley after split
    
#Print sizes of training, test, and validation sets
print(f"Training Set Size: {X_train.shape[0]}/400, {(X_train.shape[0]/400)*100}%  \nTest Set Size: {X_test.shape[0]}/400, {(X_test.shape[0]/400)*100}% \nValidation Test Size: {X_val.shape[0]}/400, {(X_val.shape[0]/400)*100}%\n")

#2. Apply PCA on the training data, preserving 99% of the variance, to reduce the dataset's dimensionality
scaler = StandardScaler()  # Initialize Scaler
X_train = scaler.fit_transform(X_train) #Transform training data
X_val = scaler.transform(X_val) #Transform validation data
X_test = scaler.transform(X_test) #Transform testing data
print(f"Training Data Dimensions: {X_train.shape[1]}")
pca = PCA(0.99, whiten=True) #Initialize PCA, preserving 99% of the datas variance
X_train = pca.fit_transform(X_train)  #Apply PCA to training data
X_test = pca.transform(X_test) #Transform test data
X_val = pca.transform(X_val) #Transform validation data
print(f"Reduced Training Data Dimensions: {X_train.shape[1]}")  #Print reduced training data shape

#3a. Use k-fold cross validation to fine tune an autoencoder
#Define hyperparameters 
learning_rates = [0.01, 0.001, 0.0001] #Learning rate options
regularizations = [0.01, 0.001, 0.0001]  #Regularization parameter options
hidden_unit_setup = [  #Hidden unit options for hidden layers
    [128, 64, 128],
    [64, 32, 64],
    [32, 16, 32]
]
k_fold = KFold(n_splits=5, shuffle=True, random_state=66) #Initialize k-fold cross validation
best_parameters = None #Store best parameters, initialize with none
best_score = np.inf #Variable to store best cv score when comparing various configurations
for rate in learning_rates: #Loop through learning rates
    for regularization in regularizations: #Loop through regularizations
        for hidden_units in hidden_unit_setup: #Loop through hidden unit options
            cv_scores = [] #Initialize empty list for cv scores
            for train_index, val_index in k_fold.split(X_train):  #Split data, one for each fold
                X_tr, X_val = X_train[train_index], X_train[val_index] #Select appropriate data for given fold
                input_size = X_train.shape[1] #Get input layer size
                input_img = Input(shape=(input_size,)) #Input layer
                hidden_layer1 = Dense(hidden_units[0], kernel_regularizer=l2(regularization))(input_img) #Top Hidden Layer 1
                hidden_layer1 = BatchNormalization()(hidden_layer1) #Batch Normalization
                hidden_layer1 = ReLU()(hidden_layer1) #Regularization
                code = Dense(hidden_units[1], kernel_regularizer=l2(regularization))(hidden_layer1) #Central Layer 2
                code = BatchNormalization()(code) #Batch Normalization
                code = ReLU()(code) #Regularization
                hidden_layer3 = Dense(hidden_units[2], kernel_regularizer=l2(regularization))(code) #Top Hidden Layer 3
                hidden_layer1 = BatchNormalization()(hidden_layer3) #Batch Normalization
                hidden_layer1 = ReLU()(hidden_layer3) #Regularization
                output_img = Dense(input_size, activation='sigmoid')(hidden_layer3) #Output Layer
                autoencoder = Model(input_img,output_img) #Autoencoder Model
                autoencoder.compile(optimizer=Adam(learning_rate=rate), loss='mean_squared_error') #Compile the model
                autoencoder.fit(X_tr, X_tr, epochs=50, batch_size=64, validation_data=(X_val, X_val)) #Train the autoencoder
                validation_loss = autoencoder.evaluate(X_val, X_val) #Evaluate using validation set
                cv_scores.append(validation_loss)  #Add cv score for current fold
            average_cv_score = np.mean(cv_scores) #Get average of all cv scores
            if average_cv_score < best_score: #Conditional to assign best_score as lowest score
                best_score = average_cv_score #Assign if it passes conditional
                best_parameters = {  #Assign best parameters
                    'learning_rate': rate,
                    'regularization': regularization,
                    'hidden_units': hidden_units
                }
print(f"Best Hyperparameters: {best_parameters} \nVal Loss Score: {best_score}")  #Print final best parameters 

#4. Run the best model with the test set and display the original image and reconstructed image
best_rate = best_parameters['learning_rate']  #Get the best learning rate
best_regularization = best_parameters['regularization'] #Get the best regularization
best_hidden_units = best_parameters['hidden_units'] #Get the best number of hidden units
#Train final model with best parameters
input_size = X_train.shape[1] #Get input layer size
input_img = Input(shape=(input_size,)) #Input layer
hidden_layer1 = Dense(hidden_units[0], kernel_regularizer=l2(regularization))(input_img) #Top Hidden Layer 1
hidden_layer1 = BatchNormalization()(hidden_layer1) #Batch Normalization
hidden_layer1 = ReLU()(hidden_layer1) #Regularization
code = Dense(hidden_units[1], kernel_regularizer=l2(regularization))(hidden_layer1) #Central Layer 2
code = BatchNormalization()(code) #Batch Normalization
code = ReLU()(code) #Regularization
hidden_layer3 = Dense(hidden_units[2], kernel_regularizer=l2(regularization))(code) #Top Hidden Layer 3
hidden_layer1 = BatchNormalization()(hidden_layer3) #Batch Normalization
hidden_layer1 = ReLU()(hidden_layer3) #Regularization
output_img = Dense(input_size, activation='sigmoid')(hidden_layer3) #Output Layer
autoencoder = Model(input_img,output_img) #Autoencoder Model
autoencoder.compile(optimizer=Adam(learning_rate=best_rate), loss='mean_squared_error') #Compile the model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=64, validation_data=(X_val, X_val)) #Train the autoencoder
reconstructed_imgs = autoencoder.predict(X_test) #Make predictions for test set
num_imgs = 5  #Set the number of images to be displayed
plt.figure(figsize=(10,4)) #Create a plot to hold each pair of images
for i in range(num_imgs):  #Create images in set range
    ax = plt.subplot(2, num_imgs, i+ 1)  #Create first subplot
    original_img = pca.inverse_transform(X_test[i].reshape(1, -1))  #Inverse PCA transformation
    original_img = scaler.inverse_transform(original_img) #Inverse Scaler transformation
    plt.imshow(original_img.reshape(64,64))  #Reshape the image and plot
    plt.title("Original")  #Label for images set
    plt.axis("off") #Turn off plot axis
    
    ax = plt.subplot(2, num_imgs, i+1+num_imgs) #Create second subplot
    reconstructed_img = pca.inverse_transform(reconstructed_imgs[i].reshape(1, -1))  #Inverse PCA transformation
    reconstructed_img = scaler.inverse_transform(reconstructed_img) #Inverse Scaler transformation
    plt.imshow(reconstructed_img.reshape(64,64))  #Reshape the image and plot
    plt.title("Reconstructed") #Label for images set
    plt.axis("off") #Turn off plot axis
        


