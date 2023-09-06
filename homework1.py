#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse
import numpy as np

STEP_SIZE = 0.0001
MAX_ITERATIONS = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(num_iterations):
        gradient = np.dot(X.T, sigmoid(np.dot(X, theta)) - y) / m
        theta -= learning_rate * gradient
    return theta

def predict(X, theta):
    return np.round(sigmoid(np.dot(X, theta)))

def load_data(data_file, label_file, max_word_id=None):
    data = np.loadtxt(data_file, dtype=int)
    
    if label_file:  # Only load labels if label file is provided.
        labels = np.loadtxt(label_file, dtype=int)
    else:
        labels = None
    
    if not max_word_id:
        max_word_id = np.max(data[:, 1])
    
    max_doc_id = np.max(data[:, 0])
    
    matrix_data = np.zeros((max_doc_id, max_word_id))
    for row in data:
        matrix_data[row[0]-1, row[1]-1] = row[2]
    
    return matrix_data, labels

def process_data(X, y=None):
    # Add intercept term
    intercept = np.ones((X.shape[0], 1))
    X = np.hstack((intercept, X))
    
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homework 1", 
                                     epilog="CSCI 4360/6360 Data Science II: Fall 2023")
    parser.add_argument("train_data", type=str, help="File containing training data")
    parser.add_argument("train_label", type=str, help="File containing training labels")
    parser.add_argument("test_data", type=str, help="File containing testing data")
    args = parser.parse_args()

    # Load and process training data
    X_train, y_train = process_data(*load_data(args.train_data, args.train_label))

    max_word_id = X_train.shape[1] - 1  # Subtracting 1 for the intercept term
    
    # Train the logistic regression model
    theta = logistic_regression(X_train, y_train, STEP_SIZE, MAX_ITERATIONS)

    # Load and process test data (ensure the test data has the same number of columns as the training data)
    X_test, _ = process_data(*load_data(args.test_data, None, max_word_id=max_word_id))

    predictions = predict(X_test, theta)
    print(predictions)



# In[ ]:





# In[ ]:





# In[ ]:




