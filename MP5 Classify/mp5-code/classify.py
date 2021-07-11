# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np
import math 
import queue


def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    #print(train_set[0])
    W = [0]*len(train_set[0])
    #print(type(W))
    b = 0 

    # iteration until max_iter times, 
    for i in range(max_iter):
        for x, y in zip(train_set, train_labels):
            if (np.dot(W, x) + b) > 0 :
                y_hat = 1
            else:
                y_hat = 0

            W += learning_rate * (y - y_hat) * x 
            b += learning_rate * (y - y_hat) 

    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set

    # return variable
    dev_labels = []

    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    
    for feature_vector in dev_set:
        if (np.dot(W, feature_vector) + b) > 0:
            label = 1  
        else:
            label = 0
        dev_labels.append(label)
        
    return dev_labels  

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    knn_labels = []

    for dev in dev_set:

        neighbor = queue.PriorityQueue()
        class1_sum = 0
        class2_sum = 0

        for i, train in enumerate(train_set):
            #print(i, train)

            #find Euclidean Distance between train and data 
            Edist = math.sqrt(np.sum(pow(train - dev, 2)))

            #save all classified datas into the neighbor queue, but to break ties, we use negative labels here
            neighbor.put((Edist*-1, train_labels[i]))
            
            # we are trying to find maximum values only for k values so taking out unnecessary queue items 
            while neighbor.qsize() > k:
                #print(neighbor.get())
                neighbor.get()
            
        while neighbor.qsize():
            #print("these are neighbor.get()[1]:", neighbor.get()[1])
            if (neighbor.get()[1]):
                class1_sum += 1
            else:
                class2_sum += 1

        # Determine whether the class falls into class I or class II        
        if class1_sum > class2_sum:
            knn_labels.append(True)
        else:
            knn_labels.append(False)

    #print("final knn classfied labels:", knn_labels)
    return knn_labels
