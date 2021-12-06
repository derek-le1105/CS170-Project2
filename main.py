import os
import random
import time
import copy
import numpy as np

 

def main():     #csv file contains my name and small(56) and large(28) data set i should use        SMALL 86: 57 seconds to run, LARGE 27: 
    start_time = time.time()
    print("Welcome to the Feature Selection Algorithm")
    #fileName = "Data/" + input("Type in the name of the file you want to test: ")
    fileName = "Data/Ver_2_CS170_Fall_2021_LARGE_data__28.txt"
    #algDec = input("Type in the corresponding algorithm you want to run \n1) Forward Selection \n2) Backward Elimination\n")
    algDec = 1
    print(f"Running {fileName}")
    data = textfileToNPMatrix(fileName)
    if algDec == 1:
        a, b = feature_search_demo(data)
    print(f"Highest accuracy recorded was at: {a} with a set of: {b}")
    print(f"Program took {round(time.time() - start_time, 3)} seconds to run")

def textfileToNPMatrix(file):
    matrix = np.loadtxt(file)
    return matrix

def feature_search_demo(data):
    numOfInstances, numOfFeatures = data.shape  #numOfInstances is number of rows in np array, numOfFeatures is number of columns
    print(f"This dataset has {numOfFeatures - 1} features(excluding class attribute) with {numOfInstances} instances")
    highestAccuracy = 0
    indexAtHighest = 0
    current_set_of_features = []

    for i in range(1, numOfFeatures):
        '''if ((start_time - time.time()) > 100):
            break
        elif ((start_time - time.time()) % 60):
            print(start_time-time.time())'''
        print(f"\nOn the {i}th level of the search tree")
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for j in range(1, numOfFeatures):
            if j not in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, j)
                print(f"--Considering adding the {j} feature with accuracy of {accuracy}")
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = j
                if best_so_far_accuracy > highestAccuracy:
                    highestAccuracy = best_so_far_accuracy
                    indexAtHighest = i
        
        current_set_of_features.append(feature_to_add_at_this_level)
        print(f"At level {i}, I added feature {feature_to_add_at_this_level} to the current set with accuracy of {best_so_far_accuracy}")
    return highestAccuracy, current_set_of_features[:indexAtHighest]

def leave_one_out_cross_validation(data, current_set_of_features, feature_to_add):      #current set of features is a list of the features we want to test
    newData = copy.deepcopy(data)
    for i in range(newData.shape[0]):       #makes unwanted columns equal to 0
        for j in range(1, newData.shape[1]):
            if j not in current_set_of_features and j != feature_to_add:
                newData[i][j] = 0
    number_correctly_classified = 0
    for i in range(newData.shape[0]):           #goes through each instance(row) in data
        object_to_classify = newData[i][1:]
        label_object_to_classify = newData[i][0]       #gets the label classifier in the first column         

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for j in range(newData.shape[0]):        #iterate over every other instance besides ith instance
            if j != i:
                #print(f"Ask if {i} is nearest neighbor with {j}")
                distance = np.sqrt(sum(np.square(object_to_classify - newData[j][1:])))
                if distance < nearest_neighbor_distance:                                            
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = j
                    nearest_neighbor_label = newData[nearest_neighbor_location][0]

        #print(f"Object {i} is class {label_object_to_classify}")
        #print(f"Its nearest neighbor is {nearest_neighbor_location}, which is in class {nearest_neighbor_label}")
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified +=1
    accuracy = round((number_correctly_classified / data.shape[0]) * 100, 3)
    del newData
    return accuracy
main()