import os
import random
import numpy as np

def main():     #csv file contains my name and small(56) and large(28) data set i should use
    print("Welcome to the Feature Selection Algorithm")
    #fileName = "Data/" + input("Type in the name of the file you want to test: ")
    fileName = "Data/CS170_Fall_2021_SMALL_data__56.txt"
    #algDec = input("Type in the corresponding algorithm you want to run \n1) Forward Selection \n2) Backward Elimination\n")
    data = textfileToNPMatrix(fileName)
    feature_search_demo(data)

def textfileToNPMatrix(file):
    matrix = np.loadtxt(file)
    return matrix

def feature_search_demo(data):
    numOfInstances, numOfFeatures = data.shape  #numOfInstances is number of rows in np array, numOfFeatures is number of columns
    print(f"This dataset has {numOfFeatures - 1} features(excluding class attribute) with {numOfInstances} instances")

    current_set_of_features = []

    for i in range(0, numOfFeatures):
        #print(f"\nOn the {i+1}th level of the search tree")
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for j in range(0, numOfFeatures):
            if j not in current_set_of_features:
                #print(f"--Considering adding the {j+1} feature")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, j+1)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = j
        
        current_set_of_features.append(feature_to_add_at_this_level)
        print(f"At level {i+1}, I added feature {feature_to_add_at_this_level + 1} to the current set")

def leave_one_out_cross_validation(data, current_set_of_features, feature_to_add):      #current set of features is a list of the features we want to test
    number_correctly_classified = 0
    for i in range(data.shape[0]):           #goes through each instance(row) in data
        #object_to_classify = (data[i][current_set_of_features], data[i])            #row with current_set_of_features

        if not current_set_of_features:
            object_to_classify = data[i][i+1]
        else:
            object_to_classify = data[i][current_set_of_features]   #gets ith row and current_set_of_features
        label_object_to_classify = data[i][0]       #gets the label classifier in the first column

        nearest_neighbor_distance = 999
        nearest_neighbor_location = 0

        for j in range(data.shape[0]):        #iterate over every other instance besides ith instance
            if j != i:
                #print(f"Ask if {i} is nearest neighbor with {j}")
                distance = np.sqrt(sum((object_to_classify - data[j][feature_to_add])**2))          #ASK PROFESSOR IF LOGIC REGARDING C_S_O_F AND F_T_A is correct
                if distance < nearest_neighbor_distance:                                            #if ObjToClas and distance is correct
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = j
                    nearest_neighbor_label = data[nearest_neighbor_location][0]

        #print(f"Object {i} is class {label_object_to_classify}")
        #print(f"Its nearest neighbor is {nearest_neighbor_location}, which is in class {nearest_neighbor_label}")
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified +=1
    accuracy = number_correctly_classified / data.shape[0]
    return accuracy
main()