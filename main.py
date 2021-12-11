import time
import copy
import numpy as np
start_time = time.time()
backwardSelected = False

def main():     #csv file contains my name and small(56) and large(28) data set i should use        SMALL 86: 57 seconds to run, LARGE 27: 
    global backwardSelected
    t = time.localtime()                #returns local time
    current_time = time.strftime("%H:%M:%S", t)
    print("Welcome to the Feature Selection Algorithm")
    datasetSize = input("Do you want to test the 'LARGE' or 'SMALL' dataset? Input large/small in CAPS please: ").upper()
    datasetNumber = input("Which dataset do you want to test? Please input from 1-100 for the desired dataset: ")
    fileName = "Data/Ver_2_CS170_Fall_2021_" + datasetSize + "_data__" + datasetNumber + ".txt"
    algDec = int(input("Do you want to run Forward Selection or Backward Elimination? Enter 1 or 2 for the respective algorithm: "))
    print(f"Running {fileName} at {current_time}")
    data = textfileToNPMatrix(fileName)
    backwardSelected = (False, True)[algDec == 2]
    if backwardSelected:
        a, b = backwardElimination(data)
    else:
        a, b = feature_search_demo(data)
    print(f"Highest accuracy recorded was at: {a} with a set of: {b}")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f"Program took {round(time.time() - start_time, 3)} seconds to run and finished at {current_time}")

def textfileToNPMatrix(file):
    matrix = np.loadtxt(file)               #converts textfile into a numpy matrix
    return matrix

def backwardElimination(data):
    timeExceeded = False
    lowestAccuracyFound = 100
    indexAtHighest = 0
    numOfInstances, numOfFeatures = data.shape
    print(f"This dataset has {numOfFeatures - 1} features(excluding class attribute) with {numOfInstances} instances")
    current_set_of_features = [x for x in range(1, numOfFeatures)]
    class1Count, class2Count = 0, 0
    for i in range(numOfInstances):
        if data[i][0] == 1:
            class1Count += 1
        elif data[i][0] == 2:
            class2Count += 1
    
    defaultRate = round((max(class1Count, class2Count) / numOfInstances) * 100, 3)
    print(f"Starting from the empty set, our default rate is {defaultRate}")
    
    for i in range(1, numOfFeatures):
        print(f"\nOn the {i}th level of the search tree")
        feature_to_remove_at_this_level = []
        best_so_far_accuracy = 0
        if timeExceeded:
            break

        for j in range(1, numOfFeatures):
            if timeExceeded:
                break
            if j in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, j)
                print(f"--Removing feature {j} from current set yields accuracy of {accuracy} ")
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = j
                if best_so_far_accuracy < lowestAccuracyFound:
                    lowestAccuracyFound = best_so_far_accuracy
                    indexAtHighest = i
                #timeElapsed = time.time() - start_time
                #if timeElapsed > 7200:                     #if the time to run program exceeds 2 hours, stop running
                    #timeExceeded = True
        
        current_set_of_features.remove(feature_to_remove_at_this_level)
        print(f"At level {i}, I removed feature {feature_to_remove_at_this_level} from the current set to achieve an accuracy of {best_so_far_accuracy}")
        
    return best_so_far_accuracy, feature_to_remove_at_this_level

def feature_search_demo(data):
    timeExceeded = False
    numOfInstances, numOfFeatures = data.shape  #numOfInstances is number of rows in np array, numOfFeatures is number of columns
    print(f"This dataset has {numOfFeatures - 1} features(excluding class attribute) with {numOfInstances} instances")
    highestAccuracyFound = 0
    indexAtHighest = 0
    current_set_of_features = []
    accDecreasing = False

    class1Count, class2Count = 0, 0
    for i in range(numOfInstances):
        if data[i][0] == 1:
            class1Count += 1
        elif data[i][0] == 2:
            class2Count += 1
    
    defaultRate = round((max(class1Count, class2Count) / numOfInstances) * 100, 3)
    print(f"Starting from the empty set, our default rate is {defaultRate}")

    for i in range(1, numOfFeatures):
        print(f"\nOn the {i}th level of the search tree")
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0
        if timeExceeded:
            break

        for j in range(1, numOfFeatures):
            if timeExceeded:                    #if the time to run program exceeds 2 hours, stop running
                break

            if j not in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, j)
                print(f"--Considering adding feature {j} with accuracy of {accuracy}")
                timeElapsed = time.time() - start_time
                #if timeElapsed > 7200:         #if the time to run program exceeds 2 hours, stop running
                    #timeExceeded = True
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = j
                if best_so_far_accuracy > highestAccuracyFound:
                    highestAccuracyFound = best_so_far_accuracy
                    indexAtHighest = i
        
        current_set_of_features.append(feature_to_add_at_this_level)
        print(f"At level {i}, I added feature {feature_to_add_at_this_level} to the current set with accuracy of {best_so_far_accuracy}")
        '''if (accDecreasing and highestAccuracyFound > best_so_far_accuracy):      #called before the next if statement so if algorithm notices accuracy decreasing, we run one more time 
            break                                                                                   #to see if there is a local maxima 
        if(highestAccuracyFound > best_so_far_accuracy):                                                 #when we first notice the accuracy decreasing, give a warning to user 
            accDecreasing = True
            print("Warning! Accuracy is beginning to decrease so we'll continue search in case of local maxima!")'''
    return highestAccuracyFound, current_set_of_features[:indexAtHighest]

def leave_one_out_cross_validation(data, current_set_of_features, feature_to_add):      #current set of features is a list of the features we want to test
    newData = setColsToZero(data, current_set_of_features, feature_to_add)
    number_correctly_classified = 0
    for i in range(newData.shape[0]):           #goes through each instance(row) in data
        object_to_classify = newData[i][1:]
        label_object_to_classify = newData[i][0]       #gets the label classifier in the first column         

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for j in range(newData.shape[0]):        #iterate over every other instance besides ith instance
            if j != i:
                distance = np.sqrt(sum(np.square(object_to_classify - newData[j][1:])))
                if distance < nearest_neighbor_distance:                                            
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = j
                    nearest_neighbor_label = newData[nearest_neighbor_location][0]
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified +=1
    accuracy = round((number_correctly_classified / data.shape[0]) * 100, 3)
    return accuracy

def setColsToZero(data, current_set_of_features, feature_to_add):
    newData = copy.deepcopy(data)           #make a copy of the dataset that we can manipulate without changing the original dataset
    if backwardSelected:
        for i in range(newData.shape[0]):       #makes unwanted columns equal to 0
            for j in range(1, newData.shape[1]):
                if (j in current_set_of_features and j == feature_to_add) or j not in current_set_of_features:
                    newData[i][j] = 0
    else:
        for i in range(newData.shape[0]):       #makes unwanted columns equal to 0
                for j in range(1, newData.shape[1]):
                    if j not in current_set_of_features and j != feature_to_add:
                        newData[i][j] = 0
    return newData

main()

