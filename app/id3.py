import pandas as pd
import numpy as np
from numpy import log2 as log
import pdb


eps = np.finfo(float).eps


def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique() #Get classes in dataset
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy


def find_entropy_attribute(df,attribute):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    target_variables = df[Class].unique() #Get classes in dataset
    variables = df[attribute].unique() #Get attribute possible values
    entropy2 = 0
    for variable in variables: #For each possible value
        entropy = 0
        for target_variable in target_variables: #For each class
            num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
            fraction2 = den/len(df)
            entropy2 += -fraction2*entropy
            return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        #         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]


def get_subtable(df, node,value):
    return df[df[node] == value].reset_index(drop=True)

def buildAttValues(df):
    att_values = {}
    for column in df:
        att_values[column] = []
        for value in df[column]:
            att_values[column].append(value) if value not in att_values[column] else att_values[column]
    return att_values


def buildTreeWrapper(*, train_set, original_data_set):
    att_values = buildAttValues(original_data_set)
    return buildTree(train_set, att_values)


def buildTree(df,att_values,tree=None):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name

    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = find_winner(df)

    #Create an empty dictionary to create tree
    if tree is None:
        tree={}
        tree[node] = {}

    #We make loop to construct a tree by calling this function recursively.
    #In this we check if the subset is pure and stops if it is pure.

    for value in att_values[node]:
        subtable = get_subtable(df,node,value)

        # if there no such value for this attribute assign most frequent class
        if len(subtable) == 0:
            most_freq_class = df[Class].value_counts().idxmax()
            tree[node][value] = most_freq_class
            continue
         
        clValue,counts = np.unique(subtable[Class],return_counts=True)

        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable, att_values) #Calling the function recursively

    return tree

def test(tree, df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    len = df.shape[0]
    correct = 0
    mse = 0
    me = 0
    for index, row in df.iterrows():
        predicted_class = getClass(tree, row)
        real_class = row[Class]
        if predicted_class == real_class:
            correct += 1
        mse += (predicted_class - real_class)**2
        me += abs(predicted_class - real_class)
    mse = mse / len
    me = me / len
    return correct / len, mse, me


def find_closest_value(a_list, given_value):
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    if (type(given_value) is not str):
        return min(a_list, key=absolute_difference_function)
    return a_list[0]

def getClass(tree, row):
    attr = list(tree.keys())[0]
    subtree = tree[attr][row[attr]]
    while type(subtree) is dict:
        attr = list(subtree.keys())[0]
        subtree = subtree[attr][row[attr]]
    return subtree
