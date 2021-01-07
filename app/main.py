import pandas as pd
import numpy as np
eps = np.finfo(float).eps
from numpy import log2 as log
from pprint import pprint
from sklearn.model_selection import train_test_split
import pdb

DIVORCE_DATA_SET = "../data/divorce.csv"

df = pd.read_csv(DIVORCE_DATA_SET, sep=";")
train_set, test_set = train_test_split(df, test_size=0.2)


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


def buildTree(df,tree=None):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name

    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = find_winner(df)

    #Get distinct values of that attribute
    attValue = np.unique(df[node])

    #Create an empty dictionary to create tree
    if tree is None:
        tree={}
        tree[node] = {}

   #We make loop to construct a tree by calling this function recursively.
    #In this we check if the subset is pure and stops if it is pure.

    for value in attValue:

        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable[Class],return_counts=True)

        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable) #Calling the function recursively

    return tree

def test(tree, df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    len = df.shape[0]
    print(len)
    correct = 0
    for index, row in df.iterrows():
        if getClass(tree, row) == row[Class]:
            correct += 1
    print("accuracy: " + str(correct / len))


def find_closest_value(a_list, given_value):
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    return min(a_list, key=absolute_difference_function)


def getClass(tree, row):
    attr = list(tree.keys())[0]
    subtree = tree[attr][row[attr]]
    while type(subtree) is dict:
        attr = list(subtree.keys())[0]
        try:
            subtree = subtree[attr][row[attr]]
        except KeyError as e:
            keys = list(subtree[attr].keys())
            closest_key = find_closest_value(keys, row[attr])
            print("There was a key error on key: " + str(row[attr]))
            print("Available keys: " + str(keys))
            print("Choosing closest key: " + str(closest_key))
            subtree = subtree[attr][closest_key]
            continue
    return subtree

t = buildTree(train_set)
pprint(t)
test(t, test_set)
