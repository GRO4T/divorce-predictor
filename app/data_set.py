import pandas as pd
from sklearn.model_selection import train_test_split

data_set = []
train_set = []
prune_set = []
test_set = []


def load_dataset(*, path, test_size=0.2, prune_size=0.2, class_name="Class"):
    global data_set
    global train_set
    global prune_set
    global test_set
    data_set = pd.read_csv(path, sep=";")
    df1 = data_set.pop(class_name)
    data_set[class_name] = df1
    temp_set, test_set = train_test_split(data_set, test_size=test_size)
    train_set, prune_set = train_test_split(temp_set, test_size=prune_size)


def resplit_dataset(*, test_size=0.2, prune_size=0.2):
    global train_set
    global prune_set
    global test_set
    temp_set, test_set = train_test_split(data_set, test_size=test_size)
    train_set, prune_set = train_test_split(temp_set, test_size=prune_size)
