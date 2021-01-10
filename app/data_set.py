import pandas as pd
from sklearn.model_selection import train_test_split

data_set = []
train_set = []
test_set = []


def load_dataset(*, path, test_size=0.2, class_name="Class"):
    global data_set
    global train_set
    global test_set
    data_set = pd.read_csv(path, sep=";")
    df1 = data_set.pop(class_name)
    data_set[class_name] = df1
    train_set, test_set = train_test_split(data_set, test_size=test_size)


def resplit_dataset(*, test_size=0.2):
    global train_set
    global test_set
    train_set, test_set = train_test_split(data_set, test_size=test_size)
