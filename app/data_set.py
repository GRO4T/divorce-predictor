import pandas as pd
from sklearn.model_selection import train_test_split

df = []
train_set = []
test_set = []

def load_dataset(*, path, test_size=0.2, a_class="Class"):
    global df
    global train_set
    global test_set
    df = pd.read_csv(path, sep=";")
    df1 = df.pop(a_class)
    df[a_class] = df1
    train_set, test_set = train_test_split(df, test_size=test_size)

def resplit_dataset(*, test_size=0.2):
    global train_set
    global test_set
    train_set, test_set = train_test_split(df, test_size=test_size)

