import pandas as pd

df = pd.read_csv("data/students/student-mat.csv", sep=";", index_col=0)
print(df.keys())
df.to_csv("data/students/preprocessed/student-mat.csv", sep=";", index=False)

df = pd.read_csv("data/students/student-por.csv", sep=";", index_col=0)
df.to_csv("data/students/preprocessed/student-por.csv", sep=";", index=False)
