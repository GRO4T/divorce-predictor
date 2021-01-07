import pandas as pd

df = pd.read_csv("data/students/student-mat.csv", sep=",")
df.to_csv("data/students/preprocessed/student-mat.csv", sep=";")

df = pd.read_csv("data/students/student-por.csv", sep=",")
df.to_csv("data/students/preprocessed/student-por.csv", sep=";")
