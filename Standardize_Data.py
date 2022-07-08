from sklearn.preprocessing import StandardScaler
import pandas
import numpy


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = [
    "preg",
    "plas",
    "pres",
    "skin",
    "test",
    "mass",
    "pedi",
    "age",
    "class",
]  # names of the features

dataframe = pandas.read_csv(url, names=names)  # read the data
pandas.array = dataframe.values  # convert to numpy array

X = pandas.array[:, 0:8]  # features
Y = pandas.array[:, 8]  # target

scaler = StandardScaler().fit(X)  # scale the data
rescalerX = scaler.transform(X)  # rescale the data

numpy.set_printoptions(precision=3)  # print the data
print(rescalerX[0:5, :])
