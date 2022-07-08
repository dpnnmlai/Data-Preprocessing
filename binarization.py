from sklearn.preprocessing import Binarizer, binarize
import pandas
import numpy


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"  # url of the data

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

binarizer = Binarizer(threshold=0.0).fit(X)  # binarize the data
binarX = binarizer.transform(X)  # binarize the data

numpy.set_printoptions(precision=3)  # print the data
print(binarX[0:6, :])
