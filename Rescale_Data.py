import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler


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
array = dataframe.values  # convert to numpy array


X = array[:, 0:8]  # features
Y = array[:, 8]  # target

scaler = MinMaxScaler(feature_range=(0, 1))  # scale the data

rescale_Data = scaler.fit_transform(X)  # rescale the data


numpy.set_printoptions(precision=3)  # print the data
print(rescale_Data[0:5, :])
