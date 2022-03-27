from posixpath import split

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

iris_data = pd.read_csv("Iris.csv")
iris_data = iris_data.dropna(how="any")
y = iris_data.pop("Species")
X = iris_data
y = LabelEncoder().fit_transform(y)
y = pd.DataFrame(y, columns=["species"])
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

train_X.to_csv("train_X.csv", index=False)
test_X.to_csv("test_X.csv", index=False)
train_y.to_csv("train_y.csv", index=False)
test_y.to_csv("test_y.csv", index=False)
