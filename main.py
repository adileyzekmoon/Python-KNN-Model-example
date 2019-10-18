import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy

# read in data file
data = pandas.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clss = le.fit_transform(list(data["class"]))

predict = "class"
x = list(zip(buying,maint, door, persons, lug_boot, safety))
y = list(clss)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(x_train, y_train)
acc = KNN_model.score(x_test, y_test)
print(acc)

prediction = KNN_model.predict(x_test)

for a in range(len(prediction)):
    print("Predicted: ", prediction[a], "Data: ", x_test[a], "Actual: ", y_test[a])