from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

'''
https://python-course.eu/machine-learning/k-nearest-neighbor-classifier-with-sklearn.php
'''

iris = datasets.load_iris()
# print(iris)
data, labels = iris.data, iris.target
# print('labels', labels)

res = train_test_split(data, labels,
                       train_size=0.8,
                       test_size=0.2,
                       random_state=12)
# train_data, test_data, train_labels, test_labels = res
x_train, x_test, y_train, y_test = res
# Create and fit a nearest-neighbor classifier (no parameters)
'''
knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)
print("Predictions from the classifier:\n")
test_data_predicted = knn.predict(test_data)
print("Predictions from the classifier:")
learn_data_predicted = knn.predict(train_data)
'''
# normalize to see quotient of correctly classified items

def knn_classification(df):
    res = train_test_split(data, labels,
                           train_size=0.8,
                           test_size=0.2,
                           random_state=12)
    x_train, x_test, y_train, y_test = res
    model = KNeighborsClassifier(algorithm='auto',
                                 leaf_size=30,
                                 metric='minkowski',
                                 p=2,  # 2 is equivalent to euclidian distance
                                 metric_params=None,
                                 n_jobs=1,
                                 n_neighbors=5,
                                 weights='uniform')
    model.fit(x_train, y_train)
    test_data_predicted = model.predict(x_test)
    ascore = accuracy_score(test_data_predicted, y_test)
    print(ascore)
