from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
# print(data.feature_names)  # to see the features in the data
# print(data.target_names)  # to see the targets

# split data into test and train
malignant_train, malignant_test, benign_train, benign_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)

classifier_model = KNeighborsClassifier(n_neighbors=3)  # initialize classifier
classifier_model.fit(malignant_train, benign_train)

print(classifier_model.score(malignant_test, benign_test))

# use classifier_model.predict([]) to classify new data.
