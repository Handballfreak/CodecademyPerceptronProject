from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


# logic gates inputs
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# labels and
labels_and = [0, 0, 0, 1]
# labels xor
labels_xor = [0, 1, 1, 0]
# labels or
labels_or = [0, 1, 1, 1]
#labels pointer
labels = labels_and

# plot points
x = [point[0] for point in data]
y = [point[1] for point in data]
plt.scatter(x, y, c = labels)
plt.show()


# Perceptron erstellen
classifier = Perceptron(max_iter = 40)
classifier.fit(data, labels)
print(classifier.score(data, labels))


# decision boundary 
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(distance) for distance in distances]


# Heatmap
distance_matrix = np.reshape(abs_distances, (100, 100))
heatmap = plt.pcolormesh(x_values, y_values, distance_matrix)
plt.colorbar(heatmap)
plt.show()
