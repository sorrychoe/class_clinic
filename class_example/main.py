from class_example.GradientDescent import GradientDescent

import numpy as np
import matplotlib.pyplot as plt
# Generate some data
X = np.random.rand(100, 2)
y = np.dot(X, np.array([2, 3])) + np.random.normal(0, 0.1, size=(100,))


# get learning rate, max_iter, threshold
learning_rate = float(input("Learning Rate를 입력해주세요(0~1)>>> "))
max_iter = int(input("반복 횟수를 입력해주세요(500~2000)>>>"))
threshold = float(input("임계 값을 입력해주세요>>>(0~1)"))

# Create a GradientDescent object and fit the data
gd = GradientDescent(learning_rate, max_iter, threshold)
gd.fit(X, y)

# Make predictions and plot cost history
y_pred = gd.predict(X)
gd.plot_cost_history()