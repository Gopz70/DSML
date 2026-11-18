import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-5, 5, 100)
functions = {
 "Step": np.where(x >= 0, 1, 0),
 "Sigmoid": 1 / (1 + np.exp(-x)),
 "Tanh": np.tanh(x),
 "ReLU": np.maximum(0, x),
 "Softmax": np.exp(x) / np.sum(np.exp(x))
}
for name, y in functions.items():
 plt.figure()
 plt.plot(x, y)
 plt.title(name + " Function")
 plt.xlabel("Input")
 plt.ylabel("Output")
 plt.grid()
 plt.show()
