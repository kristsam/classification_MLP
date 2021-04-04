import mlp
import os
import read as r

h = mlp.choose_func()
path_to_data = os.path.dirname(os.path.realpath(__file__))+'\\..\\data\\mnist\\'
x_train, y_train, x_test, y_test = r.read(path_to_data)
mlp = mlp.MultiLayerPerceptron()
mlp.add(100,h)
mlp.compile(0.01, 0.01)
mlp.fit(x_train, y_train, 100, 0.2)
#mlp.predict()