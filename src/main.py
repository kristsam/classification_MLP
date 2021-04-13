import mlp
import os
import read as r

path_to_data = os.path.dirname(os.path.realpath(__file__))+'\\..\\data\\mnist\\'
d = 0
diagram_file = os.path.dirname(os.path.realpath(__file__))+'\\..\\diagrams\\diagram'+str(d)+'.png'
d =+ 1
x_train, y_train, x_test, y_test = r.read(path_to_data)
model = mlp.MultiLayerPerceptron()
model.add(100,activation='h3')
model.compile([0.01], [0.01])
model.fit(x_train, y_train, batch_size=100, epochs=3, validation_split=0.2, report=diagram_file)
#model.predict()