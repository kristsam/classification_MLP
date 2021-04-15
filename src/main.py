import mlp
import os
import read as r

path_to_data = os.path.dirname(os.path.realpath(__file__))+'\\..\\data\\mnist\\'
diagram_file = os.path.dirname(os.path.realpath(__file__))+'\\..\\diagrams\\'
x_train, y_train, x_test, y_test = r.read(path_to_data)

M = [100,200,300]
h_act =['h1','h2','h3']
NB = [100,200]
test_error = []

for h in h_act: 
    for m in M:
        for nb in NB:
            model = mlp.MultiLayerPerceptron()
            model.add(m,activation=h)
            model.compile([0.0001,0.00001], [0.0001])
            model.fit(x_train, y_train, batch_size=nb, epochs=50, validation_split=0.2, report=diagram_file)
            y_test_predictions = model.predict(x_test)
            test_error.append(model.score(y_test_predictions, y_test))

for i in range(0,len(test_error)):
    print('Test '+str(i)+' error='+str(test_error[i])+'.')
