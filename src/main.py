import mlp
import os, time
import read as r
import layers

start = time.time()

CIFAR_10 = 'cifar-10'
MNIST = 'mnist'

path_to_data = os.path.dirname(os.path.realpath(__file__))+'\\..\\data\\'
diagram_file = os.path.dirname(os.path.realpath(__file__))+'\\..\\diagrams\\'
diagram_files = [diagram_file+CIFAR_10+'\\', diagram_file+MNIST+'\\']

data_list = []
data_list.append(r.read_cifar_10(path_to_data+CIFAR_10+'\\'))
data_list.append(r.read(path_to_data+MNIST+'\\'))


M = [100,200,300]
h_act =['softplus','tanh','cos']
NB = [100,200]
test_error = []

# for each dataset
for i in range(0, len(data_list)):
    for h in h_act: 
        for m in M:
            for nb in NB:
                model = mlp.MultiLayerPerceptron()
                model.add(layers.HiddenLayer(m, activation=h, add_column=True))
                model.compile([0.0001,0.00001], [0.00001])
                model.fit(data_list[i][0], data_list[i][1], batch_size=nb, epochs=50, validation_split=0.2, report=diagram_files[i])
                y_test_predictions = model.predict(data_list[i][2])
                y_test_score = model.score(y_test_predictions, data_list[i][3])
                print("Error = "+str(y_test_score)+" for test set. Model chose learning rate = "+str(model.learning_rate_best)+" , lambda = "+str(model.lam_best))


print('Time for the whole process was '+str(int((time.time()-start)/60))+' mins.')
