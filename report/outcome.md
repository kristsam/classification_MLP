# Outcome 
The outcome differs depending on the problem to be solved and its' difficulties.  

1. MNIST  
For _MNIST_ data, the outcome was quite positive as the models' validation predictions had in most cases less than 20% error after the 3th epoch. For the same function, mini-batch, hidden layer units and lambda, higher `learning rate` (~0.0001) seems to act better in the model because it learns faster this easy problem. For `mini-batch` higher values, the model has almost identical behaviour as for lower values, but it's being trained faster. The same stands for hiddens' layers' units amount too, it does not effect the results in a considerable way. As for the functions, `tanh`, `softplus` and `cos` seems to have the same behaviour too. `Lambda` (weight regularization) = 0.001 in general but any similar value would act almost the same and sometimes ever better. With all being said, the only parameter that changes the result is the learning rates' value for the first epochs to run but the final output for lot of epochs will be ~identical.

_First specific example_  
![diagram](../diagrams/mnist/11diagram0.png)
![diagram](../diagrams/mnist/11diagram1.png)
![diagram](../diagrams/mnist/11diagram2.png)
![diagram](../diagrams/mnist/11diagram3.png)
![diagram](../diagrams/mnist/11diagram4.png)
![diagram](../diagrams/mnist/11diagram5.png)  
_Test data output:_  
![error](../diagrams/mnist/11error.png)  

_Second specific example_  
![diagram](../diagrams/mnist/2diagram0.png)
![diagram](../diagrams/mnist/2diagram1.png)
![diagram](../diagrams/mnist/2diagram2.png)
![diagram](../diagrams/mnist/2diagram3.png)
![diagram](../diagrams/mnist/2diagram4.png)
![diagram](../diagrams/mnist/2diagram5.png)  
_Test data output:_  
![error](../diagrams/mnist/2error.png)  


_Random diagrams_  
![diagram1](../diagrams/mnist/diagram1.png)
![diagram4](../diagrams/mnist/diagram4.png)
![diagram5](../diagrams/mnist/diagram5.png)
![diagram13](../diagrams/mnist/diagram13.png)
![diagram14](../diagrams/mnist/diagram14.png)
![diagram29](../diagrams/mnist/diagram29.png)
![diagram30](../diagrams/mnist/diagram30.png)

2. CIFAR-10  
For _CIFAR-10_ data, the outcome was quite frustrating. The one hidden layer models' predictions in validation set -and even in training set- had 50% ~ 60% error. This makes sense as this classification problem is quite more difficult comparing to _MNIST_. The model is too simple and the data may be few to get a better approach in this task. As _MNIST_, the only parameter which seems to change the result is the `learning rate` value, where higher values (~0.0001) seems to act better than lower ones (~1e-05). 

_Random diagrams_  
![diagram1](../diagrams/cifar-10/diagram1.png)
![diagram3](../diagrams/cifar-10/diagram4.png)
![diagram4](../diagrams/cifar-10/diagram5.png)
![diagram4](../diagrams/cifar-10/diagram13.png)
![diagram4](../diagrams/cifar-10/diagram14.png)
![diagram4](../diagrams/cifar-10/diagram29.png)
![diagram4](../diagrams/cifar-10/diagram30.png)  

## Note
The orange model in diagrams representes the __developlment set__.