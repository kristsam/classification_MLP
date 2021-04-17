# Hiddens' layer weight cost derivative.
The computation for hiddens' layer weight cost derivative was based on back propagation and chain rule. There are some variables defined as to be more simple. The `h` function can be `softplus`, or `tanh`, or `cos`.    
![back propagation 1](back_propagation_proof_1.jpg)
![back propagation 2](back_propagation_proof_2.jpg)
## Note
__We have to remove the first column of (T-Y)*W<sup>(2)</sup> because it was added in matrix Z.__
