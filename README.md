# risk-aware-federated-learning

This repository is dedicated to showcasing three essential code scripts that demonstrate the power and potential of risk-aware federated learning.

## Motivating Example
In the "Motivating Example" script, we provide a the scenario where 3 classes have been distributed with heterogeneous way to 3 users (each user has 1 pattern). The users are selected by the RAM with probabilities p_1 = 0.5, p_2 = 0.4, and p_3 = 0.1. 
The script highlights at that the risk-aware federated learning (when the parameter a = 0.1) the decision boundary manages to classify all the data points, compared to the risk-neutral case (when the parameter a = 1. ).
The hyperparameters are tuned like:
* cvar_alpha_list = [1.,0.1]
* parameeter gamma = 0.1
* number of users K = 3
* number of global rounds T = 4000
* number of local epochs H = 10
* users probabilities are [  0.5 , 0.4, 0.1 ]
* The RAM allows only users_per_round = 1 users to send to the server
* learning rates for theta:  learning_rate = 0.001
* learning rates for t: learning_rate_t =  0.001

## MNIST Dataset
In the MNIST Dataset script, has 30 users. The 3 of the less often users hold 1(and/or 2) patterns exclusively. The RAM has imposed a non-uniform availability for the users. The script highlights the that the risk-aware federated learning (when the parameter a =0.3) the algorithm achieves better overall testing accuracy and better testing accuracy at the patterns that belong exclusively on the less often users compared to the risk-neutral case (when the parameter a = 1. ).
The hyperparameters are tuned like:
* cvar_alpha_list = [1.,0.3]
* parameeter gamma = 0.3
* number of users K = 30
* number of global rounds T = 4000
* number of local epochs H = 10
* The RAM allows only users_per_round = 1 users to send to the server
* learning rates for theta:  learning_rate = 0.001
* learning rates for t: learning_rate_t =  0.0001
* the batch size is batch_size = 128
* the model has layer with hidden_size=(128,128)


## FashionMNIST Dataset
In the FashionMNIST Dataset script, has 30 users. The 3 of the less often users hold 2 patterns exclusively. The RAM has imposed a non-uniform availability for the users. The script highlights the that the risk-aware federated learning (when the parameter a -> 0., better performance for a=0.1) the algorithm achieves better overall testing accuracy and better testing accuracy at the patterns that belong exclusively on the less often users compared to the risk-neutral case (when the parameter a = 1. ).
The hyperparameters are tuned like:
* cvar_alpha_list = [1.,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3 , 0.2, 0.1]
* parameeter gamma = [1.,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3 , 0.2, 0.1]
* number of users K = 30
* number of global rounds T = 6000
* number of local epochs H = 20
* The RAM allows only users_per_round = 1 users to send to the server
* learning rates for theta:  learning_rate = 0.01
* learning rates for t: learning_rate_t =  0.0005
* the batch size is batch_size = 128
* the model has layer with hidden_size=(120, 84)
* the model has layer with conv_size =(6,16)
* the model has layer with kernel_size = 5
* the model has layer with padding=  2 



Feel free to explore, experiment with this project. 

