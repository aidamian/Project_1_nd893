# Udacity nd893 Project #1 "Navigation" Report 


### 1. The Model
The agent model is based on a configurable Double DQN. The value/advantage streams are enabled with `dueling=True` (default). The in terms of architecture details model start with a 128-units linear layer followed by a `ReLU` activation. The dueling streams have each a 256-unit linear FC layer followed by a `ReLU` activation. For advantage offsetting we are using mean-shifting instead of max.

```
Agent model:
QNetwork(
  (relu): ReLU()
  (denses): ModuleList(
    (0): Linear(in_features=37, out_features=128, bias=True)
    (1): ReLU()
  )
  (pre_adv_layer): Linear(in_features=128, out_features=256, bias=True)
  (pre_val_layer): Linear(in_features=128, out_features=256, bias=True)
  (adv_layer): Linear(in_features=256, out_features=4, bias=True)
  (val_layer): Linear(in_features=256, out_features=1, bias=True)
)
```

### 2. The Agent and it's replay buffer

The agent can be configured to use either simple of double DQN training approach with `doble_dqn=True` (default). The models are by default created with `dueling=True` based on the agent constructor. The agent wrapper is also responsible for optimizer setup - we are using Adam with a learning rate of 1e-5. The updates are done using a discount factor `GAMMA` or 0.99.
A simple GPU capability analysis method has been added. 

### 3. The training method

The core of the training loop is the `dqn_train` function. The functions displays numerous debug information and logs all this training informatio in the `training_X.log` file where (`X` denotes the training iteration). The training loop returns the `scores` and also the `i_episode` - number of episodes that were needed to find a solution.

### 4. Training with grid-search 

In this area of the project code we explore a proposed minimal set of hyperparameters. The actual training loop execution uses the `dqn_train` functions in order to iterate through multiple dictionaries of hyperparameters within the `settings` list. The list can be either manually or grid-search generated (random or exhaustive). The first part uses the previously defined `get_device_info` to show GPU-related info if available.
Two hyperparameter sets are present in the Jupyter notebook, both with `dueling` and `double_dqn` options enables - the only difference is the starting point of the epsilon and the decay rate.
Most of the hyperparemeters are chosen at each iteration are saved in the training log.

### 5. The results

Finally we plot the results for the best grid search iteration allong with the found parameters. The best training iteration is considered the one with the smallest amount of steps until a solution is found.
Multiple results sets have been generated and the episodes-to-solutions ranges between 700 episodes up to 1300 episodes. Below is presented the partial training log for two hyperparameter settings as previously mentioned:


#### 5.1 Log #1

Below are the logs for `0.995 eps decay, 0.05 min epsilon`
```
GPU: Tesla K80, Mem:11.2GB, Procs:13
Initializing QNetwork: dueling=True
Agent model:
QNetwork(
  (relu): ReLU()
  (denses): ModuleList(
    (0): Linear(in_features=37, out_features=128, bias=True)
    (1): ReLU()
  )
  (pre_adv_layer): Linear(in_features=128, out_features=256, bias=True)
  (pre_val_layer): Linear(in_features=128, out_features=256, bias=True)
  (adv_layer): Linear(in_features=256, out_features=4, bias=True)
  (val_layer): Linear(in_features=256, out_features=1, bias=True)
)
 Running on 'cuda:0':Tesla K80
 Double DQN: 'True'
 Dueling DQN: 'True'
Initializing QNetwork: dueling=True
Training iteration 1...
  eps_start=1.0  eps_min=0.05  eps_decay=0.995
Episode  100  AvgS:  0.7  [-3.0- 4.0]  Eps: 0.6058  AvgTime: 1.5s/ep			teps 
Episode  200  AvgS:  3.7  [-1.0-12.0]  Eps: 0.3670  AvgTime: 1.6s/ep			teps 
Episode  300  AvgS:  6.0  [-1.0-17.0]  Eps: 0.2223  AvgTime: 1.6s/ep			teps 
Episode  400  AvgS:  9.9  [ 0.0-17.0]  Eps: 0.1347  AvgTime: 1.6s/ep			teps 
Episode  500  AvgS: 11.2  [ 2.0-18.0]  Eps: 0.0816  AvgTime: 1.6s/ep			teps 
Episode  600  AvgS: 12.3  [ 4.0-21.0]  Eps: 0.0500  AvgTime: 1.6s/ep			teps 
Episode  700  AvgS: 12.7  [ 5.0-23.0]  Eps: 0.0500  AvgTime: 1.5s/ep			teps 
Episode  800  AvgS: 12.6  [ 2.0-23.0]  Eps: 0.0500  AvgTime: 1.6s/ep			teps 
Episode  821  Score: 17.0  AvgS: 13.1  Eps: 0.0500  Time: 1.6s /  299 steps 
Environment (well) solved in 821 episodes!	Average Score: 13.06
Training iteration 1 done in 0.4 hrs
```