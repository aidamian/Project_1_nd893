# Udacity nd893 Project #1 "Navigation" Report 


### The Model
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

### The Agent and it's replay buffer

The agent can be configured to use either simple of double DQN training approach with `doble_dqn=True` (default). The models are by default created with `dueling=True` based on the agent constructor. A simple GPU capability analysis method has been added.

### The training method

The core of the training loop is the `dqn_train` function. The functions displays numerous debug information and logs all this training informatio in the `training_X.log` file where (`X` denotes the training iteration). The training loop returns the `scores` and also the `i_episode` - number of episodes that were needed to find a solution.

### Training with grid-search 

In this area of the project code we explore a proposed minimal set of hyperparameters. The actual training loop execution uses the `dqn_train` functions in order to iterate through multiple dictionaries of hyperparameters within the `settings` list. The list can be either manually or grid-search generated (random or exhaustive). The first part uses the previously defined `get_device_info` to show GPU-related info if available.

### The results

Finally we plot the results for the best grid search iteration allong with the found parameters. The best training iteration is considered the one with the smallest amount of steps until a solution is found.
Multiple results sets have been generated and the episodes-to-solutions ranges between 700 and 1300
