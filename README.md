# PyTorch implementation of Causal Upper Bounded Reward Shaping Functions

This is a PyTorch implementation for the paper [Confounding Robust Continuous Control via Automatic Reward Shaping](https://openreview.net/forum?id=ZFtjCJqEQf) 

## Usage & Implementation Details

Current implementation is based on using the Minari datasets and removing dimensions to create further confounding bias as an example.

To run 
```
python main.py --data_set "hopper" --state_to_remove "[2]"
```
important args include 

data-set: minari dataset used
state_to_remove: list of states to remove from observation space

To include own dataset, change preprocessing function in fin_train_value_state_new_continuous.py in line 498 
