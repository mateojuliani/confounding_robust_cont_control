# PyTorch implementation of Causal Upper Bounded Reward Shaping Functions

This is a PyTorch implementation for the continuous casual upper bounded state value functions, as proposed in [Confounding Robust Continuous Control via Automatic Reward Shaping]([https://openreview.net/forum?id=ZFtjCJqEQf](https://www.arxiv.org/abs/2602.10305)).

## Usage & Implementation Details

### Getting Started
Current implementation is based on using the Minari datasets and removing dimensions to create further confounding bias as an example.

To run 
```
python main.py --data_set "hopper" --state_to_remove "[2]"
```
important args include 

* data-set: minari dataset used
  * Currently supported args: hopper, ant, halfcheetah, walker2d, pen, door, relocate. Further datasets can be added on the load_minari_data function in fin_preprocess_offline_data.py.
    
* state_to_remove: list of states to remove from observation space

To include your own dataset, change preprocessing function in fin_train_value_state_new_continuous.py in line 498 

### Files & Functions 
* main.py - main function
* fin_preprocess_offline_data.py - Used to preprocess Minari datasets
  * load_minari_data: Gets and parses Minari datasets & converts them into PyTorch tensors.
  * normalize_state_space: Normalizes state and next state to have mean zero and unit variance.
  * normalize_reward: Applies either Z-score normalization or min-max scaling to individual rewards based on the configuration flags provided in the args object.
  * normalize_rewards_and_rewards_to_go: Calculates global reward statistics (mean, std, min, max) and passes them to the reward normalization logic.
  * preprocess_data_full: Loads the raw data, applies normalization, and returns tensors on the specified device.
* fin_train_value_state_new_continuous.py - Main training file for causal upper-bounded state value functions
  * train_value_state_function_con - general function to 1) preprocess data, 2) initialize actor / critic architecture, 3) train causal upper bounded state value functions
  * train_reward_probability_state_delta - function to train reward, state transition, and behavioral models.
  * train_critic - function to train casual upper bounded state value functions.
  * sample_not_a_state_continuous - estimates value of state of action not taken by using negative sampling / clipping.
 
## Citations

If you used the continuous casual upper bounded state value functions for your experiments, consider citing the following paper:

<pre>
@inproceedings{
 juliani2026confounding,
 title={Confounding Robust Continuous Control via Automatic Reward Shaping},
 author={Mateo Juliani and Mingxuan Li and Elias Bareinboim},
 booktitle={The 25th International Conference on Autonomous Agents and Multi-Agent Systems},
 year={2026},
 url={https://openreview.net/forum?id=ZFtjCJqEQf}
}

