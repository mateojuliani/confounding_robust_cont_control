
import argparse
import torch
import os
from fin_train_value_state_new_continuous import train_value_state_function_con

def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"Invalid truth value: {val}")


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default = "causal_max",
        help="the name of this experiment")
    parser.add_argument("--state_to_remove", type=list, default=[2],
        help="states to remove")
    parser.add_argument("--gym-id", type=str, default="Hopper-v5", 
        help="the id of the gym environment")
    parser.add_argument("--data-set", type=str, default="hopper", #halfcheetah, hopper, etc
        help="the id of the minari dataset")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=22,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=300_000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument('--log_interval', type=int, default=50_000)

    #state value specific args
    parser.add_argument('--vs_epochs', type=int, default=200) #100\
    parser.add_argument('--vs_batch_size', type=int, default=1028)
    parser.add_argument('--vs_soft_update', type=bool, default=True)
    parser.add_argument('--vs_hard_update_epochs', type=int, default=25)
    parser.add_argument('--vs_bins', type=int, default=8) 
    parser.add_argument('--vs_state_to_remove', type=int, default=5)
    parser.add_argument('--vs_max_h_val', type=int, default=998)
    parser.add_argument('--vs_policy_delay', type=int, default=3)
    parser.add_argument('--vs_gamma', type=float, default=0.99)
    parser.add_argument('--vs_tau', type=float, default=0.005)

    # options: causal, non_causal
    #Causal includes the state value function had the agent taken action not A, whereas non-causal does not include this portion.
    #paper uses causal setting
    parser.add_argument('--vs_target_function', type=str, default='causal')  
    parser.add_argument('--vs_norm_rewards', type=bool, default=True)
    parser.add_argument('--vs_mean_norm', type=bool, default=True)
    parser.add_argument('--vs_observed_policy_lr', type=float, default=1e-4)
    parser.add_argument('--vs_state_delta_lr', type=float, default=1e-5)
    parser.add_argument('--vs_q_critic_lr', type=float, default=1e-4)
    parser.add_argument('--vs_actor_lr', type=float, default=1e-4)
    parser.add_argument('--vs_optimal_state_lr', type=float, default=1e-5)

    # options: expected_reward, b_norm
    #b_norm is used in the paper and UpperBound the state value function, whereas expected reward does not and is only included for demonstrational purposes.
    parser.add_argument('--vs_best_reward_eval', type=str, default='expected_reward')  
    parser.add_argument('--vs_max_reward', type=bool, default=True)
    parser.add_argument('--vs_max_best_state', type=bool, default=True)
    parser.add_argument('--vs_reward_transition_epoch_limit', type=int, default=50)
    parser.add_argument('--vs_pretraining_epochs', type=int, default=50) #300
    parser.add_argument("--state_value_function_save_name", type=str, default = f"v_critic_mean_0_max_reward_max_state_128_layers_state_7_removed_750_epochs")
    

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99, #20
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10, #10
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    #parser.add_argument("--ema_normalize", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    #parser.add_argument("--clip_max_state", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--loss_function", type=str, default = f"mse") #l1 or mse

    #used to specify specific Minari datasets, otherwise use all available under that environment's datasets.
    parser.add_argument("--dataset_selection", type=str, default = None) 
    

    #confounding / reward shaping variables
    parser.add_argument("--add_confounding", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="adds confounding var")
    parser.add_argument("--use_reward_shaping", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="bool for using reward shaping or not")
    parser.add_argument("--reward_shaping_model", type=str, default="state_value_models/v_critics_causal_real_h_12_bins_04_28_2025.pth",
        help="model name for reward shaping")
    parser.add_argument("--q_critic", type=bool, default=False,
        help="use q critic or v critic")
    
    
    parser.add_argument("--action_bin_numbers", type=int, default=5,
        help="how many bins per action space")
    parser.add_argument("--pbrs_gamma", type=float, default=0.99,
        help="how many bins per action space")
    parser.add_argument("--descript", type=str, default = "causal 01 max reward with h gamma 1",#default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    
    #server specific things
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--tb_directory', type=str, default="explogs")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args








if __name__ == "__main__":

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_value_state_function_con(args=args, device=device, data_set = args.data_set)

