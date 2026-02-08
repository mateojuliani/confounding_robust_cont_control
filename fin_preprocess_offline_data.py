import torch
import minari
import numpy as np

def load_minari_data(env_name = "hopper", dataset_selection = None):

    if env_name in ["hopper", "ant", "halfcheetah", "walker2d"]:
        data_set_list = ["simple-v0", "medium-v0", "expert-v0"] 
        dataset_name = "mujoco"
    elif env_name in ["pen", "door", "relocate"]:
        data_set_list = ["human-v2", "cloned-v2", "expert-v2"]
        dataset_name = "D4RL"
    else:
        return Exception("invalid env name")

    if dataset_selection is not None:
        data_set_list = [dataset_selection]
        print(data_set_list)

    
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    steps = []
    truncated_l = []
    terminated_l = []
    c = 0
    t = 0
    for data_set in data_set_list:
        

        try: 
            dataset = minari.load_dataset(f"{dataset_name}/{env_name}/{data_set}",download=True)
        except:
            continue 

        for episode in dataset.iterate_episodes():
            step_counter = 0
            obs = episode.observations
            acts = episode.actions
            rews = episode.rewards
            truncated = episode.truncations 
            terminated = episode.terminations
            infos = episode.infos
            
            for t in range(len(obs) - 1):
                states.append(torch.tensor(obs[t], dtype=torch.float32))
                actions.append(torch.tensor(acts[t], dtype=torch.float32))
                rewards.append(torch.tensor(rews[t], dtype=torch.float32))
                next_states.append(torch.tensor(obs[t+1], dtype=torch.float32))
                done = np.logical_or(truncated[t], terminated[t])
                truncated_l.append(torch.tensor(truncated[t], dtype = torch.bool))
                terminated_l.append(torch.tensor(terminated[t], dtype = torch.bool))
                dones.append(torch.tensor(done, dtype=torch.bool))
                steps.append(torch.tensor(step_counter, dtype=torch.int32))
                step_counter += 1
                
            c+=1
        


    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards).unsqueeze(1)
    next_states = torch.stack(next_states)
    truncateds = torch.stack(truncated_l).unsqueeze(1)
    terminals = torch.stack(terminated_l).unsqueeze(1)
    dones = torch.stack(dones).unsqueeze(1)
    steps = torch.stack(steps).unsqueeze(1)

    return steps, states, actions, rewards, next_states, dones.int(), truncateds.int(), terminals.int()


def normalize_state_space(states, next_states):

    states = (states - states.mean(dim=0, keepdim=True)) / states.std(dim=0, keepdim=True)
    next_states = (next_states - next_states.mean(dim=0, keepdim=True)) / next_states.std(dim=0, keepdim=True)

    return states, next_states

def normalize_reward(args, r, mean_reward, std_reward, min_reward, max_reward):

        if args.vs_mean_norm and args.vs_norm_rewards:

            r_norm = (r - mean_reward) / (std_reward + 1e-7)

        elif args.vs_norm_rewards:

            num = max_reward + min_reward
            dom = max_reward - min_reward

            r_norm = (r - min_reward) / dom
        else:
            r_norm = r

        return r_norm

def normalize_rewards_and_rewards_to_go(args, r, get_rewards_to_go = True, env_name = "hopper"):

        mean_reward = torch.mean(r)
        std_reward = torch.std(r)
        min_reward = torch.min(r)
        max_reward = torch.max(r)

        print(mean_reward)
        print(std_reward)
        print(min_reward)
        print(max_reward)

        r_norm = normalize_reward(args, r, mean_reward, std_reward, min_reward, max_reward)
        return r_norm



def preprocess_data_full(args, device, data_set = "halfcheetah", action_num_bins = 8, normalize_states = False, dataset_selection = None): #hopper

    #load data 
    steps_all, states_all, actions_all, rewards_all, next_states_all, dones, truncateds, terminals = load_minari_data(env_name = data_set, dataset_selection = dataset_selection)

    if normalize_states:
        states_all, next_states_all = normalize_state_space(states_all, next_states_all)

    rewards_all = normalize_rewards_and_rewards_to_go(args, rewards_all, get_rewards_to_go = False)

    return steps_all.to(device), states_all.to(device), actions_all.to(device), rewards_all.to(device), next_states_all.to(device), dones.to(device), truncateds.to(device), terminals.to(device)


