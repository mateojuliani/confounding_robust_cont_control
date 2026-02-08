from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import math

import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchrl.objectives.common import LossModule, set_exploration_type, ExplorationType, ValueEstimators
from torchrl.modules import (
    SafeModule, 
    ProbabilisticActor, 
    TanhNormal, 
    IndependentNormal,
    NormalParamExtractor,
    ValueOperator,
)
from torchrl.data.tensor_specs import BoundedContinuous, TensorSpec, UnboundedContinuous
from fin_preprocess_offline_data import preprocess_data_full


class GaussianNN(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim=256):
        """
        Given an input, outputs a tensor 2x the size of the output dim to get mean action and expected std of that output
        Technically does not have to be a normal distribution
        """

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = target_dim * 2

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
        
        self.network.apply(self.init_weights)

    def forward(self, x):
        output = self.network(x)
        mean, log_std = torch.chunk(output, 2, dim=-1)

        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        return TanhNormal(loc=mean, scale=std)
    
    @staticmethod
    def init_weights(m):
      if isinstance(m, nn.Linear):
          nn.init.uniform_(m.weight, -0.01, 0.01)  
          nn.init.zeros_(m.bias)  
    
class RegressionNN(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim=256):
        """
        Similar to GuassianNN, but just outputs mean / 1 dimension 
        """

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = target_dim * 2

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):

        output = self.network(x)
        return output

class Critic(nn.Module): #Estimate our Q bar value

    #input: state, time step
    #output: value of state, action, time step pair
    def __init__(self, state_dim, action_dim, max_h, max_r, min_r, gamma):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.network.apply(self.init_weights)
        self.max_h = max_h
        self.gamma = gamma 
        self.max_r = max_r 
        self.min_r = min_r

    def forward(self, state):
        min_vs = self.min_r / (1- self.gamma)
        max_vs = self.max_r / (1- self.gamma)
        return torch.clamp(self.network(state), min = min_vs, max = max_vs)


    @staticmethod
    def init_weights(m):
      if isinstance(m, nn.Linear):
          nn.init.uniform_(m.weight, -0.01, 0.01)  #gain=nn.init.calculate_gain('tanh'))
          nn.init.zeros_(m.bias) 



class CausalUpperBoundEstimator:
    def __init__(self,
                 state_dim,
                 action_dim,
                 state_mean,
                 state_std,
                 max_action,
                 max_location,
                 max_h,
                 max_reward,
                 min_reward,
                 mean_reward,
                 std_reward,
                 device,
                 args,
                 gamma = 0.99,
                 tau = 0.005,
                 df = None,
                 observed_policy_lr = 1e-3,
                  state_delta_lr = 1e-3,
                  q_critic_lr = 1e-3,
                  actor_lr = 1e-3,
                  optimal_state_lr = 1e-5,
                 reward_lr = 1e-4,
                 num_sample_neg_a: int = 30,
                 neg_action_thres: float = .1,
                 ):

        self.critic = Critic(state_dim, action_dim, max_h, max_reward, min_reward, gamma).to(device)
        self.critic_target = Critic(state_dim, action_dim, max_h, max_reward, min_reward, gamma).to(device)

        self.critic_twin = Critic(state_dim, action_dim, max_h, max_reward, min_reward, gamma).to(device)
        self.critic_target_twin = Critic(state_dim, action_dim, max_h, max_reward, min_reward, gamma).to(device)

        self.policy = GaussianNN(state_dim, action_dim, max_action).to(device)
        self.policy_fin = GaussianNN(state_dim, action_dim, max_action).to(device)

        self.state_transition_model = RegressionNN(input_dim = state_dim+action_dim, target_dim = state_dim).to(device)
        self.state_transition_model_fin = RegressionNN(input_dim = state_dim+action_dim, target_dim = state_dim).to(device)

        self.reward_model = RegressionNN(input_dim = state_dim+action_dim, target_dim = 1).to(device)
        self.reward_model_fin = RegressionNN(input_dim = state_dim+action_dim, target_dim = 1).to(device)


        #Copy weights in from actor & critic
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target_twin.load_state_dict(self.critic_twin.state_dict())

        #get optimizer
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=q_critic_lr, weight_decay=1e-5)
        self.critic_twin_optimizer = optim.Adam(self.critic_twin.parameters(), lr=q_critic_lr, weight_decay=1e-5)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr, weight_decay=1e-5)
        self.state_transition_model_optimizer = optim.Adam(self.state_transition_model.parameters(), lr=state_delta_lr, weight_decay=1e-5)
        self.reward_model_optimizer = optim.Adam(self.reward_model.parameters(), lr=reward_lr, weight_decay=1e-5)

        self.reward_loss_min = 1000000
        self.state_loss_min = 1000000
        self.prob_loss_min = 1000000




        self.args = args
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau  # Soft update rate
        self.b = max_reward
        self.min_reward = min_reward
        self.mean_reward = mean_reward
        self.std_reward = std_reward

        self.state_mean = state_mean
        self.state_std = state_std
        self.max_h = max_h
        self.device = device

        self.num_sample_neg_a = num_sample_neg_a
        self.neg_action_thres = neg_action_thres

        print("state mean")
        print(self.state_mean)

        print("state std")
        print(self.state_std)


    def update_target_networks(self, tau = -1, critic = True, actor = True, optimal_state = True, state_delta = True):

      if tau == -1:
        tau = self.tau

      for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

      for param, target_param in zip(self.critic_twin.parameters(), self.critic_target_twin.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def copy_model(self, model1, model2):

        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            param2.data.copy_(param1.data)


    def select_action(self, state):
      action_probs = self.actor(state)
      dist = torch.distributions.Categorical(action_probs)
      action = dist.sample()  
      return action


    def sample_not_a_state_continuous(self, s, a, next_step, num_samples, twin = False, max = False, max_reward = False):

        with torch.no_grad():
          org_batch_size, num_dims = a.shape
          dist = self.policy_fin(s)
          new_actions = dist.rsample((num_samples,))
          original_actions = a.unsqueeze(-1).expand(-1, -1, num_samples).permute(0, 2, 1).reshape(-1, num_dims)
          sampled_actions = new_actions.transpose(0,1).reshape(org_batch_size, num_dims, num_samples).permute(0, 2, 1).reshape(-1, num_dims)
          

          pos_thresh = torch.full(
              (1, num_dims),  
              fill_value=self.neg_action_thres,
              dtype=torch.float32,
              device=self.device
          )

          neg_thresh = torch.full(
              (1, num_dims),  
              fill_value=-1*self.neg_action_thres,
              dtype=torch.float32,
              device=self.device
          )


          max_action = original_actions + pos_thresh
          min_action = original_actions + neg_thresh
                    
          
          max_delta = (original_actions - torch.clamp(max_action, min = -1, max=1))
          min_delta = (original_actions - torch.clamp(min_action, min = -1, max=1))
          actual_delta = (original_actions - sampled_actions)


          stacked_actions = torch.stack([max_delta, min_delta, actual_delta], dim=1)
          _, max_index = torch.abs(stacked_actions).max(dim=1, keepdim = True)

          batch_size, _, dim = stacked_actions.shape

          batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, 1, dim) 
          dim_indices = torch.arange(dim).view(1, 1, -1).expand(batch_size, 1, -1)   

          fin_action_delta = stacked_actions[batch_indices, max_index, dim_indices]  

          fin_action_delta = fin_action_delta.permute(0, 2, 1).squeeze(-1)

          clean_not_a_actions = original_actions - fin_action_delta
          expanded_states = s.unsqueeze(-1).expand(-1, -1, num_samples).permute(0, 2, 1).reshape(-1, self.state_dim)


          h_reshaped = next_step.expand(-1, num_samples).reshape(-1, 1) #not used, but kept in for legacy purposes 

          state_action_pair = torch.cat([expanded_states, clean_not_a_actions], 1)

          mean = self.state_transition_model_fin(state_action_pair)
          not_a_states = expanded_states + mean

          mean_reward = self.reward_model_fin(state_action_pair)
          
          not_a_states = (not_a_states - self.state_mean) / (self.state_std + 1e-7) #normalize states
      

          values_twin = self.critic_target_twin(not_a_states)
          values = self.critic_target(not_a_states)

          
          values = torch.min(values_twin, values).reshape(org_batch_size, num_samples, 1)


          if max:
            fin = values.max(dim=2).values.max(dim=1).values.unsqueeze(1)
          else:
            fin = values.max(dim=2).values.mean(dim=1).unsqueeze(1)

          if max_reward:
            fin_reward_reshaped = mean_reward.reshape(org_batch_size, num_samples).max(dim=1).values.unsqueeze(1)
          else:
            fin_reward_reshaped = mean_reward.reshape(org_batch_size, num_samples).mean(dim=1).unsqueeze(1)


          dist_new = self.policy_fin(expanded_states)
          not_a_log_prob = dist_new.log_prob(clean_not_a_actions).reshape(org_batch_size, num_samples).mean(dim=1).unsqueeze(1)

        return fin, fin_reward_reshaped, not_a_log_prob



    def train_critic(self, s, a, sp, r, d, h, trunc, soft_update, overall_step_count, target_function="all", norm_rewards=True, mean_norm = False, reward_type = "b_norm", 
    max_best_state = False, max_reward = False, policy_delay=3, debug=False, clipped=False, switch = 0):

        r_norm = r
        b_norm = self.b
        next_step = h+1

        sp_norm = (sp - self.state_mean) / (self.state_std + 1e-7)
        s_norm = (s - self.state_mean) / (self.state_std + 1e-7)



        with torch.no_grad():


            dist = self.policy_fin(s)
            a_given_s = dist.log_prob(a).unsqueeze(1)
            

            # Get Q-values for all actions in the next state
            state_action_pair = torch.cat([s, a], 1)
            pred_next_step_delta = self.state_transition_model_fin(state_action_pair)
            pred_next_step = s + pred_next_step_delta

            pred_next_step = (pred_next_step - self.state_mean) / (self.state_std + 1e-7)

            #if the step is truncated, then the next obs is meaningless, so we will forcast it ourselves
            sp_norm_updated = torch.where(trunc == 1, pred_next_step, sp_norm)
            v_values_next = self.critic_target(sp_norm_updated)
            v_values_twin_next = self.critic_target_twin(sp_norm_updated)

            v_given_current_action = torch.min(v_values_next, v_values_twin_next)

            if target_function == "non_causal":
              #value_of_action_not_taken = v_given_current_action * 0
              value_of_action_taken = (r_norm + (1 - d) * self.gamma * v_given_current_action)
              value_of_action_not_taken = value_of_action_taken
            
            elif target_function == "causal":

                #value_of_action_taken = a_given_s * (r_norm + (1 - d) * self.gamma * v_given_current_action)
                value_of_action_taken = (r_norm + (1 - d) * self.gamma * v_given_current_action)

                #Sample States had action not A been taken
                not_a_state_v, not_a_reward, not_a_log_prob  = self.sample_not_a_state_continuous(s, a, next_step, 25, twin = False, max = max_best_state, max_reward = max_reward)

                #ensures we are actually upper bounding the states 
                not_a_state_v = torch.max(not_a_state_v, v_given_current_action)
                  




                #NOTE: d here is only if its a terminal 
                #included reward type expected reward for demonstrated purposes, 
                #however expected_reward will not upper bound the state value function
                #use b_norm instead
                if reward_type == "expected_reward":
                    #value_of_action_not_taken = not_a_log_prob * (not_a_reward + (1 - d) * self.gamma * not_a_state_v)
                    value_of_action_not_taken = (not_a_reward + (1 - d) * self.gamma * not_a_state_v)
                elif reward_type == "b_norm":
                    #value_of_action_not_taken = not_a_log_prob * (b_norm + (1 - d) * self.gamma * not_a_state_v)
                    value_of_action_not_taken = (b_norm + (1 - d) * self.gamma * not_a_state_v)
                else:
                    raise Exception("Invalid Reward_type")

            else:
                raise Exception("Invalid Target Function")


            if target_function == "non_causal":
              a_prob_ratio = 1
              not_a_prob_ratio = 0
            else:
               
              a_given_s = torch.clamp(a_given_s, min = -50, max = -0.01)
              not_a_log_prob = torch.clamp(not_a_log_prob, min = -50, max = -0.01)
              a_prob_ratio = torch.exp(a_given_s) / (torch.exp(not_a_log_prob) + torch.exp(a_given_s))
              not_a_prob_ratio = 1 - a_prob_ratio


            target_V = (a_prob_ratio * value_of_action_taken + not_a_prob_ratio * value_of_action_not_taken).detach()
          

        if self.args.loss_function == "mse":
          v_criterion = nn.MSELoss()
        elif self.args.loss_function == "l1":
          v_criterion = nn.SmoothL1Loss()
        else:
          raise Exception("Invalid Loss Function")



        # Update critic loss
        current_V = self.critic(s_norm)
        critic_loss = v_criterion(current_V, target_V)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
        self.critic_optimizer.step()

        # Optimize Q_twin
        current_V_twin = self.critic_twin(s_norm)
        critic_twin_loss = v_criterion(current_V_twin, target_V)


        self.critic_twin_optimizer.zero_grad()
        critic_twin_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_twin.parameters(), max_norm=1)
        self.critic_twin_optimizer.step()


        if overall_step_count % policy_delay == 0 and overall_step_count > 0 and soft_update:
          self.update_target_networks()

        if target_function == "non_causal":
            return critic_loss, current_V.detach().mean().item(), target_V.detach().mean().item(), 0, current_V, target_V

        return critic_loss, current_V.detach().mean().item(), target_V.detach().mean().item(), not_a_state_v.mean(), current_V, target_V


    def train_reward_probability_state_delta(self, s, a, sp, r, d, h, epoch, reward_transition_epoch_limit = 50, norm_rewards=True, mean_norm = False):

        r_norm = r
        b_norm = self.b

        state_action_pair = torch.cat([s, a], 1)


        probs_actor_loss = 0
        state_delta_loss = 0

        #train probability actor
        dist = self.policy(s)  
        log_prob = dist.log_prob(a)
        probs_actor_loss = -log_prob.mean()

        self.policy_optimizer.zero_grad()
        probs_actor_loss.backward()
        self.policy_optimizer.step()


        #Train State transition model
        mse = nn.MSELoss()
        state_delta_val = sp - s
        state_transition_mean = self.state_transition_model(state_action_pair)
        state_transition_loss = mse(state_transition_mean, state_delta_val)

        self.state_transition_model_optimizer.zero_grad()
        state_transition_loss.backward()
        self.state_transition_model_optimizer.step()

        mean_reward = self.reward_model(state_action_pair)
        reward_loss = mse(mean_reward, r_norm)

        self.reward_model_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_model_optimizer.step()

        probs_return_loss = probs_actor_loss.item()

        return probs_return_loss, state_transition_loss.item(), reward_loss.item()


def train_value_state_function_con(
                            args, 
                            device,
                            data_set,
                            debug = False
                            ):
    
    #load data 

    steps, states, d_actions, rewards, next_states, dones, truncateds, terminals = preprocess_data_full(args = args, 
                                                                                                        device=device, 
                                                                                                        normalize_states = False,  
                                                                                                        data_set=data_set, 
                                                                                                        action_num_bins = args.vs_bins,
                                                                                                        dataset_selection = args.dataset_selection
                                                                                                        )


    print(steps.shape)
    print(states.shape)
    print(d_actions.shape)
    print(rewards.shape)
    print(next_states.shape)
    print(terminals.shape)

    print(torch.max(rewards))
    print(torch.min(rewards))
    print(torch.std(rewards))
    print(torch.mean(rewards))
    print(torch.median(rewards))

    if len(args.state_to_remove) > 5:
      states_to_remove_print = f"{args.state_to_remove[0]}_to_{args.state_to_remove[-1]}"
    else: 
      states_to_remove_print = '_'.join(str(x) for x in args.state_to_remove)


    init_time_pre = int(time.time())
    init_time = datetime.fromtimestamp(init_time_pre).strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"state_value_training__continuous_{args.exp_name}__{states_to_remove_print}__{init_time}"
    writer = SummaryWriter(log_dir=f"{args.tb_directory}/{run_name}")
    #log the config vals
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    #writer.add_text("Config Parameters", config_text)

    description = """
    ## Experiment Notes:
    - 0 1 norm, max reward
    """
    writer.add_text("Run Description", description)





    state_indices = torch.tensor([i for i in range(states.shape[1]) if i not in args.state_to_remove]).to(device)  #columns to remove


    epochs = args.vs_epochs
    batch_size = args.vs_batch_size
    soft_update = args.vs_soft_update

    s = torch.index_select(states, 1, state_indices).to(device)
    sp = torch.index_select(next_states, 1, state_indices).to(device)
    a = d_actions.to(device)
    r = rewards.to(device)
    d = terminals.to(device)
    trunc = truncateds.to(device)
    h = steps.to(device)


    agent = CausalUpperBoundEstimator(
        state_dim = s.shape[1],
        action_dim = a.shape[1],
        state_mean = s.mean(dim=0, keepdim=True),
        state_std =  s.std(dim=0, keepdim=True),
        max_action = 1,
        max_h = args.vs_max_h_val,
        max_reward = torch.max(r).item(),
        min_reward = torch.min(r).item(),
        mean_reward = torch.mean(r).item(),
        std_reward = torch.std(r).item(),
        device = device,
        max_location = 1,
        gamma = args.vs_gamma,
        tau = args.vs_tau,
        observed_policy_lr = args.vs_observed_policy_lr,
        state_delta_lr = args.vs_state_delta_lr,
        q_critic_lr = args.vs_q_critic_lr,
        actor_lr = args.vs_actor_lr,
        optimal_state_lr = args.vs_optimal_state_lr,
        args = args

    )



    

    random_indices = torch.randperm(steps.shape[0])
    s = s[random_indices]
    a = a[random_indices]
    sp = sp[random_indices]
    r = r[random_indices]
    d = d[random_indices]
    h = h[random_indices]
    trunc = trunc[random_indices]

    
    for epoch in range(0, args.vs_pretraining_epochs):
      sum_reward_loss = 0
      sum_prob_loss = 0
      sum_state_trans_loss = 0
      batches = 0
      for i in range(0, len(s), batch_size):
        batch_trunc = trunc[i:i+batch_size]
        mask = (batch_trunc == 0).squeeze(-1)

        #ignore truncated states bc state diff forecast will be messed up. 
        batch_s = s[i:i+batch_size][mask]
        batch_a = a[i:i+batch_size][mask]
        batch_sp = sp[i:i+batch_size][mask]
        batch_r = r[i:i+batch_size][mask]
        batch_d = d[i:i+batch_size][mask]
        batch_h = h[i:i+batch_size][mask]

        probs_actor_loss, state_transition_loss, reward_loss = agent.train_reward_probability_state_delta(
                    batch_s,
                    batch_a,
                    batch_sp,
                    batch_r,
                    batch_d,
                    batch_h,
                    epoch = epoch,
                    norm_rewards = args.vs_norm_rewards,
                    mean_norm = args.vs_mean_norm,
                    reward_transition_epoch_limit = args.vs_reward_transition_epoch_limit
                    )
        
        sum_prob_loss += probs_actor_loss
        sum_state_trans_loss += state_transition_loss
        sum_reward_loss += reward_loss
        batches += 1

      observed_policy_avg_loss = sum_prob_loss / batches
      avg_reward_loss = sum_reward_loss / batches
      avg_state_trans_delta = sum_state_trans_loss / batches
      
      writer.add_scalar("probability_loss", observed_policy_avg_loss, epoch)
      writer.add_scalar("reward_loss", avg_reward_loss, epoch)
      #writer.add_scalar("state_delta_loss", avg_state_delta, epoch)
      writer.add_scalar("state_trans_loss", avg_state_trans_delta, epoch)
      
      print(f"Pretraining Epoch: {epoch} Probability: {round(observed_policy_avg_loss, 3)} Reward: {round(avg_reward_loss, 3)} State Delta: {round(avg_state_trans_delta, 3)} ") 
      
      #save down env models
      if (observed_policy_avg_loss < agent.prob_loss_min and observed_policy_avg_loss > 0) or epoch == 0:
        print("saved policy")
        agent.prob_loss_min = observed_policy_avg_loss
        agent.copy_model(agent.policy, agent.policy_fin) 

      if (avg_reward_loss < agent.reward_loss_min and avg_reward_loss > 0.001) or epoch == 0:
        print("saved reward")
        agent.reward_loss_min = avg_reward_loss
        agent.copy_model(agent.reward_model, agent.reward_model_fin) 

      if (avg_state_trans_delta < agent.state_loss_min and avg_state_trans_delta > 0.005) or epoch == 0:
        print("saved state")
        agent.state_loss_min = avg_state_trans_delta
        agent.copy_model(agent.state_transition_model, agent.state_transition_model_fin)

    min_loss = 10000
    prior_loss = 0
    best_epoch = 0 
    switch = 1
    for epoch in range(0, epochs):
        
        overall_step_count = 0
        sum_loss = 0
        sum_target_q = 0
        batches = 0
        sum_q = 0
        actor_loss_val = 0
        optimal_state_loss_val = 0
        q_current_action_val = 0
        q_max_action_val = 0
        prob_loss_sum = 0
        sum_reward_loss = 0
        sum_state_delta_loss = 0
        sum_state_trans_loss = 0

        for i in range(0, len(s), batch_size):

            #remove truncated states to avoid weird calcs
            batch_trunc = trunc[i:i+batch_size]
            mask = (batch_trunc == 0).squeeze(-1)


            batch_s = s[i:i+batch_size]
            batch_a = a[i:i+batch_size]
            batch_sp = sp[i:i+batch_size]
            batch_r = r[i:i+batch_size]
            batch_d = d[i:i+batch_size]
            batch_h = h[i:i+batch_size]
            batch_trunc = trunc[i:i+batch_size]




            critic_loss, current_q, target_q, q_max_action, V, V_target = agent.train_critic(
                        batch_s,
                        batch_a,
                        batch_sp,
                        batch_r,
                        batch_d,
                        batch_h,
                        batch_trunc,
                        soft_update = soft_update,
                        overall_step_count = overall_step_count,
                        policy_delay = args.vs_policy_delay,
                        target_function = args.vs_target_function,
                        norm_rewards = args.vs_norm_rewards,
                        mean_norm = args.vs_mean_norm,
                        reward_type = args.vs_best_reward_eval,
                        max_best_state = args.vs_max_best_state,
                        max_reward = args.vs_max_reward, 
                        switch = switch
                        )
      



            #reward_loss = 0
            sum_loss += critic_loss.item()
            sum_reward_loss += reward_loss
            sum_q += current_q
            sum_target_q += target_q
            q_max_action_val += q_max_action
            prob_loss_sum += probs_actor_loss
            batches += 1
            overall_step_count += 1


        avg_loss = sum_loss / batches
        avg_q = sum_q / batches
        avg_t_q = sum_target_q / batches
        avg_a_l = actor_loss_val / batches
        optimal_state_loss_val_avg = optimal_state_loss_val / batches
        avg_q_current_action_val = q_current_action_val / batches
        avg_q_max_action_val = q_max_action_val / batches
        observed_policy_avg_loss = prob_loss_sum/ batches
        avg_reward_loss = sum_reward_loss / batches
        avg_state_delta = sum_state_delta_loss / batches
        avg_state_trans_delta = sum_state_trans_loss / batches

        print(f"Epoch: {epoch} Mean Current_Q: {round(avg_q, 3)} Target Q: {round(avg_t_q, 3)} Q Loss: {round(avg_loss, 3)} Timestamp: {datetime.fromtimestamp(datetime.now().timestamp()).strftime("%Y-%m-%d %H:%M:%S")} ")

        writer.add_scalar("Mean_Current_Q", avg_q, epoch)
        writer.add_scalar("Mean_Target_Q", avg_t_q, epoch)
        writer.add_scalar("Critic_loss", avg_loss, epoch)
        writer.add_scalar("actor_loss", avg_a_l, epoch)
        writer.add_scalar("optimal_state_loss", optimal_state_loss_val_avg, epoch)
        writer.add_scalar("q_current_action", avg_q_current_action_val, epoch)
        writer.add_scalar("q_max_action", avg_q_max_action_val, epoch)




        if avg_loss < min_loss and avg_loss < prior_loss and epoch > 60: #
            min_loss = avg_loss

            #if current loss is less than prior epochs and the current loss is lower than the previous min_loss, save down the model as the "best model"
            #sets code to 300 
            torch.save(agent.critic.state_dict(), f"state_value_models/continuous_models/{args.data_set}/best_epoch_{args.vs_best_reward_eval}__{states_to_remove_print}__{args.state_value_function_save_name}__{init_time}.pth")
            best_epoch = epoch



        prior_loss = avg_loss
    writer.close()


    print(f"best epoch {best_epoch} best loss: {min_loss}")
    print(f"Saving Model: state_value_models/continuous_models/{args.data_set}/{args.vs_best_reward_eval}__{states_to_remove_print}__{args.state_value_function_save_name}__{init_time}_final.pth")
    torch.save(agent.critic.state_dict(), f"state_value_models/continuous_models/{args.data_set}/{args.vs_best_reward_eval}__{states_to_remove_print}__{args.state_value_function_save_name}__{init_time}_final.pth")








