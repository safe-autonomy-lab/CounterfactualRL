#!/usr/bin/env python
# coding: utf-8
# import package
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gym
import numpy as np
import pandas as pd
import math
import torch

from stable_baselines3 import DQN, PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

from typing import Optional
import gym
from gym.envs.registration import register
pd.set_option('display.max_columns', None)
import sys
sys.path.append("/anaconda3/envs/CF_diabetic_train_ppo_generalize/lib/python3.8/site-packages/gym")
sys.path.append("/anaconda3/envs/CF_diabetic_train_ppo_generalize/lib/python3.8/site-packages/simglucose")
import simglucose
#import cf_generator
import matplotlib.pyplot as plt
from random import sample
import warnings
warnings.filterwarnings('ignore')
import argparse
# modified the core.py in gym==0.21.0 to make it work

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('New folder ok.')
    else:
        print('There is this folder')

    return

# wrapper used
class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=3600):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode over?, additional informations
        """
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Overwrite the truncation signal when when the number of steps reaches the maximum
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

class NormalizeActionWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, env):
    # Retrieve the action space
    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
    # Retrieve the max/min values
    self.low, self.high = action_space.low, action_space.high

    # We modify the action space, so all actions will lie in [-1, 1]
    env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)

    # Call the parent constructor, so we can access self.env later
    super(NormalizeActionWrapper, self).__init__(env)
  
  def rescale_action(self, scaled_action):
      """
      Rescale the action from [-1, 1] to [low, high]
      (no need for symmetric action space)
      :param scaled_action: (np.ndarray)
      :return: (np.ndarray)
      """
      return self.low + (0.5 * (scaled_action + 1.0) * (self.high -  self.low))

  def reset(self):
    """
    Reset the environment 
    """
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    # Rescale action from [-1, 1] to original [low, high] interval
    rescaled_action = self.rescale_action(action)
    obs, reward, done, info = self.env.step(rescaled_action)
    return obs, reward, done, info

# Used in RL ZOO, for saving vec_normalize state with the best model
class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


# In[7]:


def get_patient_trace(trace_df, step, current_episode, current_episode_step, current_action, current_state, current_obs, 
                      current_obs_new, current_reward,current_episode_return, 
                      current_done, current_patient_info,final_s_outcome,final_r_outcome, env_id):
    sample_time = current_patient_info['sample_time']
    patient_name = current_patient_info['patient_name']
    meal = current_patient_info['meal']
    patient_state = current_patient_info['patient_state'].tolist()
    time = current_patient_info['time']
    year = current_patient_info['time'].year
    month = current_patient_info['time'].month
    day = current_patient_info['time'].day
    hour = current_patient_info['time'].hour
    minute = current_patient_info['time'].minute
    bg = current_patient_info['bg']
    lbgi = current_patient_info['lbgi']
    hbgi = current_patient_info['hbgi']
    risk = current_patient_info['risk']
    
    current_step_info_dict = {'env_id':env_id,'step':step, 'episode':current_episode, 'episode_step':current_episode_step,
                              'action':current_action[0], 'state':current_state,
                              'observation_CGM':current_obs.CGM, 'observation_new_CGM':current_obs_new.CGM,
                              'observation_CHO':current_obs.CHO, 'observation_new_CHO':current_obs_new.CHO,
                              'observation_ROC':current_obs.ROC, 'observation_new_ROC':current_obs_new.ROC,
                              'observation_insulin':current_obs.insulin, 'observation_new_insulin':current_obs_new.insulin,
                                'reward':current_reward, 'episode_return':current_episode_return,
                              'done':current_done, 
                              'sample_time':sample_time, 'patient_name':patient_name, 'meal':meal,
                              'patient_state':patient_state, 'year':year, 'month':month, 'day':day,
                              'hour':hour, 'minute':minute,'bg':bg, 'lbgi':lbgi, 'hbgi':hbgi,'risk':risk, 
                              'final_s_outcome':final_s_outcome, 'final_r_outcome':final_r_outcome}
    #print('current_step_info_dict: ', current_step_info_dict)
    trace_df = trace_df.append(current_step_info_dict, ignore_index=True)
    return trace_df


# In[8]:


# define reward function based on other papers
# in simglucose/simglucose/simulation/env.py.parameter passed is the last hour CGM values
def custom_reward_func_0(BG_last_hour):
    # from paper: Basal Glucose Control in Type 1 Diabetes using Deep Reinforcement Learning: An In Silico Validation
    bg = BG_last_hour[-1]
    if 90<=bg<=140:
        reward = 1
    elif (70<=bg<90) or (140<bg<=180):
        reward = 0.1
    elif 180<bg<=300:
        reward = -0.4-(bg-180)/200
    elif 30<=bg<70:
        reward = -0.6+(bg-70)/100
    else:
        reward = -1
    return reward

def custom_reward_func_1(BG_last_hour, bg_ref=5.5, a=1):
    # from paper: On-line policy learning and adaptation for real-time personalization of an artificial pancreas
    # Gaussian reward function, 1 mg/dL = 0.0555 mmol/L
    # bg_ref: reference value of the glucose concentration, default 80*0.0555=4.44
    # 70*0.0555=3.9, 100*0.555=5.6, 180*0.0555=9.99
    #a: width of the desired glucose band for normoglycemia
    v = -1* pow(BG_last_hour[-1]*0.0555-bg_ref,2)/(2*pow(a,2))
    reward = -1 + math.exp(v) # reward in [-1,0]
    #print('reward func 1.')
    return reward

def custom_reward_func_2(BG_last_hour, bg_average_normal=80):
    # paper: A learning automata-based blood glucose regulation mechanism in type 2 diabetes
    reward = abs(BG_last_hour[-1]-bg_average_normal)/BG_last_hour[-1]
    #print('reward func 2.')
    return reward

def custom_reward_func_3(BG_last_hour, bg_ref=80):
    # paper:Glucose Level Control Using Temporal Difference Methods
    reward = -abs(BG_last_hour[-1]-bg_ref)
    #print('reward func 3.')
    return reward

def risk_func(bg):
    #risk=10*(1.509*math.log(bg, 1.084)-5.381)
    fBG = 1.509 * (np.log(bg)**1.084 - 5.381)
    rl = 10 * fBG[fBG < 0]**2
    rh = 10 * fBG[fBG > 0]**2
    LBGI = np.nan_to_num(np.mean(rl))
    HBGI = np.nan_to_num(np.mean(rh))
    RI = LBGI + HBGI
    #print(RI)
    return RI
def custom_reward_func_4(BG_last_hour):
    # paper: Reinforcement Learning for Blood Glucose Control: Challenges and Opportunities
    if len(BG_last_hour)>=2:
        reward = risk_func(BG_last_hour[-1])-risk_func(BG_last_hour[-2])
    else:
        reward = 0
    #print('reward func 4.')
    return reward

def custom_reward_func_5(BG_last_hour):
    #print('reward func 5.')
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1


# In[9]:


#Calculate set of population stats on the data
#takes in dataframe of BG values
# Example DF:
# pt | time0 | time1 | time2 | ... | time_n
# 0  | 50    | 52    | 55    | ... | 100
# 1  | 130   | 133   | 150   | ... | 330
def calculatePopulationStats(df):
    # Variance of BG for each patient
    varn = df.var(axis=1).mean()
    #print('varn: ', varn)

    # Time Hypo
    total_time = df.shape[1]
    hypoCount = df[df < 70.0].count(axis=1) / total_time
    hypoPercent = hypoCount.mean() * 100

    #Time Hyper
    hyperCount = df[df > 180.0].count(axis=1) / total_time
    hyperPercet = hyperCount.mean() * 100

    #Time in Range
    TIRReal = (total_time - (df[df < 70.0].count(axis=1) + df[df > 180.0].count(axis=1))) / total_time
    TIR = TIRReal.mean() * 100

    #Glycemic Variability Index
    gviReal = getGVI(df)

    #Patient Glycemic Status
    aveBG = df.mean(axis=1)
    TORReal = 1 - TIRReal #time out of range --> BG > 180 or BG < 70
    pgsReal = gviReal * aveBG * TORReal
    avePGS = pgsReal.mean()
    results_dict = {'varn':varn, 'hypoPercent':hypoPercent, 'hyperPercent':hyperPercet, 'TIR':TIR, 
                    'gviReal':gviReal.values[0], 'pgsReal':pgsReal.values[0],'averageBG':aveBG.values[0]}
    return results_dict

def getGVI(df):
    def lineDiff(timeStart, timeEnd, cgmStart, cgmEnd):
        return np.sqrt((timeEnd - timeStart) ** 2 + (abs(cgmEnd - cgmStart)) ** 2)

    diffs = df.diff(axis=1).abs().sum(axis=1)
    expectedDiffs = df.apply(lambda row: lineDiff(0, df.shape[1], row[0], row[df.shape[1]-1]), axis=1)
    return diffs/expectedDiffs

#Plot single cgm trace
def plotEGVTrace(trace, title=None):
    time = range(len(trace))
    plt.figure(figsize=(12, 7))
    plt.plot(time, trace)
    plt.xlabel("Time")
    plt.ylabel("Glucose mg/dL")

    if title != None:
        plt.title(title)

def calculate_aveg_risk_indicators(patient_trace_path):
  patient_trace_df = pd.read_csv(patient_trace_path, index_col=0)
  aveg_LBGI = patient_trace_df['lbgi'].describe()['mean']
  aveg_HBGI = patient_trace_df['hbgi'].describe()['mean']
  aveg_Risk = patient_trace_df['risk'].describe()['mean']
  max_LBGI = patient_trace_df['lbgi'].describe()['max']
  max_HBGI = patient_trace_df['hbgi'].describe()['max']
  max_Risk = patient_trace_df['risk'].describe()['max']

  result_dict = {'aveg_LBGI': aveg_LBGI, 'aveg_HBGI':aveg_HBGI, 'aveg_Risk':aveg_Risk,
           'max_LBGI':max_LBGI, 'max_HBGI':max_HBGI, 'max_Risk':max_Risk}

  return result_dict


# In[ ]:

# get parameters
parser = argparse.ArgumentParser(description='cf diabetic train ppo argparse')
parser.add_argument('--arg_exp_id', '-arg_exp_id', help='exp_id', default=0, type=int)
parser.add_argument('--arg_patient_type', '-arg_patient_type', help='patient type', default='adult', type=str)
#parser.add_argument('--arg_patient_id', '-arg_patient_id', help='patient id', default=1, type=int,required=True)
parser.add_argument('--arg_cuda', '-arg_cuda', help='cuda id', default=1, type=int)
parser.add_argument('--arg_train_step_each_env', '-arg_train_step_each_env', help='train step', default=5000, type=int)
parser.add_argument('--arg_callback_step', '-arg_callback_step', help='callback step', default=1000, type=int)
parser.add_argument('--arg_lr', '-arg_lr', help='learn rate', default=0.001, type=float)
#parser.add_argument('--arg_test_step_each_env', '-arg_test_step', help='test step', default=3400, type=int)
parser.add_argument('--arg_train_round', '-arg_train_round', help='train_round', default=5, type=int)
parser.add_argument('--arg_test_epochs_each_env', '-arg_test_epochs_each_env', help='test_epochs_each_env', default=5, type=int)
parser.add_argument('--arg_max_test_time_each_env', '-arg_max_test_time_each_env', help='max_test_time_each_env', default=3400, type=int)

args_ppo_diabetic = parser.parse_args()

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:{cuda_id}".format(cuda_id=args_ppo_diabetic.arg_cuda))
print(device)
# store envs used for train and test, train with 1,4,5,6, test with 7,8,9
ENV_ID_list = []
patient_id_list = [1,4,5,6,7,8,9]
patient_type = args_ppo_diabetic.arg_patient_type
reward_fun_id = 0
noise_type = 'normal_noise'
render = False
reward = 0
done = False
reward_func_dict = {0: custom_reward_func_0, 1: custom_reward_func_1, 2: custom_reward_func_2, 3: custom_reward_func_3,
                    4: custom_reward_func_4, 5: custom_reward_func_5}
all_env_dict = {}
for id in patient_id_list:
    ENV_NAME = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type, patient_id=id)
    ENV_ID_list.append(ENV_NAME)
    register(
        id='simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type, patient_id=id),
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': '{patient_type}#{id}'.format(patient_type=patient_type, id=str(id).zfill(3)),
                'reward_fun': reward_func_dict[reward_fun_id]}
    )
    # make env for this patient
    #all_env_dict[ENV_NAME] = gym.make(ENV_NAME)
    all_env_dict[ENV_NAME] = Monitor(gym.make(ENV_NAME))

trace_file_folder = 'Counterfactual_Explanation/diabetic_example/trained_ppo_generalize/patient_{patient_type}_lr_{arg_lr}_exp_{arg_exp_id}'.format(patient_type=patient_type,
                                                                                    arg_lr=args_ppo_diabetic.arg_lr, arg_exp_id=args_ppo_diabetic.arg_exp_id)
mkdir(trace_file_folder)
print(trace_file_folder)
test_trace_result_path = '{trace_file_folder}/test_result'.format(trace_file_folder=trace_file_folder)
mkdir(test_trace_result_path)
# Set Callback to save evaluation results
callback_freq=args_ppo_diabetic.arg_callback_step
# learning rate & training step
lr = args_ppo_diabetic.arg_lr
train_step_each_env=args_ppo_diabetic.arg_train_step_each_env

# new best model is saved according to the mean_reward
save_folder = '{trace_file_folder}/save_model'.format(trace_file_folder=trace_file_folder)
mkdir(save_folder)
best_result_folder="{save_folder}/logs/{noise_type}/lr_{lr}".format(save_folder=save_folder,
                                                            noise_type=noise_type,lr=lr)

# eval_callback = EvalCallback(eval_env,
#                              #callback_on_new_best=save_vec_normalize, # save vec_normalize state of the env for best model
#                              best_model_save_path=best_result_folder,
#                              log_path=best_result_folder,
#                              eval_freq=callback_freq,
#                              n_eval_episodes=5,deterministic=True, render=False)
# Save a checkpoint every k steps
checkpoint_callback = CheckpointCallback(
  save_freq=callback_freq,
  save_path="{save_folder}/logs/{noise_type}/lr_{lr}/checkpoint".format(save_folder=save_folder,noise_type=noise_type,lr=lr),
  name_prefix="ppo_diabetic_generalize_model_{noise_type}".format(noise_type=noise_type),
  save_replay_buffer=False,
  save_vecnormalize=True,
)
# Create the callback list
#callback = CallbackList([checkpoint_callback, eval_callback])
callback = CallbackList([checkpoint_callback])


def train_ppo_generalize(train_round, ENV_ID_list, all_env_dict, train_step_each_env, save_folder, test_trace_result_path):
    first_train_flag = 1
    # The noise objects for DDPG
    noise_type = 'normal_noise'
    # set training log
    tmp_path = "{save_folder}/logs/{noise_type}/lr_{lr}/train_log".format(save_folder=save_folder,noise_type=noise_type, lr=lr)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    for i in range(train_round):
        for env_id in ENV_ID_list[0:4]:
            print('train ppo round: ', i, ' env_id: ', env_id)
            idx = ENV_ID_list.index(env_id)
            # make env for this patient
            # all_env_dict[ENV_NAME] = gym.make(ENV_NAME)
            all_env_dict[ENV_NAME] = Monitor(gym.make(ENV_NAME))
            env = all_env_dict[env_id]
            env.reset()
            if first_train_flag==1:
                check_point_mark_0 = '{i}_{idx}'.format(i=i, idx=idx)
                model_save_path_0 = '{save_folder}/ppo_diabetic_generalize_{check_point_mark}'.format(
                    save_folder=save_folder,
                    check_point_mark=check_point_mark_0)
                model = PPO("MlpPolicy", env, verbose=10, learning_rate=lr, tensorboard_log=tmp_path)
                #model.set_logger(new_logger)
                model.learn(total_timesteps=train_step_each_env, callback=callback, reset_num_timesteps=False, tb_log_name='PPO_generalize')
                model.save(model_save_path_0)
                first_train_flag = 0
                model_save_path_1 = model_save_path_0
            else:
                #print('Load model from: ', model_save_path_1)
                model = PPO.load(model_save_path_1)
                model.set_env(env, force_reset=True)
                #model.set_logger(new_logger)
                model.learn(total_timesteps=train_step_each_env, callback=callback, reset_num_timesteps=False, tb_log_name='PPO_generalize')
                check_point_mark_1 = '{i}_{idx}'.format(i=i, idx=idx)
                model_save_path_1 = '{save_folder}/ppo_diabetic_generalize_{check_point_mark}'.format(
                    save_folder=save_folder,
                    check_point_mark=check_point_mark_1)
                model.save(model_save_path_1)
            env.close()
        test_ppo_generalize(model_save_path_1, ENV_ID_list, all_env_dict,
                            test_epochs_each_env=args_ppo_diabetic.arg_test_epochs_each_env,
                            max_test_time_each_env=args_ppo_diabetic.arg_max_test_time_each_env,
                            train_round=i, results_folder=test_trace_result_path)

    return

def test_ppo_generalize(ppo_model_path, ENV_ID_list, all_env_dict, test_epochs_each_env, max_test_time_each_env, train_round, results_folder):
    trace_file_path = '{folder}/test_patient_trace_rf_{rf}_round_{train_round}.csv'.format(
        folder=results_folder, rf=reward_fun_id, train_round=train_round)
    trace_df_list = []

    for env_id in ENV_ID_list[4:]:
        trace_len = 0
        test_env = all_env_dict[env_id]
        # test the best model in a new env
        print('Begin testing: ', env_id)
        result_col = ['id', 'hypoPercent', 'hyperPercent', 'TIR', 'step', 'reward_func']
        results_path = '{folder}/medical_metrics_results_{env_id}_train_{train_round}.csv'.format(folder=results_folder,
                                                                                                  env_id=env_id,train_round=train_round)
        result_df = pd.DataFrame(columns=result_col)
        test_env.training = False
        test_env.norm_reward = False
        test_model = PPO.load(ppo_model_path, env=test_env)
        observation = test_env.reset()
        # store the past trace of this epoch
        trace_df = pd.DataFrame(columns=['env_id', 'step', 'episode', 'episode_step', 'action', 'state',
                                         'observation_CGM', 'observation_new_CGM',
                                         'observation_ROC', 'observation_new_ROC',
                                         'observation_CHO', 'observation_new_CHO',
                                         'observation_insulin', 'observation_new_insulin',
                                         'reward', 'episode_return', 'done',
                                         'sample_time', 'patient_name', 'meal',
                                         'patient_state', 'year', 'month', 'day',
                                         'hour', 'minute', 'bg', 'lbgi', 'hbgi', 'risk',
                                         'final_s_cgm_outcome', 'final_s_cho_outcome',
                                         'final_s_roc_outcome',
                                         'final_s_insulin_outcome',
                                         'final_r_outcome'],
                                dtype=object)

        num_episodes = 0
        for epoch in range(test_epochs_each_env):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            # num_episodes = 0
            episode_return = 0
            episode_length = 0

            # Iterate over the steps of each epoch
            for t in range(max_test_time_each_env):
                # print(t)
                if render:
                    test_env.render()
                # Get the logits, action, and take one step in the environment
                action, _states = test_model.predict(observation)
                # print('action: ', action)
                observation_new, reward, done, info = test_env.step(action)
                episode_return += reward
                episode_length += 1
                trace_df = get_patient_trace(trace_df, t, num_episodes, episode_length, action, _states, observation,
                                             observation_new, reward, episode_return,
                                             done, info, 0, 0, env_id)

                # Update the observation
                # print('OBS: ', observation, ' OBS new: ', observation_new)
                observation = observation_new
                # Finish trajectory if reached to a terminal state
                terminal = done
                if (t == max_test_time_each_env - 1) or terminal:
                    # print(trace_df)
                    final_s_outcome_this_eps = [observation.CGM, observation.CHO, observation.ROC, observation.insulin]
                    final_r_outcome_this_eps = episode_return  # the final total reward of this episode
                    trace_df = trace_df.astype('object')
                    # print(trace_df.final_s_outcome[trace_df.episode == num_episodes])
                    # print(episode_length, final_s_outcome_this_eps)
                    trace_df.final_s_cgm_outcome[trace_df.episode == num_episodes] = [final_s_outcome_this_eps[
                                                                                          0]] * episode_length
                    trace_df.final_s_cho_outcome[trace_df.episode == num_episodes] = [final_s_outcome_this_eps[
                                                                                          1]] * episode_length
                    trace_df.final_s_roc_outcome[trace_df.episode == num_episodes] = [final_s_outcome_this_eps[
                                                                                          2]] * episode_length
                    trace_df.final_s_insulin_outcome[trace_df.episode == num_episodes] = [final_s_outcome_this_eps[
                                                                                              3]] * episode_length
                    trace_df.final_r_outcome[trace_df.episode == num_episodes] = [
                                                                                     final_r_outcome_this_eps] * episode_length

                    observation_CGM_list = []
                    observation_CGM_new_list = []
                    observation_CHO_list = []
                    observation_CHO_new_list = []
                    observation_ROC_list = []
                    observation_ROC_new_list = []
                    observation_insulin_list = []
                    observation_insulin_new_list = []

                    for index, row in trace_df[trace_df.episode == num_episodes].iterrows():
                        observation_CGM_list.append(row['observation_CGM'])
                        observation_CGM_new_list.append(row['observation_new_CGM'])
                        observation_CHO_list.append(row['observation_CHO'])
                        observation_CHO_new_list.append(row['observation_new_CHO'])
                        observation_ROC_list.append(row['observation_ROC'])
                        observation_ROC_new_list.append(row['observation_new_ROC'])
                        observation_insulin_list.append(row['observation_insulin'])
                        observation_insulin_new_list.append(row['observation_new_insulin'])

                    trace_df.observation_CGM[trace_df.episode == num_episodes] = observation_CGM_list
                    trace_df.observation_new_CGM[trace_df.episode == num_episodes] = observation_CGM_new_list
                    trace_df.observation_CHO[trace_df.episode == num_episodes] = observation_CHO_list
                    trace_df.observation_new_CHO[trace_df.episode == num_episodes] = observation_CHO_new_list
                    trace_df.observation_ROC[trace_df.episode == num_episodes] = observation_ROC_list
                    trace_df.observation_new_ROC[trace_df.episode == num_episodes] = observation_ROC_new_list
                    trace_df.observation_insulin[trace_df.episode == num_episodes] = observation_insulin_list
                    trace_df.observation_new_insulin[trace_df.episode == num_episodes] = observation_insulin_new_list

                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = test_env.reset(), 0, 0

            trace_df_list.append(trace_df)
            trace_len += len(trace_df)
        trace_len /= num_episodes
        print('Average trace len this patient: ', trace_len)
    trace_df_all = pd.concat(trace_df_list)
    trace_df_all.to_csv(trace_file_path)
    # # get medical metric values for each patient
    # patient_type_list = ['adult']
    # patient_id_num_list = [7, 8, 9]
    # for id in ENV_ID_list:
    #         BG_df = pd.DataFrame()
    #         df = pd.read_csv(trace_file_path)
    #         df = df[df['env_id']==id]
    #         part_df = df[['observation_CGM']]
    #         part_df = part_df.rename(columns={"observation_CGM": id})
    #         BG_df[id] = part_df[id]
    #         bg_df = BG_df.T
    #         results_dict_1 = calculatePopulationStats(bg_df)
    #         results_dict_1['id'] = id
    #         results_dict_1['step'] = len(bg_df)
    #         results_dict_1['reward_func'] = reward_fun_id
    #         results_dict_1['patient_type'] = patient_type
    #         results_dict_2 = calculate_aveg_risk_indicators(trace_file_path)
    #         final_result = results_dict_1.copy()
    #         final_result.update(results_dict_2)
    #         result_df = result_df.append(final_result, ignore_index=True)
    # print('medical result_df: ', result_df)
    # result_df.to_csv(results_path)

    return


train_ppo_generalize(args_ppo_diabetic.arg_train_round, ENV_ID_list, all_env_dict, train_step_each_env, save_folder, test_trace_result_path)


