#!/usr/bin/env python
# coding: utf-8

# import package
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import gym
import numpy as np
import pandas as pd
import math
import torch
import random
import time
import shutil
import psutil
import gc
gc.set_threshold(100*1024*1024)

from stable_baselines3 import PPO
from stable_baselines3 import DDPG_CF, TD3_CF
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

from typing import Optional
from datetime import datetime

from gym.envs.registration import register
from gym.wrappers.normalize_cf import NormalizeObservation

pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)
import sys
# sys.path.append("/anaconda3/envs/CF_diabetic/lib/python3.8/site-packages/gym")
# sys.path.append("/anaconda3/envs/CF_diabetic/lib/python3.8/site-packages/simglucose")
sys.path.append("/.conda/envs/CF_diabetic/lib/python3.8/site-packages/gym")
sys.path.append("/.conda/envs/CF_diabetic/lib/python3.8/site-packages/simglucose")
# modified the core.py in gym==0.21.0 to make it work

from gym import cf_generator
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from random import sample
from collections import namedtuple
import argparse
from memory_profiler import profile
#Observation = namedtuple('Observation', ['CGM', 'CHO', 'ROC','insulin', 'BG'])

#
def get_mem_use():
    mem=psutil.virtual_memory()
    mem_gb = mem.used/(1024*1024*1024)
    return mem_gb
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
        assert isinstance(action_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
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
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

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
    results_dict = {'varn':varn, 'hypoPercent':hypoPercent, 'hyperPercent':hyperPercet, 'TIR':TIR, 'gviReal':gviReal.values[0], 'pgsReal':pgsReal.values[0]}
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

def store_time_index(ENV_NAME, time_index_list):
    # save {eps_idx: time_index} of training and testing
    df = pd.DataFrame(columns=['ENV_NAME','orig_episode', 'orig_end_time_index'])
    orig_episode_list = []
    orig_time_index_list = []
    for info_dict in time_index_list:
        for k, v in info_dict.items():
            trace_eps, current_time_step = k, v
        orig_episode_list.append(trace_eps)
        orig_time_index_list.append(current_time_step)
    df['orig_episode'] = orig_episode_list
    df['orig_end_time_index'] = orig_time_index_list
    df['ENV_NAME'] = [ENV_NAME]*len(orig_episode_list)
    #df.to_csv(save_file_path)
    return df

def calculate_effect_value_sequence(actionable_feature_sequence, current_time_index, time_horizon):
    action_idx_list = list(range(current_time_index - time_horizon + 1, current_time_index + 1, 1))
    for idx in range(len(action_idx_list)):
        action_idx_list[idx] = 1 - (current_time_index - action_idx_list[idx]) / (time_horizon + 1)
    action_idx_list = torch.tensor(action_idx_list)
    if len(actionable_feature_sequence) != len(action_idx_list):
        print('Different length, ', actionable_feature_sequence, action_idx_list)
    #print('action_idx_list: ', action_idx_list)
    #print('actionable_feature_sequence: ', actionable_feature_sequence)
    effect_value = torch.sum(torch.tensor(actionable_feature_sequence)*action_idx_list).item()
    return effect_value

def add_action_effect_to_trace(trace_df):
    action_effect_list = [] #iob
    max_time_horizon = 100 #5h
    total_eps_num = max(trace_df['episode'].tolist())
    for eps in range(total_eps_num+1):
        #print('eps: ', eps)
        this_trace = trace_df[trace_df['episode']==eps]
        for i in range(len(this_trace)):
            if i<max_time_horizon:
                past_action_seq = this_trace[:i+1]['action'].tolist()
                #print(i, 'this_trace index: ', list(this_trace[:i+1]['action'].index))
                time_horizon = i+1
            elif i>=max_time_horizon:
                past_action_seq = this_trace[i-time_horizon+1:i+1]['action'].tolist()
                time_horizon = max_time_horizon
                #print(i, 'this_trace index: ', list(this_trace[i-time_horizon+1:i+1]['action'].index))
            #print('len(past_action_seq): ', len(past_action_seq))
            effect_value = calculate_effect_value_sequence(past_action_seq, i, time_horizon)
            action_effect_list.append(effect_value)
    trace_df['action_effect'] = action_effect_list
    return trace_df

def get_patient_trace(env, trace_df, step, current_episode, current_episode_step, current_action, current_state,
                      current_obs,
                      current_obs_new, current_reward, current_accumulated_reward,
                      current_done, current_patient_info, final_s_outcome, final_r_outcome, patient_type, patient_id):
    # store the state info of patient, sensor, scenario for reseting while generating cf
    patient_state = current_patient_info['patient_state']#.tolist()
    time = (step + 1) * current_patient_info['sample_time']
    meal = current_patient_info['meal']
    T1DSimEnv_this_patient = env.env.env.env
    #print('env: ', T1DSimEnv_this_patient)
    #last_foodtaken = str(env.patient._last_foodtaken)
    last_foodtaken = str(T1DSimEnv_this_patient.patient._last_foodtaken)
    patient_Gsub = T1DSimEnv_this_patient.patient.observation.Gsub
    is_eating = T1DSimEnv_this_patient.patient.is_eating

    cgm_sensor_params = T1DSimEnv_this_patient.sensor._params  # sensor params should not change within the same episode, no need to convert to str
    cgm_sensor_seed = T1DSimEnv_this_patient.sensor.seed
    cgm_sensor_last_cgm = str(T1DSimEnv_this_patient.sensor._last_CGM)

    scenario_seed = 0 #T1DSimEnv_this_patient.scenario.seed
    scenario_random_gen = None # T1DSimEnv_this_patient.scenario.random_gen
    scenario = T1DSimEnv_this_patient.scenario.scenario

    sample_time = current_patient_info['sample_time']
    patient_name = current_patient_info['patient_name']
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

    current_step_info_dict = {'patient_type':patient_type, 'patient_id':patient_id,
        'step': step, 'episode': current_episode, 'episode_step': current_episode_step,
                              'time': time,
                              'action': current_action[0], 'state': current_state,
                              'observation_CGM':current_obs.CGM, 'observation_new_CGM':current_obs_new.CGM,
                              'observation_CHO':current_obs.CHO, 'observation_new_CHO':current_obs_new.CHO,
                              'observation_ROC':current_obs.ROC, 'observation_new_ROC':current_obs_new.ROC,
                              'observation_insulin':current_obs.insulin, 'observation_new_insulin':current_obs_new.insulin,
                              'reward': current_reward, 'accumulated_reward': current_accumulated_reward,
                              'done': current_done,
                              'sample_time': sample_time, 'patient_name': patient_name, 'meal': meal,'is_eating':is_eating,
                              'patient_state': patient_state, 'last_foodtaken': float(last_foodtaken),
                              'cgm_sensor_params': cgm_sensor_params, 'cgm_sensor_seed': int(cgm_sensor_seed),
                              'cgm_sensor_last_cgm': float(cgm_sensor_last_cgm),
                              'scenario_seed': int(scenario_seed), 'scenario_random_gen': scenario_random_gen,
                              'scenario': scenario,
                              'year': year, 'month': month, 'day': day,
                              'hour': hour, 'minute': minute, 'bg': bg, 'lbgi': lbgi, 'hbgi': hbgi, 'risk': risk,
                              'final_s_outcome': final_s_outcome, 'final_r_outcome': final_r_outcome}
    #print('current_step_info_dict: ', patient_Gsub, current_step_info_dict['observation_CGM'], current_step_info_dict['cgm_sensor_last_cgm'])
    trace_df = trace_df.append(current_step_info_dict, ignore_index=True)
    return trace_df

def static_accumulate_reward(accumulated_reward_df, improve_percentage_thre=0.1):
    id_list = list(set(accumulated_reward_df['id'].to_list()))
    aveg_accumulated_difference_df = pd.DataFrame(columns=['id', 'aveg_accumulated_reward_difference', 'aveg_improve_percentage',
                                                           'max_accumulated_reward_difference', 'max_improve_percentage'])
    better_difference_count = 0
    max_difference_count = 0
    #better_improve_percentage_ = 0
    for id in id_list:
        df = accumulated_reward_df[accumulated_reward_df['id']==id]
        aveg_difference = df['difference'].mean()
        max_difference = df['difference'].max()
        aveg_improve_percentage = aveg_difference/abs(df['orig_accumulated_reward'].mean()) if df['orig_accumulated_reward'].mean()!=0 else -9999
        max_improve_percentage = max_difference / abs(df['orig_accumulated_reward'].mean()) if df['orig_accumulated_reward'].mean() != 0 else -9999
        dict = {'id':id, 'aveg_accumulated_reward_difference':aveg_difference, 'aveg_improve_percentage':aveg_improve_percentage,
                'max_accumulated_reward_difference':max_difference, 'max_improve_percentage':max_improve_percentage}
        aveg_accumulated_difference_df = aveg_accumulated_difference_df.append(dict, ignore_index=True)
        if aveg_difference>0:
            better_difference_count += 1
        if max_difference>0:
            max_difference_count += 1
    better_difference_perc = better_difference_count/len(id_list)
    max_difference_perc = max_difference_count / len(id_list)
    improve_percentage = len(aveg_accumulated_difference_df[aveg_accumulated_difference_df['aveg_improve_percentage']>=improve_percentage_thre])/len(aveg_accumulated_difference_df)
    return round(better_difference_perc, 3), round(max_difference_perc, 3), round(improve_percentage,3)

def convert_time_str(time_str):
    #format = '%Y-%m-%d %H:%M:%S'
    #time = datetime.strptime(time_str, format)
    time = pd.to_datetime(time_str)
    #print('time_str: ', time_str, ' time: ', time, type(time))
    return time

def convert_patient_state(patient_state_str):
    #print('patient_state_str: ', patient_state_str)
    patient_state_number_list = []
    patient_state_str_list = patient_state_str[1:-1].split(' ')
    #print('patient_state_str_list: ', patient_state_str_list)
    for str in patient_state_str_list:
        if str!='':
            if str[-2:]=='\n':
                str = str[:-2]
            patient_state_number_list.append(float(str)) #convert str to float
    #print('patient_state_number_list: ', patient_state_number_list)
    return patient_state_number_list

def convert_scenario_0(scenario_str):
    # {'meal': {'time': [472.0, 767.0, 1121.0], 'amount': [66, 66, 75]}}
    #print('scenario_str: ', scenario_str)
    dict = {}
    t_list = []
    amount_list = []
    print(scenario_str)
    str_1 = scenario_str[1:-1].split('{')[1][:-1] # 'time': [472.0, 767.0, 1121.0], 'amount': [66, 66, 75]
    #print('str_1: ', str_1)
    time_str_list = str_1.split('amount')[0][9:-3].split(', ')
    #print('time_str_list: ', time_str_list)
    amount_str_list = str_1.split('amount')[1][3:-1].split(', ')
    #print('amount_str_list: ', amount_str_list)
    for t in time_str_list:
        if t!='':
            if t[0]=='[':
                t = t[1:]
            elif t[-1]==']':
                t = t[:-1]
            t_list.append(float(t))
    for a in amount_str_list:
        if a!='':
            if a[0]=='[':
                a = a[1:]
            elif a[-1]==']':
                a = a[:-1]
            amount_list.append(float(a))
    dict = {'meal':{'time':t_list, 'amount':amount_list}}
    #print('dict: ', dict)
    return dict

def convert_scenario(scenario_str):
    # {'meal': {'time': [472.0, 767.0, 1121.0], 'amount': [66, 66, 75]}}
    #print('scenario_str: ', scenario_str)
    t_list = []
    amount_list = []
    #print(scenario_str[2:-2].split('), ('))
    list_1 = scenario_str[2:-2].split('), (')
    for l in list_1:
        time, amount = [float(x) for x in l.split(', ')]
        #print(time, amount)
        t_list.append(time)
        amount_list.append(amount)
    # str_1 = scenario_str[1:-1].split('{')[1][:-1] # 'time': [472.0, 767.0, 1121.0], 'amount': [66, 66, 75]
    # #print('str_1: ', str_1)
    # time_str_list = str_1.split('amount')[0][9:-3].split(', ')
    # #print('time_str_list: ', time_str_list)
    # amount_str_list = str_1.split('amount')[1][3:-1].split(', ')
    # #print('amount_str_list: ', amount_str_list)
    # for t in time_str_list:
    #     if t!='':
    #         if t[0]=='[':
    #             t = t[1:]
    #         elif t[-1]==']':
    #             t = t[:-1]
    #         t_list.append(float(t))
    # for a in amount_str_list:
    #     if a!='':
    #         if a[0]=='[':
    #             a = a[1:]
    #         elif a[-1]==']':
    #             a = a[:-1]
    #         amount_list.append(float(a))
    #dict = {'meal':{'time':t_list, 'amount':amount_list}}
    #print('scenario dict: ', dict)
    scenario_list = list(zip(t_list, amount_list))
    #print('scenario_list: ',scenario_list)
    #print('dict: ', dict)
    return scenario_list

def set_fixed_orig_trace(orig_trace_df, save_path, orig_id, patient_type, patient_id):
    #print('Set the fixed orig traces.')
    #print('orig trace time type: ', trace_df['time'].tolist()[0], type(trace_df['time'].tolist()[0]))
    cgm_sensor_params = [trace_df['cgm_sensor_params'].tolist()[0]]*len(orig_trace_df)
    scenario_random_gen = [trace_df['scenario_random_gen'].tolist()[0]] * len(orig_trace_df)
    patient_state_list = []
    time_list = []
    scenario_dict_list = []
    done_list = []
    for item, row in orig_trace_df.iterrows():
        # convert time
        #print('orig trace time type: ', trace_df['time'].tolist()[0], type(trace_df['time'].tolist()[0]))
        time_str = row['time']
        time = convert_time_str(time_str)
        patient_state = convert_patient_state(row['patient_state'])
        scenario_dict = convert_scenario(row['scenario'])
        time_list.append(time)
        patient_state_list.append(patient_state)
        scenario_dict_list.append(scenario_dict)
        if row['done']=='FALSE':
            done_list.append(False)
        else:
            done_list.append(True)
    orig_trace_df['time'] = time_list
    orig_trace_df['patient_state'] = patient_state_list
    orig_trace_df['scenario'] = scenario_dict_list
    orig_trace_df['done'] = done_list
    orig_trace_df['cgm_sensor_params'] = cgm_sensor_params
    orig_trace_df['scenario_random_gen'] = scenario_random_gen
    trace_df.to_csv('{save_path}/original_trace_convert_{orig_id}_p_{patient_type}_{patient_id}.csv'.format(save_path=save_path,
                                                                                    orig_id=orig_id, patient_type=patient_type,
                                                                                                            patient_id=patient_id))
    return orig_trace_df
# def convert_str_to_list(str, split_type=","):
#     # lander_fixtures_vertices, lander_linearVelocity, lander_mass_center, lander_position,
#     # leg_0_fixtures_vertices, leg_0_linearVelocity, leg_0_mass_center, leg_0_position
#     # leg_1_fixtures_vertices, leg_1_linearVelocity, leg_1_mass_center, leg_1_position
#     # joint_0_limits, joint_1_limits,
#     str_1 = str[1:-1]
#     str_list = [float(x) for x in str_1.split(split_type)]
#     #print('Before convert str: ', str_1, type(str_1))
#     #print('After convert str: ', str_list, type(str_list))
#     return str_list

# def get_patient_trace_from_save_data(trace_df):
#     all_patient_state = []
#     for item, row in trace_df.iterrows():
#         patient_state_str = row['patient_state']
#         patient_state_list = convert_str_to_list(patient_state_str, split_type=",")
#         all_patient_state.append(patient_state_list)
#     trace_df['patient_state'] = all_patient_state
#
#     return trace_df

# get parameters
parser = argparse.ArgumentParser(description='cf diabetic argparse')
# parser.add_argument('--arg_exp_id', '-arg_id', help='experiment id', default=0, type=int, required=True)
# parser.add_argument('--arg_whole_trace_step', '-arg_wts', help='total length of the original traces', default=2000, type=int,required=True)
# parser.add_argument('--arg_cuda', '-arg_cuda', help='cuda id', default=1, type=int,required=True)
# parser.add_argument('--arg_cflen', '-arg_cflen', help='length of CF traces', default=20, type=int,required=True)
# parser.add_argument('--arg_train_eps', '-arg_train_eps', help='training eps after sample a new original trace', default=10, type=int,required=True)
# parser.add_argument('--arg_test_eps', '-arg_test_eps', help='testing eps after sample a new original trace', default=10, type=int,required=True)
# parser.add_argument('--arg_update_iteration', '-arg_update_iteration', help='ddpg update iterration', default=10, type=int,required=True)
# parser.add_argument('--arg_train_split', '-arg_train_split', help='split for training set', default=0.8, type=float,required=True)
parser.add_argument('--arg_exp_id', '-arg_id', help='experiment id', default=0, type=int)
parser.add_argument('--arg_whole_trace_step', '-arg_wts', help='total length of the original traces', default=500, type=int)
parser.add_argument('--arg_cuda', '-arg_cuda', help='cuda id', default=0, type=int)
parser.add_argument('--arg_cflen', '-arg_cflen', help='length of CF traces', default=20, type=int)
#parser.add_argument('--arg_train_eps', '-arg_train_eps', help='training eps after sample a new original trace', default=10, type=int)
#parser.add_argument('--arg_test_eps', '-arg_test_eps', help='testing eps after sample a new original trace', default=10, type=int)
#parser.add_argument('--arg_update_iteration', '-arg_update_iteration', help='ddpg update iterration', default=10, type=int)
parser.add_argument('--arg_train_split', '-arg_train_split', help='split for training set', default=0.8, type=float)

parser.add_argument('--arg_total_used_num', '-arg_total_used_num', help='total used number of traces for training and testing', default=5, type=int)
#parser.add_argument('--arg_actor_lr', '-arg_actor_lr', help='actor learning rate', default=0.0001, type=float)
#parser.add_argument('--arg_critic_lr', '-arg_critic_lr', help='critic learning rate', default=0.001, type=float)
#parser.add_argument('--arg_lambda_lr', '-arg_lambda_lr', help='lambda learning rate', default=0.001, type=float)
parser.add_argument('--arg_patidim', '-arg_patidim', help='encoded patient state dim', default=3, type=int)
#parser.add_argument('--arg_improve_percentage_thre', '-arg_improve_percentage_thre', help='improve percentage threshold for better cf', default=0.1, type=float)
#parser.add_argument('--arg_train_sample_num', '-arg_train_sample_num', help='number of samples generated in training to push in buffer for each orig trace', default=150, type=int)
parser.add_argument('--arg_train_round', '-arg_train_round', help='train round for all training index', default=3, type=int)
#parser.add_argument('--arg_train_mark', '-arg_train_mark', help='train mark for updating model', default=3, type=int)
parser.add_argument('--arg_with_encoder', '-arg_with_encoder', help='if using encoder in training', default=0, type=int)
parser.add_argument('--arg_iob_param', '-arg_iob_param', help='param for calculate iob', default=0.15, type=float)
parser.add_argument('--arg_dist_func', '-arg_dist_func', help='assign distance loss function', default='dist_pairwise', type=str)
#parser.add_argument('--arg_clip_value', '-arg_clip_value', help='gradient clip value for actor', default=0, type=float)
parser.add_argument('--arg_delta_dist', '-arg_delta_dist', help='delta value used in distance function, how much is CF action different from orig action at 1 step', default=0.5, type=float)
#parser.add_argument('--arg_test_mark', '-arg_test_mark', help='marker for test', default=30, type=int)
parser.add_argument('--arg_train_one_trace', '-arg_train_one_trace', help='flag for training for 1 trace', default=0, type=int)

parser.add_argument('--arg_lambda_start_value', '-arg_lambda_start_value', help='lambda_start_value', default=1.0, type=float)
parser.add_argument('--arg_generate_train_trace_num', '-arg_generate_train_trace_num', help='generate_train_trace_num', default=100, type=int)
parser.add_argument('--arg_total_test_trace_num', '-arg_total_test_trace_num', help='total_test_trace_num', default=100, type=int)
#parser.add_argument('--arg_total_train_steps', '-arg_total_train_steps', help='total_train_steps', default=3500, type=int)
parser.add_argument('--arg_log_interval', '-arg_log_interval', help='log_interval', default=50, type=int)
parser.add_argument('--arg_batch_size', '-arg_batch_size', help='batch_size', default=256, type=int)
parser.add_argument('--arg_gradient_steps', '-arg_gradient_steps', help='arg_gradient_steps', default=250, type=int)
parser.add_argument('--arg_epsilon', '-arg_epsilon', help='epsilon', default=0.1, type=float)
parser.add_argument('--arg_total_timesteps_each_trace', '-arg_total_timesteps_each_trace', help='train time steps after sampling a new orig trace', default=500, type=int)
parser.add_argument('--arg_run_baseline', '-arg_run_baseline', help='run_baseline', default=0, type=int)
parser.add_argument('--arg_learning_rate', '-arg_learning_rate', help='learning_rate for SB3 actor and critic', default=0.00001, type=float)
parser.add_argument('--arg_thre_for_fix_a', '-arg_thre_for_fix_a', help='threshold for fixing action', default=0.1, type=float)
parser.add_argument('--arg_if_user_assign_action', '-arg_if_user_assign_action', help='if use the action assigned by user', default=0, type=int)
parser.add_argument('--arg_if_single_env', '-arg_if_single_env', help='if only use 1 env to generate traces', default=1, type=int)
parser.add_argument('--arg_assigned_patient_id', '-arg_assigned_patient_id', help='assigned patient id for 1 env case', default=5, type=int)
parser.add_argument('--arg_thread', '-arg_thread', help='thread for running exp', default=1, type=int)
parser.add_argument('--arg_if_start_state_restrict', '-arg_if_start_state_restrict', help='set constrain on start state in seg', default=0, type=int)
parser.add_argument('--arg_start_state_id', '-arg_start_state_id', help='id for start state interval', default=0, type=int)
parser.add_argument('--arg_if_meal_restrict', '-arg_if_meal_restrict', help='set constrain on meal in seg', default=0, type=int)
parser.add_argument('--arg_total_r_thre', '-arg_total_r_thre', help='threshold set for total reward for orig trace', default=20.0, type=float)
parser.add_argument('--arg_min_cgm', '-arg_min_cgm', help='threshold set for arg_min_cgm when choosing segments', default=65.0, type=float)
parser.add_argument('--arg_total_trial_num', '-arg_total_trial_num', help='total_trial_num', default=1, type=int)
parser.add_argument('--arg_reward_weight', '-arg_reward_weight', help='reward weight', default=-1, type=float)
parser.add_argument('--arg_test_param', '-arg_test_param', help='test param', default=10, type=int)
parser.add_argument('--arg_ppo_train_time', '-arg_ppo_train_time', help='ppo training time', default=10, type=int)
parser.add_argument('--arg_use_exist_trace', '-arg_use_exist_trace', help='if use exist trace for exp', default=0, type=int)
parser.add_argument('--arg_exist_trace_id', '-arg_exist_trace_id', help='exist trace id', default=1, type=int)
parser.add_argument('--arg_exp_type', '-arg_exp_type', help='experiment type (1, 2, 3)', default=1, type=int)
parser.add_argument('--arg_if_constrain_on_s', '-arg_if_constrain_on_s', help='if set any constraint on state (RP 2)', default=0, type=int)
parser.add_argument('--arg_thre_for_s', '-arg_thre_for_s', help='threshold for state (RP 2)', default=-999999, type=float)
parser.add_argument('--arg_s_index', '-arg_s_index', help='index of state vector for the one to be constrained (RP 2)', default=2, type=int)
parser.add_argument('--arg_if_use_user_input', '-arg_if_use_user_input', help='if use user input action (RP 3)', default=0, type=int)
parser.add_argument('--arg_user_input_action', '-arg_user_input_action', help='value of user input action (RP 3)', default=0.00, type=float)

args_cf_diabetic = parser.parse_args()
torch.set_num_threads(args_cf_diabetic.arg_thread)

ENV_ID_list = []
#patient_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
patient_id_list = [4, 5, 6, 7, 8, 9]
#patient_id = 5
patient_type = 'adult'
reward_fun_id = 0
noise_type = 'normal_noise'
lr = args_cf_diabetic.arg_learning_rate
#train_step = 300
render = False
reward = 0
done = False
reward_func_dict = {0: custom_reward_func_0, 1: custom_reward_func_1, 2: custom_reward_func_2, 3: custom_reward_func_3,
                    4: custom_reward_func_4, 5: custom_reward_func_5}
trained_controller_dict = {}
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
    if args_cf_diabetic.arg_if_single_env==1:
        print('Load personalized PPO model.')
        # Load the saved statistics, env and model of outside controller, trained for each patient
        # ppo_folder = "/home/cpsgroup/Counterfactual_Explanation/diabetic_example/trained_ppo_{patient_type}".format(
        #     patient_type=patient_type)
        ppo_folder = "/p/citypm/AIFairness/Counterfactual_Explanation/diabetic_example/trained_ppo_{patient_type}".format(
            patient_type=patient_type)
        ppo_model_path = '{ppo_folder}/ppo_diabetic_model_{patient_type}_{patient_id}_trained_{arg_ppo_train_time}'.format(
            ppo_folder=ppo_folder, patient_type=patient_type, patient_id=id, arg_ppo_train_time=args_cf_diabetic.arg_ppo_train_time)
        # ppo_model_path = '{ppo_folder}/ppo_diabetic_model_{patient_type}_{patient_id}'.format(
        #     ppo_folder=ppo_folder,
        #     patient_type=patient_type, patient_id=id)
        stats_path = '{best_result_folder}/vecnormalize.pkl'.format(best_result_folder=ppo_folder)
        # Load the agent and save in dict
        trained_controller_dict[ENV_NAME] = PPO.load(ppo_model_path, env=all_env_dict[ENV_NAME],
                              custom_objects={'observation_space': all_env_dict[ENV_NAME].observation_space,
                                              'action_space': all_env_dict[ENV_NAME].action_space})
    else:
        print('Load generalized PPO model.')
        # Load the saved statistics, env and model of outside controller, trained with generalization
        # ppo_folder = "/home/cpsgroup/Counterfactual_Explanation/diabetic_example/trained_ppo_{patient_type}_generalize_789".format(
        #     patient_type=patient_type)
        ppo_folder = "/p/citypm/AIFairness/Counterfactual_Explanation/diabetic_example/trained_ppo_{patient_type}_generalize_789".format(
            patient_type=patient_type)
        ppo_model_path = '{ppo_folder}/ppo_diabetic_model_{patient_type}_generalize_trained_{arg_ppo_train_time}'.format(
            ppo_folder=ppo_folder, arg_ppo_train_time=args_cf_diabetic.arg_ppo_train_time,patient_type=patient_type)
        # ppo_model_path = '{ppo_folder}/ppo_diabetic_model_{patient_type}_generalize_789_179_2'.format(
        #     ppo_folder=ppo_folder,
        #     patient_type=patient_type)
        stats_path = '{best_result_folder}/vecnormalize.pkl'.format(best_result_folder=ppo_folder)
        # Load the agent and save in dict
        trained_controller_dict[ENV_NAME] = PPO.load(ppo_model_path, env=all_env_dict[ENV_NAME],
                                                     custom_objects={
                                                         'observation_space': all_env_dict[ENV_NAME].observation_space,
                                                         'action_space': all_env_dict[ENV_NAME].action_space})

#device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:{cuda_id}".format(cuda_id=args_cf_diabetic.arg_cuda))
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
print(device, ' Thread: ', args_cf_diabetic.arg_thread)
print('whole_trace_step: ', args_cf_diabetic.arg_whole_trace_step, ' check_len: ', args_cf_diabetic.arg_cflen,
      ' arg_generate_train_trace_num: ', args_cf_diabetic.arg_generate_train_trace_num, ' arg_total_test_trace_num: ', args_cf_diabetic.arg_total_test_trace_num,
      ' arg_batch_size: ', args_cf_diabetic.arg_batch_size, ' dist_func: ', args_cf_diabetic.arg_dist_func,
      ' arg_gradient_steps: ', args_cf_diabetic.arg_gradient_steps, ' arg_delta_dist: ', args_cf_diabetic.arg_delta_dist,
      ' arg_learning_rate: ', args_cf_diabetic.arg_learning_rate)

# print(type(args_cf_diabetic.arg_whole_trace_step), type(args_cf_diabetic.arg_cuda), type(args_cf_diabetic.arg_cflen),
#         type(args_cf_diabetic.arg_train_eps), type(args_cf_diabetic.arg_test_eps), type(args_cf_diabetic.arg_update_iteration),
#       type(args_cf_diabetic.arg_train_split))
test_step = args_cf_diabetic.arg_whole_trace_step #100000

#diabetic_file_folder = '/home/cpsgroup/Counterfactual_Explanation/diabetic_example/results'
diabetic_file_folder = '/p/citypm/AIFairness/Counterfactual_Explanation/diabetic_example'
all_folder = '{diabetic_file_folder}/TD3_results_slurm_e{arg_exp_type}_RP3_S_UET'.format(arg_exp_type=args_cf_diabetic.arg_exp_type,diabetic_file_folder=diabetic_file_folder)
if os.path.exists(all_folder):
    pass
else:
    mkdir(path=all_folder)
save_folder = '{all_folder}/all_trace_len_{test_step}_{dist_func}_lr_{arg_learning_rate}_grad_{arg_gradient_steps}_{arg_exp_id}'.format(test_step=test_step,all_folder=all_folder,
                                                                                                                                        arg_exp_id=args_cf_diabetic.arg_exp_id,dist_func=args_cf_diabetic.arg_dist_func,
                                                                                                                                        arg_learning_rate=args_cf_diabetic.arg_learning_rate,
                                                                                                                                        arg_gradient_steps=args_cf_diabetic.arg_gradient_steps)
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
    print('DEL EXIST Folder.')
mkdir(save_folder)

trace_file_folder = save_folder

# save parameters of this exp
argsDict = args_cf_diabetic.__dict__
with open('{save_folder}/parameters.txt'.format(save_folder=save_folder),'w') as f:
    f.writelines('-----------------------start---------------------------'+'\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg+':'+str(value)+'\n')
    f.writelines('-----------------------end---------------------------')

print('Begin running outside controller.')



train_index_all_patient_df_list = []
test_index_all_patient_df_list = []
distribution_all_patient_df_list = []
trace_df_all_patient_dict = {}

def generate_original_traces_for_each_patient(trace_file_folder, patient_type, patient_id, reward_fun_id, ENV_NAME,
                                              all_env_dict, trained_controller_dict, start_state_interval_dict):
    #start_state_interval = start_state_interval_dict[args_cf_diabetic.arg_start_state_id]
    trace_file_path = '{folder}/patient_trace_{patient_type}#{patient_id}_rf_{rf}_step.csv'.format(
        folder=trace_file_folder,
        patient_type=patient_type, patient_id=patient_id,
        rf=reward_fun_id)
    test_env = all_env_dict[ENV_NAME]
    test_model = trained_controller_dict[ENV_NAME]
    print('test_env: ', test_env, ' test_model: ', test_model)
    # Iterate over the number of epochs for generating the original trace
    epochs = 1
    # store the original trace in df
    trace_df = pd.DataFrame(columns=['patient_type', 'patient_id','step', 'episode', 'episode_step', 'time', 'action', 'state',
                                     'observation_CGM', 'observation_new_CGM',
                                     'observation_ROC', 'observation_new_ROC',
                                     'observation_CHO', 'observation_new_CHO',
                                     'observation_insulin', 'observation_new_insulin',
                                     'reward', 'accumulated_reward', 'done',
                                     'sample_time', 'patient_name', 'meal',
                                     'patient_state', 'last_foodtaken','is_eating',
                                     'cgm_sensor_params', 'cgm_sensor_seed', 'cgm_sensor_last_cgm',
                                     'scenario_seed', 'scenario_random_gen', 'scenario',
                                     'year', 'month', 'day',
                                     'hour', 'minute', 'bg', 'lbgi', 'hbgi', 'risk',
                                     'final_s_outcome', 'final_r_outcome',
                                     'final_s_cgm_outcome', 'final_s_cho_outcome', 'final_s_roc_outcome',
                                     'final_s_insulin_outcome'],
                            dtype=object)

    num_episodes = 0
    exit_flag = False

    # generat all original traces with outside controller, e.g. PPO, in test_env
    with torch.no_grad():
        for epoch in range(epochs):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            # num_episodes = 0
            accumulated_reward = 0
            episode_length = 0
            observation = test_env.reset()
            #print('obs after reset: ', observation)
            # Iterate over the steps of each epoch
            for t in range(test_step):
                # print('current time step in test env: ', t)
                if render:
                    test_env.render()
                # Get the logits, action, and take one step in the environment
                action, _states = test_model.predict(observation)
                observation_new, _, reward, done, info = test_env.step(action)
                # result = test_env.step(action)
                # print(result)
                #observation_new, reward, done, info = test_env.step(action)
                accumulated_reward += reward
                episode_length += 1
                trace_df = get_patient_trace(test_env, trace_df, t, num_episodes, episode_length, action, _states,
                                             observation,
                                             observation_new, reward, accumulated_reward,
                                             done, info, 0, 0, patient_type, patient_id)
                # Update the observation
                observation = observation_new
                # Finish trajectory if reached to a terminal state
                terminal = done
                if (t == test_step - 1):
                    # print(trace_df)
                    print('Trace is long enough.')
                    trace_df = trace_df.astype('object')

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

                    sum_return += accumulated_reward
                    sum_length += episode_length
                    observation, episode_return, episode_length = test_env.reset(), 0, 0
                    trace_df = add_action_effect_to_trace(trace_df)
                    trace_df.to_csv(trace_file_path)
                    exit_flag = True
                    break
                else:
                    if terminal:
                        num_episodes += 1
                        sum_return += accumulated_reward
                        sum_length += episode_length
                        observation, episode_return, episode_length = test_env.reset(), 0, 0
                        #print('obs after reset: ', observation)
                    else:
                        # print('Trace not long enough.')
                        continue

                if exit_flag:
                    print('Finish generating original traces.')
                    break

    check_len = args_cf_diabetic.arg_cflen  # 20 # length of CF traces
    interval = args_cf_diabetic.arg_cflen  # length of sliding window among original trace segments
    start_time_index = args_cf_diabetic.arg_cflen + 10  # the earlist time step in an original trace for training/testing
    time_index_checkpoint_list = [start_time_index + i * interval for i in range(500)]  # all possible time index for finding the original traces used in training/testing
    trace_df_per_eps_list = []  # store traces for each episode in trace_df
    eps_time_index_checkpoint_list = []  # store the {eps_idx: time_index} pairs for loading the parameters of original traces
    all_first_state_list = []
    max_total_reward = args_cf_diabetic.arg_total_r_thre
    for eps in range(num_episodes + 1):
        past_trace_this_eps = trace_df[trace_df["episode"] == eps]
        #print('len(past_trace_this_eps): ', len(past_trace_this_eps), ' len(trace_df): ', len(trace_df))
        for t in time_index_checkpoint_list:
            #print('t: ', t, ' int(len(past_trace_this_eps) * 0.75): ', int(len(past_trace_this_eps) * 0.75))
            if (t <= int(len(past_trace_this_eps) * 0.95)):
                this_segment = past_trace_this_eps[t - check_len : t]
                #print('this_segment index, Generate Orig: ', this_segment['episode'].tolist()[0], t, t - check_len,len(this_segment))
                start_state_this_seg = this_segment['observation_CGM'].tolist()[0]
                all_meal_this_seg = this_segment['meal'].sum()
                all_action_this_segment = this_segment['action'].tolist()
                if_same_action = len(set(all_action_this_segment))
                if all_meal_this_seg>0:
                    if_meal_this_seg = 1
                else:
                    if_meal_this_seg = 0
                total_reward_this_segment = this_segment['reward'].sum()
                # print('t: ', t, ' total_reward_this_segment: ', total_reward_this_segment,
                #       ' start_state_this_seg: ', start_state_this_seg, ' if_meal_this_seg: ', if_meal_this_seg)
                if (total_reward_this_segment < max_total_reward):
                    if (start_state_this_seg > args_cf_diabetic.arg_min_cgm) and (if_same_action!=1):
                        eps_time_index_checkpoint_list.append(
                            {'ENV_NAME': ENV_NAME, 'patient_type': patient_type, 'patient_id': patient_id,
                             'orig_episode': eps, 'orig_end_time_index': t,
                             'start_state_this_seg': start_state_this_seg})
                        all_first_state_list.append(start_state_this_seg)
        trace_df_per_eps_list.append(past_trace_this_eps)
    # train DDPG with original traces selected from eps_time_index_checkpoint_list，after check on the distribution of the first state
    train_index_df_this_patient, test_index_df_this_patient,distribution_df = sample_trace_segments_by_distribution(all_first_state_list,
                                        eps_time_index_checkpoint_list, patient_type, patient_id,
                                        total_use_number=args_cf_diabetic.arg_total_used_num)

    return train_index_df_this_patient, test_index_df_this_patient, trace_df,distribution_df

def sample_trace_segments_by_distribution(all_first_state_list, eps_time_index_checkpoint_list, patient_type, patient_id,
                                          total_use_number):
    if args_cf_diabetic.arg_if_start_state_restrict == 0:
        bins = [65, 100, 150, 260]
    else:
        if args_cf_diabetic.arg_start_state_id==0:
            bins = [65, 100]
            #bins = [70, 110]
        elif args_cf_diabetic.arg_start_state_id==1:
            bins = [65, 120]
            #bins = [100, 260]
            #bins = [110, 160]
        else:
            bins = [150, 260]
            #bins = [160, 260]

    bin_num = len(bins)-1
    dict_for_save = dict([(k, []) for k in range(0, bin_num)])
    train_sample_num = round(total_use_number / bin_num * args_cf_diabetic.arg_train_split)
    test_sample_num = round(total_use_number / bin_num * (1-args_cf_diabetic.arg_train_split))
    print('train_sample_num: ', train_sample_num, ' test_sample_num: ', test_sample_num)
    train_index_list = []
    test_index_list = []

    distribution_df = pd.DataFrame(columns=['patient_type','patient_id','bin'])
    distribution_df['patient_type'] = [patient_type]*len(bins)
    distribution_df['patient_id'] = [patient_id] * len(bins)
    #distribution_df['hist'] = hist.tolist()
    distribution_df['bin'] = bins
    for seg in eps_time_index_checkpoint_list:
        first_state_value = seg['start_state_this_seg']
        for i in range(0, len(bins)-1):
            lower_bound = bins[i]
            upper_bound = bins[i+1]
            #print(i, lower_bound, upper_bound)
            if lower_bound<=first_state_value<upper_bound:
                dict_for_save[i].append(seg)
    #print('dict_for_save after: ', dict_for_save)
    for k in dict_for_save.keys():
        trace_num = len(dict_for_save[k])
        print('key: ', k, ' num: ', trace_num)
    for key in dict_for_save.keys():
        value_list = dict_for_save[key]
        sample_for_train = sample(value_list, train_sample_num)
        #rest_list = [info_dict for info_dict in value_list if info_dict not in sample_for_train]
        rest_list = value_list
        sample_for_test = sample(rest_list, test_sample_num)
        train_index_list.extend(sample_for_train)
        test_index_list.extend(sample_for_test)
    train_index_df_this_patient = pd.DataFrame(columns=['ENV_NAME', 'patient_type', 'patient_id', 'orig_episode', 'orig_end_time_index','start_state_this_seg'])
    test_index_df_this_patient = pd.DataFrame(columns=['ENV_NAME', 'patient_type', 'patient_id', 'orig_episode', 'orig_end_time_index','start_state_this_seg'])
    for ix in train_index_list:
        train_index_df_this_patient = train_index_df_this_patient.append(ix, ignore_index=True)
    for ix in test_index_list:
        test_index_df_this_patient = test_index_df_this_patient.append(ix, ignore_index=True)
    train_start_state_list = train_index_df_this_patient['start_state_this_seg'].to_list()
    test_start_state_list = test_index_df_this_patient['start_state_this_seg'].to_list()
    train_distribution = pd.cut(train_start_state_list, bins)
    test_distribution = pd.cut(test_start_state_list, bins)
    print('train seg distribution: ', train_distribution.value_counts())
    print('test seg distribution: ', test_distribution.value_counts())
    return train_index_df_this_patient, test_index_df_this_patient, distribution_df

mem_before_generate_orig_trace = get_mem_use()
start_state_interval_dict = {0:[90, 140], 1:[70, 90], 2:[140,180], 3:[180, 300], 4:[30, 70]}
if args_cf_diabetic.arg_if_single_env == 0:# train and test with different envs
    if args_cf_diabetic.arg_use_exist_trace == 0:
        for id in patient_id_list:
            time_1 = time.perf_counter()
            ENV_NAME = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type, patient_id=id)
            train_index_df_this_patient, test_index_df_this_patient, trace_df, distribution_df_this_patient = generate_original_traces_for_each_patient(trace_file_folder, patient_type,
                                                                                          id,
                                                                                          reward_fun_id, ENV_NAME,
                                                                                          all_env_dict,
                                                                                          trained_controller_dict, start_state_interval_dict)
            train_index_all_patient_df_list.append(train_index_df_this_patient)
            test_index_all_patient_df_list.append(test_index_df_this_patient)
            distribution_all_patient_df_list.append(distribution_df_this_patient)
            trace_df_all_patient_dict[ENV_NAME] = trace_df
            time_2 = time.perf_counter()
        if args_cf_diabetic.arg_exp_type == 2:
            train_index_df = pd.concat(train_index_all_patient_df_list[3:])  # use 789 to train
            train_index_df = train_index_df.reset_index(drop=True)
            test_index_df = pd.concat(test_index_all_patient_df_list[3:])  # use the same 3 env for train to test, different traces
            test_index_df = test_index_df.reset_index(drop=True)
            distribution_df = pd.concat(distribution_all_patient_df_list[3:])  # use 789 to test
        else:
            train_index_df = pd.concat(train_index_all_patient_df_list[3:])  # use 789 to train
            train_index_df = train_index_df.reset_index(drop=True)
            test_index_df = pd.concat(test_index_all_patient_df_list[:3])  # use 456 to test, different traces
            test_index_df = test_index_df.reset_index(drop=True)
            distribution_df = pd.concat(distribution_all_patient_df_list[:3])  # use 456 to test
    else:
        exist_trace_folder = '{diabetic_file_folder}/exist_trace/exp_{arg_exp_type}/ppo_{arg_ppo_train_time}_generalize_group_{arg_exist_trace_id}'.format(
            diabetic_file_folder=diabetic_file_folder,
            arg_exist_trace_id=args_cf_diabetic.arg_exist_trace_id,
            arg_exp_type=args_cf_diabetic.arg_exp_type,
            arg_ppo_train_time=args_cf_diabetic.arg_ppo_train_time)
        exist_train_index_file_path = '{exist_trace_folder}/train_index_file.csv'.format(exist_trace_folder=exist_trace_folder)
        exist_test_index_file_path = '{exist_trace_folder}/test_index_file.csv'.format(exist_trace_folder=exist_trace_folder)
        exist_distribution_file_path = '{exist_trace_folder}/first_state_distribution_interval_file.csv'.format(
            exist_trace_folder=exist_trace_folder)
        for id in patient_id_list:
            ENV_NAME = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type, patient_id=id)
            exist_trace_file_path = '{exist_trace_folder}/patient_trace_{patient_type}#{patient_id}_rf_{reward_fun_id}_step.csv'.format(exist_trace_folder=exist_trace_folder,
                                                                                                                   patient_type=patient_type, patient_id=id,
                                                                                                                    reward_fun_id=reward_fun_id)
            trace_df = pd.read_csv(exist_trace_file_path)
            trace_df = set_fixed_orig_trace(trace_df, save_path=save_folder, orig_id=args_cf_diabetic.arg_exist_trace_id,
                                            patient_type=patient_type, patient_id=id)
            #trace_df_all_env_dict[g] = trace_df
            trace_df_all_patient_dict[ENV_NAME] = trace_df
        train_index_df = pd.read_csv(exist_train_index_file_path)
        test_index_df = pd.read_csv(exist_test_index_file_path)
        distribution_df = pd.read_csv(exist_distribution_file_path)
    # train_index_df = pd.concat(train_index_all_patient_df_list[3:])# 7, 8, 9 to train
    # test_index_df = pd.concat(test_index_all_patient_df_list[3:])#use 7, 8, 9 to test
    # train_index_df = train_index_df.reset_index(drop=True)
    # test_index_df = test_index_df.reset_index(drop=True)
    # distribution_df = pd.concat(distribution_all_patient_df_list[3:])  # use 7, 8, 9 to test
else:# train and test with 1 env
    if args_cf_diabetic.arg_use_exist_trace == 0:
        id = args_cf_diabetic.arg_assigned_patient_id
        ENV_NAME = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type, patient_id=id)
        train_index_df_this_patient, test_index_df_this_patient, trace_df, distribution_df_this_patient = generate_original_traces_for_each_patient(trace_file_folder, patient_type,
                                                                                          id,
                                                                                          reward_fun_id, ENV_NAME,
                                                                                          all_env_dict,
                                                                                          trained_controller_dict, start_state_interval_dict)
        train_index_all_patient_df_list.append(train_index_df_this_patient)
        test_index_all_patient_df_list.append(test_index_df_this_patient)
        distribution_all_patient_df_list.append(distribution_df_this_patient)
        trace_df_all_patient_dict[ENV_NAME] = trace_df
        train_index_df = pd.concat(train_index_all_patient_df_list)
        test_index_df = pd.concat(test_index_all_patient_df_list)
        distribution_df = distribution_all_patient_df_list[0]
    else:
        exist_trace_folder = '{diabetic_file_folder}/exist_trace/exp_{arg_exp_type}/ppo_{arg_ppo_train_time}_{patient_type}_p_{patient_id}_group_{arg_exist_trace_id}'.format(
            diabetic_file_folder=diabetic_file_folder,
            arg_exist_trace_id=args_cf_diabetic.arg_exist_trace_id,
            arg_exp_type=args_cf_diabetic.arg_exp_type,
            arg_ppo_train_time=args_cf_diabetic.arg_ppo_train_time,
            patient_type=patient_type, patient_id=args_cf_diabetic.arg_assigned_patient_id)
        exist_trace_file_path = '{exist_trace_folder}/patient_trace_{patient_type}#{patient_id}_rf_{reward_fun_id}_step.csv'.format(
            exist_trace_folder=exist_trace_folder,patient_type=patient_type,
            patient_id=args_cf_diabetic.arg_assigned_patient_id, reward_fun_id=reward_fun_id)
        exist_train_index_file_path = '{exist_trace_folder}/train_index_file.csv'.format(
            exist_trace_folder=exist_trace_folder)
        exist_test_index_file_path = '{exist_trace_folder}/test_index_file.csv'.format(
            exist_trace_folder=exist_trace_folder)
        exist_distribution_file_path = '{exist_trace_folder}/first_state_distribution_interval_file.csv'.format(
            exist_trace_folder=exist_trace_folder)
        trace_df = pd.read_csv(exist_trace_file_path)
        trace_df = set_fixed_orig_trace(trace_df, save_path=save_folder, orig_id=args_cf_diabetic.arg_exist_trace_id,
                                            patient_type=patient_type, patient_id=args_cf_diabetic.arg_assigned_patient_id)
        #trace_df = get_patient_trace_from_saved_data(trace_df, save_path=trace_file_folder,orig_id=args_cf_diabetic.arg_exist_trace_id,g=args_cf_diabetic.arg_assigned_gravity)
        #trace_df_all_env_dict[args_cf_diabetic.arg_assigned_gravity] = trace_df
        ENV_NAME = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type, patient_id=args_cf_diabetic.arg_assigned_patient_id)
        trace_df_all_patient_dict[ENV_NAME] = trace_df
        train_index_df = pd.read_csv(exist_train_index_file_path)
        test_index_df = pd.read_csv(exist_test_index_file_path)
        distribution_df = pd.read_csv(exist_distribution_file_path)
gc.collect()
mem_after_generate_orig_trace = get_mem_use()
print('Mem usage, Generate Orig Trace: ', mem_after_generate_orig_trace-mem_before_generate_orig_trace)
train_index_path = '{save_folder}/train_index_file.csv'.format(save_folder=save_folder)
test_index_path = '{save_folder}/test_index_file.csv'.format(save_folder=save_folder)
distribution_path = '{save_folder}/first_state_distribution_interval_file.csv'.format(save_folder=save_folder)
train_index_df.to_csv(train_index_path)
test_index_df.to_csv(test_index_path)
distribution_df.to_csv(distribution_path)
print('train_index_df: ', len(train_index_df))
print('test_index_df: ', len(test_index_df))

train_counter = 0
test_counter = 0
test_eps_each_trace = args_cf_diabetic.arg_total_test_trace_num
trained_timestep_list = []
test_timestep_list = []
patient_info_input_dim = 13 # len of input patient state vector
patient_info_output_dim = args_cf_diabetic.arg_patidim # len of compressed patient state vector
# train_result_path = '{save_folder}/traineps_{train_eps_each_trace}_testeps_{test_eps_each_trace}_updateiter_{update_iteration}_patientdim_{patient_info_output_dim}'.format(save_folder=save_folder,
#                                         train_eps_each_trace=train_eps_each_trace,test_eps_each_trace=test_eps_each_trace,update_iteration=update_iteration,
#                                         patient_info_output_dim=patient_info_output_dim)
# mkdir(train_result_path)
ddpg_save_directory = '{save_folder}/trained_model'.format(save_folder=save_folder)
mkdir(ddpg_save_directory)
# store_time_index(train_index, train_index_path)
# store_time_index(test_index, test_index_path)
# total_train_round = args_cf_diabetic.arg_train_round # total train round for DDPG
# train_mark = args_cf_diabetic.arg_train_mark


index_already_test_list = [] # store the {eps:time_index} pairs that have beed tested


def train(model_CF, patient_BW, delta_dist, train_index, total_train_round,train_mark, test_index, each_test_round, trace_df_per_eps_list,
          index_already_test_list, improve_percentage_thre, check_len, ENV_NAME, fixed_trace_this_eps_df, baseline_path=ppo_model_path):
    # train with a single original traces
    CF_trace_train_list = []
    loss_list = []
    gradient_df_list = []
    lambda_df_list = []
    lowest_distance_df_list = []
    train_counter = 0
    train_counter_max = len(train_index)#*total_train_round
    print('train_counter_max: ', train_counter_max*total_train_round)
    orig_trace_df_list = []
    accumulated_reward_train_set_df = pd.DataFrame(
        columns=['id', 'orig_trace_episode', 'orig_end_step', 'orig_accumulated_reward', 'cf_accumulated_reward',
                 'difference', 'percentage', 'effect_distance', 'epsilon','cf_count_distance','cf_pairwise_distance',
                 'orig_effect','cf_effect', 'cf_iob_distance','orig_iob_distance','orig_start_action_effect','if_generate_cf'])
    train_static_df = pd.DataFrame(columns=['id', 'better_accumulated_reward_percentage', 'better_improve_percentage','better_max_accumulated_reward_percentage'])
    info_dict = train_index[0]
    for k, v in info_dict.items():
        trace_eps, current_time_step = k, v

    for t_round in range(total_train_round):
        random.shuffle(train_index)
        for idx in range(0, train_counter_max): # fill buffer with cf samples, evey fill with train_mark number of orig traces, train once
            info_dict_idx = idx #idx%len(train_index)
            past_trace_this_eps = fixed_trace_this_eps_df #trace_df_per_eps_list[trace_eps]  # get the corresponding original trace df for this trace_eps
            CF_start_step = current_time_step - check_len + 1
            orig_trace_for_CF = past_trace_this_eps[CF_start_step:current_time_step + 1]
            results = cf_generator.reset_env_fill_buffer(past_trace_this_eps, current_time_step, check_len, ENV_NAME, model_CF, trace_eps, iob_param=args_cf_diabetic.arg_iob_param, sample_num=args_cf_diabetic.arg_train_sample_num)

            if results is not None:
                # this original trace can generate better CFs, save results
                train_counter += 1
                #print('train_counter: ', train_counter)
                trained_timestep_list.append(info_dict)
                (accumulated_reward_list, effect_distance_list, epsilon_list,
                 orig_effect_list, cf_effect_list, cf_IOB_distance_list, orig_IOB_distance_list, orig_start_action_effect_list,
                 cf_distance_count_list, cf_pairwise_distance_list, CF_trace) = results
                CF_trace_train_list.append(CF_trace)
                orig_trace_df_list.append(orig_trace_for_CF)

                # store the accumulated reward for original and cf trace for this one
                if len(accumulated_reward_list) == 0:
                    train_static_df = train_static_df.append(
                        {'id': train_counter, 'better_accumulated_reward_percentage': -9999,
                         'better_improve_percentage': -9999,'better_max_accumulated_reward_percentage':-9999}, ignore_index=True)
                else:
                    for item in accumulated_reward_list:
                        orig_accu_r = item[0]
                        cf_accu_r = item[1]
                        effect_distance = effect_distance_list[accumulated_reward_list.index(item)]
                        epsilon = epsilon_list[accumulated_reward_list.index(item)]
                        perc = (cf_accu_r - orig_accu_r) / abs(orig_accu_r) if orig_accu_r != 0 else -9999
                        orig_effect = orig_effect_list[accumulated_reward_list.index(item)]
                        cf_effect = cf_effect_list[accumulated_reward_list.index(item)]
                        cf_iob_distance = cf_IOB_distance_list[accumulated_reward_list.index(item)]
                        orig_iob_distance = orig_IOB_distance_list[accumulated_reward_list.index(item)]
                        orig_start_action_effect = orig_start_action_effect_list[accumulated_reward_list.index(item)]
                        cf_count_distance = cf_distance_count_list[accumulated_reward_list.index(item)]
                        cf_pairwise_distance = cf_pairwise_distance_list[accumulated_reward_list.index(item)]
                        accumulated_reward_dict = {'id': train_counter, 'orig_trace_episode': trace_eps,
                                                   'orig_end_step': current_time_step,
                                                   'orig_accumulated_reward': orig_accu_r,
                                                   'cf_accumulated_reward': cf_accu_r,
                                                   'difference': cf_accu_r - orig_accu_r, 'percentage': perc,
                                                   'effect_distance': effect_distance, 'epsilon':epsilon,
                                                   'orig_effect': orig_effect,
                                                   'cf_effect': cf_effect,
                                                   'cf_iob_distance': cf_iob_distance,
                                                   'orig_iob_distance': orig_iob_distance,
                                                   'orig_start_action_effect': orig_start_action_effect,
                                                   'cf_count_distance':cf_count_distance,
                                                   'cf_pairwise_distance':cf_pairwise_distance, 'if_generate_cf':1}
                        accumulated_reward_train_set_df = accumulated_reward_train_set_df.append(accumulated_reward_dict,
                                                                                                 ignore_index=True)
                    (better_difference_perc, max_difference_perc,improve_percentage) = static_accumulate_reward(accumulated_reward_train_set_df,
                                                                                            improve_percentage_thre)

                    train_static_df = train_static_df.append(
                        {'id': train_counter, 'better_accumulated_reward_percentage': better_difference_perc,
                         'better_improve_percentage': improve_percentage, 'better_max_accumulated_reward_percentage':max_difference_perc}, ignore_index=True)
                    # if train_counter % 30 == 0:
                    #     print('Train_counter: ', train_counter,
                    #           ' {a}% gain better accumulated reward, {b}% gain better max accumulated reward.'.format(
                    #               a=better_difference_perc * 100, b=max_difference_perc * 100))
            else:
                accumulated_reward_dict = {'id': train_counter, 'orig_trace_episode': trace_eps,
                                           'orig_end_step': current_time_step,
                                           'orig_accumulated_reward': -10000,
                                           'cf_accumulated_reward': -10000,
                                           'difference': -10000, 'percentage': -10000,
                                           'effect_distance': -10000, 'epsilon': -10000,
                                           'orig_effect': -10000,
                                           'cf_effect': -10000,
                                           'cf_iob_distance': -10000,
                                           'orig_iob_distance': -10000,
                                           'orig_start_action_effect': -10000,
                                           'cf_count_distance': -10000,
                                           'cf_pairwise_distance': -10000, 'if_generate_cf':0}
                accumulated_reward_train_set_df = accumulated_reward_train_set_df.append(accumulated_reward_dict,
                                                                                         ignore_index=True)


            if (train_counter%train_mark==0) and train_counter>0:
                #print('Update model.')
                (aveg_actor_loss_list, aveg_critic_loss_list, aveg_lambda_loss_list, aveg_encoder_loss_list,
                 all_actor_gradient_norm_list, all_critic_gradient_norm_list,
                 all_target_actor_gradient_norm_list, all_target_critic_gradient_norm_list, all_lambda_value_list) = cf_generator.train_ddpg_cf(model_CF, train_eps_each_trace)
                loss_df = pd.DataFrame(
                    columns=['id', 'orig_trace_episode', 'orig_end_step', 'aveg_actor_loss', 'aveg_critic_loss',
                             'aveg_lambda_loss',
                             'aveg_encoder_loss'])
                loss_df['aveg_actor_loss'] = aveg_actor_loss_list
                loss_df['aveg_critic_loss'] = aveg_critic_loss_list
                loss_df['aveg_lambda_loss'] = aveg_lambda_loss_list
                loss_df['aveg_encoder_loss'] = aveg_encoder_loss_list
                loss_df['orig_end_step'] = current_time_step
                loss_df['orig_trace_episode'] = trace_eps
                loss_df['id'] = train_counter
                loss_list.append(loss_df)

                gradient_df = pd.DataFrame(
                    columns=['actor_gradient', 'target_actor_gradient', 'critic_gradient', 'target_critic_gradient'])
                gradient_df['actor_gradient'] = all_actor_gradient_norm_list
                gradient_df['target_actor_gradient'] = all_target_actor_gradient_norm_list
                gradient_df['critic_gradient'] = all_critic_gradient_norm_list
                gradient_df['target_critic_gradient'] = all_target_critic_gradient_norm_list
                gradient_df['orig_end_step'] = current_time_step
                gradient_df['orig_trace_episode'] = trace_eps
                gradient_df['id'] = train_counter
                gradient_df_list.append(gradient_df)

                lambda_df = pd.DataFrame(columns=['lambda_value'])
                lambda_df['lambda_value'] = all_lambda_value_list
                lambda_df['orig_end_step'] = current_time_step
                lambda_df['orig_trace_episode'] = trace_eps
                lambda_df['id'] = train_counter
                lambda_df_list.append(lambda_df)

                cf_with_better_outcome_df = accumulated_reward_train_set_df[accumulated_reward_train_set_df['difference']>0]
                if len(cf_with_better_outcome_df)==0:
                    #print('No better CF yet.')
                    #best_outcome = round(max(accumulated_reward_train_set_df['difference'].tolist()), 2)
                    lowest_distance = -10
                    # print('At {train_counter}, among all cf traces ({trace_eps}, {current_time_step}), the lowest value of a better cf trace is: {lowest_distance}'.format(
                    #         lowest_distance=lowest_distance, trace_eps=trace_eps, current_time_step=current_time_step,
                    #         train_counter=train_counter))
                else:
                    distance_for_better_cf_df = cf_with_better_outcome_df['cf_pairwise_distance']
                    lowest_distance = min(distance_for_better_cf_df.tolist())
                    best_outcome = round(cf_with_better_outcome_df[cf_with_better_outcome_df['cf_pairwise_distance']==lowest_distance]['difference'].tolist()[0], 2)
                    # print('At {train_counter}, among all cf traces ({trace_eps}, {current_time_step}), the lowest value of a better cf trace is: {lowest_distance}, outcome: {best_outcome}'.format(
                    #         lowest_distance=lowest_distance, trace_eps=trace_eps, current_time_step=current_time_step, train_counter=train_counter,best_outcome=best_outcome))
                lowest_distance_df = pd.DataFrame(
                    columns=['id', 'orig_trace_episode', 'orig_end_step', 'lowest_distance'])
                lowest_distance_df['lowest_distance'] = lowest_distance
                lowest_distance_df['orig_end_step'] = current_time_step
                lowest_distance_df['orig_trace_episode'] = trace_eps
                lowest_distance_df['id'] = train_counter
                lowest_distance_df_list.append(lowest_distance_df)


            if train_counter % args_cf_diabetic.arg_test_mark == 0:
                model_CF.save(num_episodes=train_counter, directory=ddpg_save_directory)
                #print('train_counter: ', train_counter)
                print('Run baseline PPO on test set.')
                index_already_test_list_ppo, CF_test_trace_total_ppo, accumulated_reward_test_set_df_ppo = test(None, patient_BW, delta_dist, test_index, each_test_round, trace_df_per_eps_list,
                                                   index_already_test_list,
                                                   improve_percentage_thre, train_counter, fixed_trace_this_eps_df, mode='test_baseline',
                                                   baseline_path=baseline_path)
                # run ddpg on test set
                print('Run trained DDPG on test set.')
                index_already_test_list_ddpg, CF_test_trace_total_ddpg, accumulated_reward_test_set_df_ddpg = test(model_CF, patient_BW, delta_dist, test_index, each_test_round,
                                                    trace_df_per_eps_list,
                                                    index_already_test_list,
                                                    improve_percentage_thre, train_counter, fixed_trace_this_eps_df, mode='test_cf',
                                                    baseline_path=None)

                # save training performance
                cf_trace_train_file_path = '{save_folder}/cf_trace_file_train_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder,
                                                                                                                          trace_eps=trace_eps, current_time_step=current_time_step)
                cf_loss_train_file_path = '{save_folder}/cf_train_loss_train_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder,
                                                                                                                          trace_eps=trace_eps, current_time_step=current_time_step)
                CF_train_trace_total = pd.concat(CF_trace_train_list)
                loss_train_total = pd.concat(loss_list)
                #CF_train_trace_total.to_csv(cf_trace_train_file_path)
                loss_train_total.to_csv(cf_loss_train_file_path)
                actual_train_index_path = '{save_folder}/actual_train_index_file.csv'.format(save_folder=save_folder)
                store_time_index(trained_timestep_list, actual_train_index_path)
                #total_orig_trace_train = pd.concat(orig_trace_df_list)
                #total_orig_trace_train.to_csv('{save_folder}/orig_trace_train_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder,
                #                                                                                                         trace_eps=trace_eps, current_time_step=current_time_step))
                accumulated_reward_train_set_df.to_csv(
                    '{save_folder}/accumulated_reward_train_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder,
                                                                                                                          trace_eps=trace_eps, current_time_step=current_time_step))
                train_static_df.to_csv('{save_folder}/train_statistic_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder,
                                                                                                                          trace_eps=trace_eps, current_time_step=current_time_step))
                gradient_df_total = pd.concat(gradient_df_list)
                gradient_df_total.to_csv('{save_folder}/gradient_training_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder,
                                                                                                                          trace_eps=trace_eps, current_time_step=current_time_step))

                lambda_df_total = pd.concat(lambda_df_list)
                lambda_df_total.to_csv('{save_folder}/lambda_value_training_{trace_eps}_{current_time_step}.csv'.format(
                    save_folder=save_folder,
                    trace_eps=trace_eps, current_time_step=current_time_step))
                # lowest_distance_df_total = pd.concat(lowest_distance_df_list)
                # lowest_distance_df_total.to_csv('{save_folder}/lowest_distance_training_{trace_eps}_{current_time_step}.csv'.format(
                #     save_folder=save_folder, trace_eps=trace_eps, current_time_step=current_time_step))


    print('Train finished.')
    #print('Final train_counter: ', train_counter)

    total_train_episode = train_counter  # int(total_train_round*train_eps_each_trace)
    model_CF.save(num_episodes=train_counter, directory=ddpg_save_directory)
    # run baseline ppo on test set
    print('Run final baseline PPO on test set.')
    index_already_test_list_ppo, CF_test_trace_total_ppo, accumulated_reward_test_set_df_ppo = test(None, patient_BW, delta_dist, test_index, each_test_round, trace_df_per_eps_list,
                                        index_already_test_list, improve_percentage_thre, train_counter, fixed_trace_this_eps_df, mode='test_baseline', baseline_path=baseline_path)
    # run ddpg on test set
    print('Run final trained DDPG on test set.')
    index_already_test_list_ddpg, CF_test_trace_total_ddpg, accumulated_reward_test_set_df_ddpg = test(model_CF, patient_BW, delta_dist, test_index, each_test_round, trace_df_per_eps_list,
                                        index_already_test_list,improve_percentage_thre, train_counter, fixed_trace_this_eps_df, mode='test_cf', baseline_path=None)
    # save training performance
    cf_trace_train_file_path = '{save_folder}/cf_trace_file_train_trained_{train_counter}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder, train_counter=train_counter,
                                                                                                                          trace_eps=trace_eps, current_time_step=current_time_step)
    cf_loss_train_file_path = '{save_folder}/cf_train_loss_trained_{train_counter}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder,
                                                                                               train_counter=train_counter,trace_eps=trace_eps, current_time_step=current_time_step)
    CF_train_trace_total = pd.concat(CF_trace_train_list)
    loss_train_total = pd.concat(loss_list)
    CF_train_trace_total.to_csv(cf_trace_train_file_path)
    loss_train_total.to_csv(cf_loss_train_file_path)
    actual_train_index_path = '{save_folder}/actual_train_index_file_trained_{train_counter}_{trace_eps}_{current_time_step}.csv'.format(
        save_folder=save_folder, train_counter=train_counter,trace_eps=trace_eps, current_time_step=current_time_step)
    store_time_index(trained_timestep_list, actual_train_index_path)
    total_orig_trace_train = pd.concat(orig_trace_df_list)
    total_orig_trace_train.to_csv(
        '{save_folder}/orig_trace_train_trained_{train_counter}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder,
                                                                            train_counter=train_counter,trace_eps=trace_eps, current_time_step=current_time_step))
    accumulated_reward_train_set_df.to_csv(
        '{save_folder}/accumulated_reward_train_trained_{train_counter}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder,
                                                                                    train_counter=train_counter,trace_eps=trace_eps, current_time_step=current_time_step))
    train_static_df.to_csv('{save_folder}/train_statistic_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder, trace_eps=trace_eps, current_time_step=current_time_step))
    gradient_df_total = pd.concat(gradient_df_list)
    gradient_df_total.to_csv('{save_folder}/gradient_training_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder, trace_eps=trace_eps, current_time_step=current_time_step))

    lambda_df_total = pd.concat(lambda_df_list)
    lambda_df_total.to_csv('{save_folder}/lambda_value_training_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder,
        trace_eps=trace_eps, current_time_step=current_time_step))

    lowest_distance_df_total = pd.concat(lowest_distance_df_list)
    # lowest_distance_df_total.to_csv('{save_folder}/lowest_distance_training_{trace_eps}_{current_time_step}.csv'.format(
    #     save_folder=save_folder,
    #     trace_eps=trace_eps, current_time_step=current_time_step))
    # print('lowest_distance_df_total length after training: ', len(lowest_distance_df_total))
    return loss_train_total, total_orig_trace_train, train_static_df, gradient_df_total, lowest_distance_df_total, CF_train_trace_total, \
           CF_test_trace_total_ppo, accumulated_reward_test_set_df_ppo, CF_test_trace_total_ddpg, accumulated_reward_test_set_df_ddpg

def test(rl_model, patient_BW, delta_dist, test_index, each_test_round, trace_df_per_eps_list, index_already_test_list,improve_percentage_thre,train_counter,
         fixed_trace_this_eps_df, mode='test_cf', baseline_path=None, result_path=None):
    save_folder_baseline = '{save_folder}/baseline_results'.format(save_folder=save_folder)
    mkdir(save_folder_baseline)
    # test with a set of original traces
    test_counter = 0
    #index_not_test = [info_dict for info_dict in test_index if info_dict not in index_already_test_list]
    index_not_test = [info_dict for info_dict in test_index]
    accumulated_reward_difference_total_list = []
    orig_trace_df_list = []
    CF_trace_test_list = []
    accumulated_reward_test_set_df = pd.DataFrame(columns=['id','orig_trace_episode', 'orig_end_step', 'orig_accumulated_reward', 'cf_accumulated_reward',
                                                'difference', 'percentage', 'cf_count_distance','cf_pairwise_distance',
                                                'effect_distance', 'orig_effect', 'cf_effect', 'cf_iob_distance','orig_iob_distance', 'orig_start_action_effect','cf_aveg_cgm','if_generate_cf'])
    info_dict = test_index[0]
    for k, v in info_dict.items():
        trace_eps, current_time_step = k, v

    for info_dict in index_not_test:
        if test_counter < each_test_round:
            past_trace_this_eps = fixed_trace_this_eps_df# trace_df_per_eps_list[trace_eps]
            index_already_test_list.append(info_dict)
            results = cf_generator.cf_generator_diabetic(past_trace_this_eps,current_time_step, check_len, ENV_NAME, rl_model, mode=mode,
                                                         run_eps_each_trace=test_eps_each_trace,orig_trace_episode=trace_eps,patient_BW=patient_BW,delta_dist=delta_dist,
                                                         iob_param=args_cf_diabetic.arg_iob_param,
                                                         train_episodes_mark_test=train_counter, baseline_path=baseline_path)

            if results is not None:
                # this original trace can generate better CFs
                # print('Test one epoch.')
                test_counter += 1
                #print('test_counter: ', test_counter)
                test_timestep_list.append(info_dict)
                (accumulated_reward_list, effect_distance_list, orig_effect_list, cf_effect_list, cf_aveg_cgm_list,
                 cf_IOB_distance_list, orig_IOB_distance_list,orig_start_action_effect_list, cf_distance_count_list,
                 cf_pairwise_distance_list, CF_trace, orig_trace_for_CF) = results
                CF_trace_test_list.append(CF_trace)
                orig_trace_df_list.append(orig_trace_for_CF)
                # store the accumulated reward for original and cf trace for this one
                if len(accumulated_reward_list)==0:
                    accumulated_reward_dict = {'id': test_counter, 'orig_trace_episode': trace_eps,
                                               'orig_end_step': current_time_step,
                                               'orig_accumulated_reward': -9999,
                                               'cf_accumulated_reward': -9999,
                                               'difference': -9999, 'percentage': -9999,
                                               'effect_distance': -9999, 'orig_effect':-9999, 'cf_effect':-9999,
                                               'cf_iob_distance': -9999,
                                               'orig_iob_distance': -9999,
                                               'orig_start_action_effect':-9999,
                                               'cf_aveg_cgm': -9999, 'cf_count_distance':-9999, 'cf_pairwise_distance':-9999, 'if_generate_cf':-1
                                               }
                    accumulated_reward_test_set_df = accumulated_reward_test_set_df.append(accumulated_reward_dict,
                                                                                           ignore_index=True)
                else:
                    for item in accumulated_reward_list:
                        orig_accu_r = item[0]
                        cf_accu_r = item[1]
                        perc = (cf_accu_r - orig_accu_r) / abs(orig_accu_r) if orig_accu_r != 0 else -9999
                        effect_distance = effect_distance_list[accumulated_reward_list.index(item)]
                        orig_effect = orig_effect_list[accumulated_reward_list.index(item)]
                        cf_effect = cf_effect_list[accumulated_reward_list.index(item)]
                        cf_aveg_cgm = cf_aveg_cgm_list[accumulated_reward_list.index(item)]
                        cf_iob_distance = cf_IOB_distance_list[accumulated_reward_list.index(item)]
                        orig_iob_distance = orig_IOB_distance_list[accumulated_reward_list.index(item)]
                        orig_start_action_effect = orig_start_action_effect_list[accumulated_reward_list.index(item)]
                        cf_count_distance = cf_distance_count_list[accumulated_reward_list.index(item)]
                        cf_pairwise_distance = cf_pairwise_distance_list[accumulated_reward_list.index(item)]
                        accumulated_reward_dict = {'id':test_counter,'orig_trace_episode': trace_eps, 'orig_end_step': current_time_step,
                                                   'orig_accumulated_reward': orig_accu_r,
                                                   'cf_accumulated_reward': cf_accu_r,
                                                   'difference': cf_accu_r - orig_accu_r, 'percentage':perc,
                                                   'effect_distance':effect_distance, 'orig_effect':orig_effect, 'cf_effect':cf_effect,
                                                   'cf_iob_distance':cf_iob_distance, 'orig_iob_distance':orig_iob_distance,
                                                   'orig_start_action_effect':orig_start_action_effect,
                                                   'cf_aveg_cgm':cf_aveg_cgm, 'cf_count_distance':cf_count_distance, 'cf_pairwise_distance':cf_pairwise_distance, 'if_generate_cf':1}
                        accumulated_reward_test_set_df = accumulated_reward_test_set_df.append(accumulated_reward_dict,ignore_index=True)

            else:
                # this original trace can not generate better CFs, change to another original trace
                accumulated_reward_dict = {'id': train_counter, 'orig_trace_episode': trace_eps,
                                           'orig_end_step': current_time_step,
                                           'orig_accumulated_reward': -10000,
                                           'cf_accumulated_reward': -10000,
                                           'difference': -10000, 'percentage': -10000,
                                           'effect_distance': -10000, 'epsilon': -10000,
                                           'orig_effect': -10000,
                                           'cf_effect': -10000,
                                           'cf_iob_distance': -10000,
                                           'orig_iob_distance': -10000,
                                           'orig_start_action_effect': -10000,
                                           'cf_count_distance': -10000,
                                           'cf_pairwise_distance': -10000, 'if_generate_cf':0}
                accumulated_reward_test_set_df = accumulated_reward_test_set_df.append(accumulated_reward_dict,
                                                                                         ignore_index=True)
                continue
        else:
            break
    cf_trace_test_file_path = '{save_folder}/cf_trace_file_{mode}_trained_{train_counter}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline,
                                                                                    mode=mode, train_counter=train_counter, trace_eps=trace_eps, current_time_step=current_time_step)
    # cf_accumulated_reward_test_file_path = '{train_result_path}/cf_test_accumulated_reward_difference_{mode}.csv'.format(train_result_path=train_result_path,mode=mode)
    if len(CF_trace_test_list)!=0:
        CF_test_trace_total = pd.concat(CF_trace_test_list)
        CF_test_trace_total.to_csv(cf_trace_test_file_path)
        actual_test_index_path = '{save_folder}/actual_test_index_file_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline, trace_eps=trace_eps, current_time_step=current_time_step)
        store_time_index(test_timestep_list, actual_test_index_path)
        total_orig_trace_test = pd.concat(orig_trace_df_list)
        total_orig_trace_test.to_csv('{save_folder}/orig_trace_test_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline, trace_eps=trace_eps, current_time_step=current_time_step))
    else:
        CF_test_trace_total = None
    accumulated_reward_test_set_df.to_csv('{save_folder}/accumulated_reward_{mode}_trained_{train_counter}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline,
                                                                                            mode=mode,train_counter=train_counter, trace_eps=trace_eps, current_time_step=current_time_step))
    return index_already_test_list, CF_test_trace_total, accumulated_reward_test_set_df

def test_baseline(rl_model, patient_BW, delta_dist, test_index, train_counter,
         fixed_trace_this_eps_df, mode='test_cf', baseline_path=None):
    save_folder_baseline = '{save_folder}/baseline_results'.format(save_folder=save_folder)
    mkdir(save_folder_baseline)
    # test with a set of original traces
    test_counter = 0
    #index_not_test = [info_dict for info_dict in test_index if info_dict not in index_already_test_list]
    index_not_test = [info_dict for info_dict in test_index]
    accumulated_reward_difference_total_list = []
    orig_trace_df_list = []
    CF_trace_test_list = []
    accumulated_reward_test_set_df = pd.DataFrame(columns=['id','patient_type','patient_id','orig_trace_episode', 'orig_end_step', 'orig_accumulated_reward', 'cf_accumulated_reward',
                                                'difference', 'percentage', 'cf_count_distance','cf_pairwise_distance',
                                                'effect_distance', 'orig_effect', 'cf_effect', 'cf_iob_distance','orig_iob_distance', 'orig_start_action_effect','cf_aveg_cgm','if_generate_cf'])
    #info_dict = test_index
    #for k, v in info_dict.items():
    env_name, trace_eps, current_time_step, patient_type, patient_id = test_index['ENV_NAME'], test_index['orig_episode'], test_index['orig_end_time_index'], \
                                            test_index['patient_type'], test_index['patient_id']
    ##
    print('Test Baseline PPO with traces end at: ', current_time_step, ' trace_eps: ', trace_eps, ' env_name: ', env_name, ' model: ', baseline_path)
    past_trace_this_eps = fixed_trace_this_eps_df  # trace_df_per_eps_list[trace_eps]
    #index_already_test_list.append(info_dict)
    results = cf_generator.cf_generator_diabetic(past_trace_this_eps, current_time_step, args_cf_diabetic.arg_cflen, ENV_NAME,
                                                     rl_model, mode=mode,
                                                     run_eps_each_trace=test_eps_each_trace,
                                                     orig_trace_episode=trace_eps, patient_BW=patient_BW,
                                                     delta_dist=delta_dist,
                                                     iob_param=args_cf_diabetic.arg_iob_param,
                                                     train_episodes_mark_test=train_counter,
                                                     baseline_path=baseline_path,
                                                 total_r_thre=args_cf_diabetic.arg_total_r_thre)

    if results is not None:
        # this original trace can generate better CFs
        # print('Test one epoch.')
        test_counter += 1
        # print('test_counter: ', test_counter)
        #test_timestep_list.append(info_dict)
        (accumulated_reward_list, effect_distance_list, orig_effect_list, cf_effect_list, cf_aveg_cgm_list,
             cf_IOB_distance_list, orig_IOB_distance_list, orig_start_action_effect_list, cf_distance_count_list,
             cf_pairwise_distance_list, CF_trace, orig_trace_for_CF) = results
        CF_trace_test_list.append(CF_trace)
        orig_trace_df_list.append(orig_trace_for_CF)
        # store the accumulated reward for original and cf trace for this one
        if len(accumulated_reward_list) == 0:
            accumulated_reward_dict = {'id': test_counter, 'patient_type':patient_type, 'patient_id':patient_id,
                                       'orig_trace_episode': trace_eps,
                                           'orig_end_step': current_time_step,
                                           'orig_accumulated_reward': -9999,
                                           'cf_accumulated_reward': -9999,
                                           'difference': -9999, 'percentage': -9999,
                                           'effect_distance': -9999, 'orig_effect': -9999, 'cf_effect': -9999,
                                           'cf_iob_distance': -9999,
                                           'orig_iob_distance': -9999,
                                           'orig_start_action_effect': -9999,
                                           'cf_aveg_cgm': -9999, 'cf_count_distance': -9999,
                                           'cf_pairwise_distance': -9999, 'if_generate_cf': -1
                                           }
            accumulated_reward_test_set_df = accumulated_reward_test_set_df.append(accumulated_reward_dict,
                                                                                       ignore_index=True)
        else:
            for item in accumulated_reward_list:
                orig_accu_r = item[0]
                cf_accu_r = item[1]
                perc = (cf_accu_r - orig_accu_r) / abs(orig_accu_r) if orig_accu_r != 0 else -9999
                effect_distance = effect_distance_list[accumulated_reward_list.index(item)]
                orig_effect = orig_effect_list[accumulated_reward_list.index(item)]
                cf_effect = cf_effect_list[accumulated_reward_list.index(item)]
                cf_aveg_cgm = cf_aveg_cgm_list[accumulated_reward_list.index(item)]
                cf_iob_distance = cf_IOB_distance_list[accumulated_reward_list.index(item)]
                orig_iob_distance = orig_IOB_distance_list[accumulated_reward_list.index(item)]
                orig_start_action_effect = orig_start_action_effect_list[accumulated_reward_list.index(item)]
                cf_count_distance = cf_distance_count_list[accumulated_reward_list.index(item)]
                cf_pairwise_distance = cf_pairwise_distance_list[accumulated_reward_list.index(item)]
                accumulated_reward_dict = {'id': test_counter, 'patient_type':patient_type, 'patient_id':patient_id,
                                           'orig_trace_episode': trace_eps,
                                               'orig_end_step': current_time_step,
                                               'orig_accumulated_reward': orig_accu_r,
                                               'cf_accumulated_reward': cf_accu_r,
                                               'difference': cf_accu_r - orig_accu_r, 'percentage': perc,
                                               'effect_distance': effect_distance, 'orig_effect': orig_effect,
                                               'cf_effect': cf_effect,
                                               'cf_iob_distance': cf_iob_distance,
                                               'orig_iob_distance': orig_iob_distance,
                                               'orig_start_action_effect': orig_start_action_effect,
                                               'cf_aveg_cgm': cf_aveg_cgm, 'cf_count_distance': cf_count_distance,
                                               'cf_pairwise_distance': cf_pairwise_distance, 'if_generate_cf': 1}
                accumulated_reward_test_set_df = accumulated_reward_test_set_df.append(accumulated_reward_dict,
                                                                                           ignore_index=True)

    else:
        # this original trace can not generate better CFs, change to another original trace
        accumulated_reward_dict = {'id': test_counter, 'patient_type':patient_type, 'patient_id':patient_id,
                                   'orig_trace_episode': trace_eps,
                                       'orig_end_step': current_time_step,
                                       'orig_accumulated_reward': -10000,
                                       'cf_accumulated_reward': -10000,
                                       'difference': -10000, 'percentage': -10000,
                                       'effect_distance': -10000, 'epsilon': -10000,
                                       'orig_effect': -10000,
                                       'cf_effect': -10000,
                                       'cf_iob_distance': -10000,
                                       'orig_iob_distance': -10000,
                                       'orig_start_action_effect': -10000,
                                       'cf_count_distance': -10000,
                                       'cf_pairwise_distance': -10000, 'if_generate_cf': 0}
        accumulated_reward_test_set_df = accumulated_reward_test_set_df.append(accumulated_reward_dict,
                                                                                   ignore_index=True)

    cf_trace_test_file_path = '{save_folder}/cf_trace_file_{mode}_trained_{train_counter}_{patient_type}_{patient_id}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline,
                                                        mode=mode, train_counter=train_counter, patient_type=patient_type, patient_id=patient_id, trace_eps=trace_eps, current_time_step=current_time_step)
    if len(CF_trace_test_list)!=0:
        CF_test_trace_total = pd.concat(CF_trace_test_list)
        CF_test_trace_total.to_csv(cf_trace_test_file_path)
        actual_test_index_path = '{save_folder}/actual_test_index_file_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline, trace_eps=trace_eps, current_time_step=current_time_step)
        #store_time_index(test_timestep_list, actual_test_index_path)
        total_orig_trace_test = pd.concat(orig_trace_df_list)
        total_orig_trace_test.to_csv('{save_folder}/orig_trace_test_{patient_type}_{patient_id}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline,
                                                                                                                                            patient_type=patient_type,
                                                                                                                                            patient_id=patient_id,
                                                                                                                trace_eps=trace_eps, current_time_step=current_time_step))
        # del total_orig_trace_test
    else:
        CF_test_trace_total = None
    accumulated_reward_test_set_df.to_csv('{save_folder}/accumulated_reward_{mode}_trained_{train_counter}_{patient_type}_{patient_id}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline,
                                                        mode=mode,train_counter=train_counter, patient_type=patient_type, patient_id=patient_id, trace_eps=trace_eps, current_time_step=current_time_step))
    # del accumulated_reward_test_set_df
    # gc.collect()
    return #index_already_test_list, CF_test_trace_total, accumulated_reward_test_set_df

def get_trace_info_for_reset_env(past_trace_this_eps, current_time_index, check_len, patient_id=None, patient_type=None, user_input_folder=None):
    # check_len: length of the generated CF trace
    max_reward_per_step = 1.0
    # get the initial state for env_CF, which is the check_len step in this eps
    CF_start_step = current_time_index - check_len + 1
    # if want to change actions starting from CF_start_step, then should reset env to state at this time step
    # Get the parameters for resetting env_CF
    past_ROC = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['observation_ROC'].tolist()[0]
    past_CGM = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['observation_CGM'].tolist()[0]
    past_CHO = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['observation_CHO'].tolist()[0]
    past_patient_state = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['patient_state'].tolist()[0]
    past_time = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['time'].tolist()[0]
    past_last_foodtaken = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['last_foodtaken'].tolist()[0]
    past_action = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['observation_insulin'].tolist()[0]
    past_bg = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['bg'].tolist()[0]
    past_cgm_sensor_params = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['cgm_sensor_params'].tolist()[0]
    past_cgm_sensor_seed = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['cgm_sensor_seed'].tolist()[0]
    past_cgm_sensor_last_cgm = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['cgm_sensor_last_cgm'].tolist()[0]
    past_scenario_seed = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['scenario_seed'].tolist()[0]
    past_scenario_random_gen = \
        past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['scenario_random_gen'].tolist()[0]
    past_scenario = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['scenario'].tolist()[0]
    past_is_eating = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['is_eating'].tolist()[0]
    #print('past_scenario: ', past_scenario)
    # meal_time_list = past_scenario['meal']['time']
    # meal_amount_list = past_scenario['meal']['amount']
    # #print('ZIP: ', list(zip(meal_time_list, meal_amount_list)))
    # past_scenario = list(zip(meal_time_list, meal_amount_list))
    past_patient_state_dict_cf = {'determine': 1, 'patient_state': past_patient_state,
                                  'start_time': (CF_start_step) * 3,
                                  # past_time, sample time in simulator is 3 step
                                  'last_foodtaken': past_last_foodtaken, 'insulin': past_action, 'bg': past_bg,'is_eating':past_is_eating,
                                  'CHO': past_CHO}
    past_sensor_state_dict_cf = {'determine': 1, 'sensor_params': past_cgm_sensor_params,
                                 'sensor_seed': past_cgm_sensor_seed,
                                 'sensor_last_cgm': past_cgm_sensor_last_cgm}
    past_scenario_state_dict_cf = {'determine': 1, 'scenario_seed': past_scenario_seed,
                                   'scenario_random_gen': past_scenario_random_gen,
                                   'scenario': past_scenario, 'start_time':past_time}
    # no states for pump reset
    kwargs_cf = {'determine': 1, 'patient_state': past_patient_state_dict_cf,
                 'sensor_state': past_sensor_state_dict_cf, 'pump_state': None,
                 'scenario_state': past_scenario_state_dict_cf,
                 'ROC': past_ROC, 'CGM': past_CGM}
    # print('Parameters for reset env: ', kwargs_cf)
    #env_CF_train = NormalizeObservation(gym.make(ENV_NAME))  # for training
    #env_CF_test = gym.make(ENV_NAME)  # for testing
    # state_dim = env_CF_train.observation_space.shape[0]
    # action_dim = env_CF_train.action_space.shape[0]
    num_actions = 1
    orig_trace_for_CF = past_trace_this_eps[CF_start_step-1:current_time_index]
    #print('orig_trace_for_CF index Train&Test: ', orig_trace_for_CF['episode'].tolist()[0], len(orig_trace_for_CF), current_time_index,' reset to: ', CF_start_step)
    # orig_trace_for_CF.to_csv(original_trace_file_path)
    orig_action_trace = orig_trace_for_CF['action'].tolist()
    orig_state_trace = orig_trace_for_CF['observation_CGM'].tolist()
    orig_start_action_effect = orig_trace_for_CF['action_effect'].tolist()[0]
    # print('orig_trace_for_CF: ', orig_trace_for_CF)

    # thre_fixed_step_index_list = []
    # for k in range(CF_start_step, current_time_index + 1):  # set some step as fixed action step
    #     if orig_action_trace[k - CF_start_step] < args_cf_diabetic.arg_thre_for_fix_a:
    #         thre_fixed_step_index_list.append(1)
    #     else:
    #         thre_fixed_step_index_list.append(0)
    thre_fixed_step_index_list = []
    thre_ddpg_action_index_list = []
    for k in range(CF_start_step, current_time_index + 1):  # set some step as fixed action step
        #print('k: ', k, ' k - CF_start_step: ', k - CF_start_step)
        if orig_action_trace[k - CF_start_step] < args_cf_diabetic.arg_thre_for_fix_a:
            # thre_fixed_step_index_list.append([1] * num_actions)
            # thre_ddpg_action_index_list.append([0] * num_actions)
            thre_fixed_step_index_list.append(1)
            thre_ddpg_action_index_list.append(0)
        else:
            # thre_fixed_step_index_list.append([0] * num_actions)
            # thre_ddpg_action_index_list.append([1] * num_actions)
            thre_fixed_step_index_list.append(0)
            thre_ddpg_action_index_list.append(1)
    if args_cf_diabetic.arg_if_user_assign_action == 1:
        this_eps = past_trace_this_eps['episode'].tolist()[0]
        user_input_this_segment_path = '{user_input_folder}/user_input_{patient_type}_patient_{patient_id}_{this_eps}_{current_time_index}.csv'.format(
            user_input_folder=user_input_folder,patient_id=patient_id,patient_type=patient_type,
            this_eps=this_eps, current_time_index=current_time_index)
        user_input_this_segment_df = pd.read_csv(user_input_this_segment_path)
        user_fixed_step_index_list = user_input_this_segment_df['step'].tolist()
        user_fixed_step_index_list = [(x-CF_start_step) for x in user_fixed_step_index_list]
    else:
        user_fixed_step_index_list = []
        user_input_this_segment_df = None
    #thre_fixed_step_index_list = list(set(thre_fixed_step_index_list).difference(set(user_fixed_step_index_list)))

    J0 = sum(orig_trace_for_CF['reward'].tolist())  # the accumulated reward of the original trace
    if J0 < args_cf_diabetic.arg_total_r_thre:  # only seek CF when orig trace doen not have the max reward, else there won't be any improvement
        return kwargs_cf, orig_trace_for_CF, orig_action_trace, orig_state_trace, orig_start_action_effect, \
            J0, thre_fixed_step_index_list, thre_ddpg_action_index_list, user_fixed_step_index_list, user_input_this_segment_df
    else:
        print('No better CF to find.')
        return None

# train and test for training with multiple trace from multiple/single patient
#@profile(precision=4,stream=open("/home/cpsgroup/Counterfactual_Explanation/diabetic_example/results/memory_profiler_Run_Single_Exp.log","w+"))
def run_experiment_ddpg_sb3_single(model_CF, trace_eps, current_time_step, fixed_trace_this_eps_df, parameter_dict, env_name,
                                   this_train_round, mode='train',baseline_result_dict=None, model_type='baseline'):
    # parameters
    orig_trace_episode = trace_eps
    CF_start_step = current_time_step - args_cf_diabetic.arg_cflen + 1
    iob_param = parameter_dict['iob_param']
    patient_BW = parameter_dict['patient_BW']
    cf_len = parameter_dict['cf_len']
    patient_type = parameter_dict['patient_type']
    patient_id = parameter_dict['patient_id']
    #train_round = parameter_dict['train_round']
    total_test_trace_num = parameter_dict['total_test_trace_num']
    #ENV_NAME = parameter_dict['ENV_NAME']
    if args_cf_diabetic.arg_if_user_assign_action == 1:
        user_input_folder = '' # undecided yet
    else:
        user_input_folder = None
    reset_env_results = get_trace_info_for_reset_env(past_trace_this_eps=fixed_trace_this_eps_df, current_time_index=current_time_step,
                                    check_len=cf_len, patient_id=patient_id, patient_type=patient_type, user_input_folder=user_input_folder)
    if mode=='train':
        if reset_env_results is not None:
            (kwargs_cf, orig_trace_for_CF, orig_action_trace, orig_state_trace, orig_start_action_effect, J0,
             thre_fixed_step_index_list, thre_ddpg_action_index_list, user_fixed_step_index_list, user_input_this_segment_df) = reset_env_results
            #_,all_cf_trace_training_time,all_train_result = \
            model_CF.learn(
                log_interval=parameter_dict['log_interval'],
                kwargs_cf=kwargs_cf,
                train_round=this_train_round,
                patient_BW=patient_BW,
                iob_param=iob_param,
                CF_start_step=CF_start_step,
                current_time_index=current_time_step,
                orig_trace_episode=orig_trace_episode,
                J0=J0,
                orig_action_trace=orig_action_trace,
                orig_state_trace=orig_state_trace,
                ENV_ID=env_name,
                fixed_step_index_list=thre_fixed_step_index_list,
                ddpg_step_index_list=thre_ddpg_action_index_list,
                user_fixed_step_index_list=user_fixed_step_index_list,
                user_input_this_segment_df=user_input_this_segment_df,
                baseline_result_dict=baseline_result_dict,
                arg_reward_weight=args_cf_diabetic.arg_reward_weight,
                if_constrain_on_s=args_cf_diabetic.arg_if_constrain_on_s,
                arg_thre_for_s=args_cf_diabetic.arg_thre_for_s,
                arg_s_index=args_cf_diabetic.arg_s_index,
                arg_if_use_user_input=args_cf_diabetic.arg_if_use_user_input,
                user_input_action=args_cf_diabetic.arg_user_input_action
            )
        else:
            all_cf_trace_training_time, all_train_result = None, None
        return #all_cf_trace_training_time, all_train_result
    else:
        if reset_env_results is not None:
            (kwargs_cf, orig_trace_for_CF, orig_action_trace, orig_state_trace, orig_start_action_effect, J0,
             thre_fixed_step_index_list, thre_ddpg_action_index_list, user_fixed_step_index_list, user_input_this_segment_df) = reset_env_results
            all_cf_trace_test_list, all_test_result, all_test_statistic = model_CF.test(
                train_round=this_train_round,
                patient_BW=patient_BW,
                iob_param=iob_param,
                kwargs_cf=kwargs_cf,
                CF_start_step=CF_start_step,
                current_time_index=current_time_step,
                orig_trace_episode=orig_trace_episode,
                J0=J0,
                orig_action_trace=orig_action_trace,
                orig_state_trace=orig_state_trace,
                total_test_trace_num=total_test_trace_num,
                #ENV_NAME=ENV_NAME,
                ENV_ID=env_name,
                fixed_step_index_list=thre_fixed_step_index_list,
                ddpg_step_index_list=thre_ddpg_action_index_list,
                user_fixed_step_index_list=user_fixed_step_index_list,
                user_input_this_segment_df=user_input_this_segment_df,
                baseline_result_dict=baseline_result_dict,
                model_type=model_type,
                if_constrain_on_s=args_cf_diabetic.arg_if_constrain_on_s,
                arg_thre_for_s=args_cf_diabetic.arg_thre_for_s,
                arg_s_index=args_cf_diabetic.arg_s_index,
                arg_if_use_user_input=args_cf_diabetic.arg_if_use_user_input,
                user_input_action=args_cf_diabetic.arg_user_input_action
            )
        else:
            all_cf_trace_test_list, all_test_result, all_test_statistic = None, None, None
        return all_cf_trace_test_list, all_test_result,  all_test_statistic

#@profile(precision=4,stream=open("/home/cpsgroup/Counterfactual_Explanation/diabetic_example/results/memory_profiler_Train_All.log","w+"))
def train_ddpg_sb3_all(all_train_index_df, all_test_index_df, ENV_NAME_dummy, save_folder, callback,
                       trained_controller_dict, all_env_dict, trace_df_all_patient_dict, ENV_ID_list,baseline_result_folder,trial_index):
    # for server ver
    save_folder_train = '{save_folder}/train'.format(save_folder=save_folder)
    save_folder_test = '{save_folder}/test'.format(save_folder=save_folder)
    # if os.path.exists(save_folder_train):
    #     shutil.rmtree(save_folder_train)
    #     print('DEL.')
    # if os.path.exists(save_folder_test):
    #     shutil.rmtree(save_folder_test)
    #     print('DEL.')
    mkdir(save_folder_train)
    mkdir(save_folder_test)
    patient_info_input_dim = 13
    patient_info_output_dim = args_cf_diabetic.arg_patidim
    with_encoder = args_cf_diabetic.arg_with_encoder
    iob_param = args_cf_diabetic.arg_iob_param
    delta_dist = args_cf_diabetic.arg_delta_dist
    patient_BW = 0#this_patient_BW
    epsilon = args_cf_diabetic.arg_epsilon
    cf_len = args_cf_diabetic.arg_cflen
    generate_train_trace_num = args_cf_diabetic.arg_generate_train_trace_num  # how many cf traces generated and pyshed to buffer each time conduct collect_rollout
    total_test_trace_num = args_cf_diabetic.arg_total_test_trace_num  # how many cf traces generated for generating test result
    #test_interval = args_cf_diabetic.arg_test_interval
    gradient_steps = args_cf_diabetic.arg_gradient_steps
    #total_train_steps = args_cf_diabetic.arg_total_train_steps
    log_interval = args_cf_diabetic.arg_log_interval
    dist_func = args_cf_diabetic.arg_dist_func
    batch_size = args_cf_diabetic.arg_batch_size
    lambda_start_value = args_cf_diabetic.arg_lambda_start_value
    learning_rate = args_cf_diabetic.arg_learning_rate
    total_train_round = args_cf_diabetic.arg_train_round # total train round for DDPG
    total_timesteps_each_trace = args_cf_diabetic.arg_total_timesteps_each_trace
    #ENV_NAME = args_cf_diabetic.arg_ENV_NAME

    # for saving training results
    all_train_cf_trace_list = []
    all_train_result_list = []

    # set training log
    tmp_path = "{save_folder}/logs/train_log".format(save_folder=save_folder)
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    env_CF_train = NormalizeObservation(gym.make(ENV_NAME_dummy))
    # The noise objects for DDPG
    n_actions = env_CF_train.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # set params
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 64], qf=[256, 64]))
    # policy_kwargs = dict(net_arch=dict(pi=[256, 128], qf=[256, 128]))
    model_CF = TD3_CF(policy="MlpPolicy", env=env_CF_train, action_noise=action_noise, verbose=0,
                       learning_rate=learning_rate,
                       gradient_steps=gradient_steps,
                       batch_size=batch_size,
                       total_timesteps_each_trace=total_timesteps_each_trace,
                       callback=callback,
                       cf_len=cf_len,
                       generate_train_trace_num=generate_train_trace_num,
                       epsilon=epsilon,
                       delta_dist=delta_dist,
                       patient_info_input_dim=patient_info_input_dim,
                       patient_info_output_dim=patient_info_output_dim,
                       with_encoder=with_encoder,
                       dist_func=dist_func,
                       lambda_start_value=lambda_start_value,
                       policy_kwargs=policy_kwargs,
                       #ENV_NAME=ENV_NAME,
                       trained_controller_dict=trained_controller_dict,
                       all_env_dict=all_env_dict,
                       )
    model_CF.set_logger(new_logger)
    test_trace_df_list = []
    test_result_df_list = []
    # all_train_cf_trace_df = pd.DataFrame()
    # all_train_result_df = pd.DataFrame()
    for r in range(1, total_train_round+1): # train with all orig traces a few rounds
        counter = 0
        #all_train_index_df = all_train_index_df.sample(frac=1)
        random_train_trace_index = np.random.randint(0, len(all_train_index_df))
        time_3 = time.perf_counter()
        mem_before_train_1_trace = get_mem_use()
        for item, row in all_train_index_df.loc[[random_train_trace_index]].iterrows():
            env_name = row['ENV_NAME']
            trace_eps = row['orig_episode']
            current_time_step = row['orig_end_time_index']
            patient_type = row['patient_type']
            patient_id = row['patient_id']
            print('Train: ', r, counter,env_name,trace_eps, current_time_step)
            parameter_dict = {'patient_type':patient_type, 'patient_id':patient_id, 'patient_info_input_dim': patient_info_input_dim,
                              'patient_info_output_dim': patient_info_output_dim,
                              'with_encoder': with_encoder, 'iob_param': iob_param, 'delta_dist': delta_dist,
                              'patient_BW': patient_BW,
                              'epsilon': epsilon, 'cf_len': cf_len,
                              'generate_train_trace_num': generate_train_trace_num,
                              'total_test_trace_num': total_test_trace_num,
                              #'test_interval': test_interval,
                              'gradient_steps': gradient_steps,
                              #'total_train_steps': total_train_steps,
                              'log_interval': log_interval, 'dist_func': dist_func, 'batch_size': batch_size,
                              'lambda_start_value': lambda_start_value, 'learning_rate': learning_rate,
                              'train_round': r,
                              'ENV_NAME': env_name,
                              }
            # fixed_trace_this_eps_df = trace_df[trace_df["episode"] == trace_eps]
            fixed_orig_past_trace_df = trace_df_all_patient_dict[env_name]
            fixed_trace_this_eps_df = fixed_orig_past_trace_df[fixed_orig_past_trace_df['episode'] == trace_eps]
            #time_1 = time.perf_counter()
            # # get the distance value and the total reward of this trace from baseline
            # baseline_file_path = '{baseline_result_folder}/accumulated_reward_test_baseline_trained_0_{patient_type}_{patient_id}_{trace_eps}_{current_time_step}.csv'.format(
            #     baseline_result_folder=baseline_result_folder, patient_type=patient_type, patient_id=patient_id,
            #     trace_eps=trace_eps,
            #     current_time_step=current_time_step)
            # baseline_df = pd.read_csv(baseline_file_path)
            # baseline_total_reward_mean = baseline_df['cf_accumulated_reward'].mean()
            # baseline_total_reward_difference_mean = baseline_df['difference'].mean()
            # baseline_distance_mean = baseline_df['cf_pairwise_distance'].mean()
            # baseline_result_dict = {'baseline_total_reward_mean': baseline_total_reward_mean,
            #                         'baseline_total_reward_difference_mean': baseline_total_reward_difference_mean,
            #                         'baseline_distance_mean': baseline_distance_mean}
            #all_cf_trace_training_time_this_seg,all_train_result_this_seg = \
            run_experiment_ddpg_sb3_single(model_CF, trace_eps, current_time_step, fixed_trace_this_eps_df, parameter_dict,
                                                                    env_name, this_train_round=r, mode='train',
                                                                    baseline_result_dict=None)
            # del baseline_df, baseline_result_dict
            gc.collect()
            # del all_cf_trace_training_time_this_seg
            # del all_train_result_this_seg

            #time_2 = time.perf_counter()
            #print('Diabetic, time for training 1 original trace: ', time_2 - time_1)
            # if all_cf_trace_training_time_this_seg is not None:
            #     #all_train_cf_trace_list.append(all_cf_trace_training_time_this_seg)
            #     #all_train_result_list.append(all_train_result_this_seg)
            #     #time_6 = time.perf_counter()
            #     #print('Diabetic, time for Append result for 1 original trace: ', time_6 - time_2)
            #
            #     #all_train_cf_trace_df = pd.concat(all_train_cf_trace_list)
            #     #all_train_result_df = pd.concat(all_train_result_list)
            #     # all_train_cf_trace_df = pd.concat([all_train_cf_trace_df, all_cf_trace_training_time_this_seg])
            #     # all_train_result_df = pd.concat([all_train_result_df, all_train_result_this_seg])
            #     # time_7 = time.perf_counter()
            #     # print('Diabetic, time for Concat result for 1 original trace: ', time_7 - time_2)
            #     # all_cf_trace_training_time_this_seg.to_csv(
            #     #     '{save_folder_train}/cf_traces_train_{patient_type}_{patient_id}_{trace_eps}_{current_time_step}_r_{r}.csv'.format(
            #     #         save_folder_train=save_folder_train,
            #     #         patient_type=patient_type, patient_id=patient_id,
            #     #         trace_eps=trace_eps, current_time_step=current_time_step, r=r))
            #     # all_train_result_this_seg.to_csv('{save_folder_train}/train_result_{patient_type}_{patient_id}_{trace_eps}_{current_time_step}_r_{r}.csv'.format(save_folder_train=save_folder_train,
            #     #                                                                                                         patient_type=patient_type,patient_id=patient_id,
            #     #                                                                                                        trace_eps=trace_eps, current_time_step=current_time_step, r=r))
            #     #
            #     all_cf_trace_training_time_this_seg.to_pickle(
            #         '{save_folder_train}/cf_traces_train_{patient_type}_{patient_id}_{trace_eps}_{current_time_step}_r_{r}.pkl'.format(
            #             save_folder_train=save_folder_train,
            #             patient_type=patient_type, patient_id=patient_id,
            #             trace_eps=trace_eps, current_time_step=current_time_step, r=r))
            #     all_train_result_this_seg.to_pickle(
            #         '{save_folder_train}/train_result_{patient_type}_{patient_id}_{trace_eps}_{current_time_step}_r_{r}.pkl'.format(
            #             save_folder_train=save_folder_train,
            #             patient_type=patient_type, patient_id=patient_id,
            #             trace_eps=trace_eps, current_time_step=current_time_step, r=r))
            #
            #     # all_train_cf_trace_df.to_csv('{save_folder}/all_cf_traces_train.csv'.format(save_folder=save_folder))
            #     # all_train_result_df.to_csv('{save_folder}/all_train_result.csv'.format(save_folder=save_folder))
            #     # all_train_cf_trace_df.to_pickle('{save_folder}/all_cf_traces_train.pkl'.format(save_folder=save_folder))
            #     # all_train_result_df.to_pickle('{save_folder}/all_train_result.pkl'.format(save_folder=save_folder))
            #     #time_8 = time.perf_counter()
            #     #print('Diabetic, time for Saving result for 1 original trace: ', time_8 - time_2)
            # if counter==int(len(all_train_index_df)*0.5):
            #     time_3 = time.perf_counter()
            #     all_test_cf_trace_df_this_round, all_test_result_df_this_round = test_ddpg_sb3_all(model_CF,
            #                                                                                        all_test_index_df,
            #                                                                                        save_folder,
            #                                                                                        trace_df_all_patient_dict,
            #                                                                                        this_train_round=(r-0.5))
            #     time_4 = time.perf_counter()
            #     print('Lunar Lander, time for testing 1 original trace: ', time_4 - time_3)
            #     test_trace_df_list.append(all_test_cf_trace_df_this_round)
            #     test_result_df_list.append(all_test_result_df_this_round)
            #     all_test_cf_trace_df = pd.concat(test_trace_df_list)
            #     all_test_result_df = pd.concat(test_result_df_list)
            #     all_test_cf_trace_df.to_csv('{save_folder}/all_cf_traces_test.csv'.format(save_folder=save_folder))
            #     all_test_result_df.to_csv('{save_folder}/all_test_result.csv'.format(save_folder=save_folder))
            counter += 1
        mem_after_train_1_trace = get_mem_use()
        #print('Mem usage, Train 1 round: ', mem_after_train_1_trace - mem_before_train_1_trace, ' Total: ', mem_after_train_1_trace)
        time_4 = time.perf_counter()
        #print('Diabetic, time for Training One Full Round: ', time_4 - time_3)
        mem_before_test_1_trace = get_mem_use()
        if r%args_cf_diabetic.arg_test_param==0 and r>=20:
            print('Test.')
            test_ddpg_sb3_all(model_CF, all_test_index_df, save_folder_test, trace_df_all_patient_dict,
                              this_train_round=r,baseline_result_folder=baseline_result_folder,trial_index=trial_index)
            test_ddpg_sb3_all(model_CF, all_train_index_df, save_folder_train, trace_df_all_patient_dict,
                              this_train_round=r, baseline_result_folder=baseline_result_folder, trial_index=trial_index)
            #mem_after_test_1_trace = get_mem_use()
            #print('Mem usage, Test 1 round: ', mem_after_test_1_trace - mem_before_test_1_trace, ' Total: ',mem_after_test_1_trace)
            #time_5 = time.perf_counter()
        # if r%1==0:
        #     # all_test_cf_trace_df_this_round, all_test_result_df_this_round = \
        #     test_ddpg_sb3_all(model_CF, all_test_index_df,save_folder_test, trace_df_all_patient_dict, this_train_round=r,
        #                       baseline_result_folder=baseline_result_folder)
        #     mem_after_test_1_trace = get_mem_use()
        #     #print('Mem usage, Test 1 round: ', mem_after_test_1_trace - mem_before_test_1_trace, ' Total: ', mem_after_test_1_trace)
        #     #time_5 = time.perf_counter()
        #     #print('Diabetic, time for Testing One Full Round: ', time_5 - time_4)
        # test_trace_df_list.append(all_test_cf_trace_df_this_round)
        # test_result_df_list.append(all_test_result_df_this_round)
        # all_test_cf_trace_df = pd.concat(test_trace_df_list)
        # all_test_result_df = pd.concat(test_result_df_list)
        # all_test_cf_trace_df.to_csv('{save_folder}/all_cf_traces_test.csv'.format(save_folder=save_folder))
        # all_test_result_df.to_csv('{save_folder}/all_test_result.csv'.format(save_folder=save_folder))
        # all_test_cf_trace_df_this_round.to_pickle('{save_folder_test}/cf_traces_test_r_{r}.pkl'.format(save_folder_test=save_folder_test, r=r))
        # all_test_result_df_this_round.to_pickle('{save_folder_test}/test_result_r_{r}.pkl'.format(save_folder_test=save_folder_test, r=r))
        #time_6 = time.perf_counter()
        # del all_test_cf_trace_df_this_round
        # del all_test_result_df_this_round
        # gc.collect()
        #print('Diabetic, Save Test Result Time: ', time_6 - time_5)

    return

#@profile(precision=4,stream=open("/home/cpsgroup/Counterfactual_Explanation/diabetic_example/results/memory_profiler_Test_All.log","w+"))
def test_ddpg_sb3_all(model_CF, all_test_index_df, save_folder, trace_df_all_patient_dict, this_train_round,baseline_result_folder,trial_index):
    # for server ver
    patient_info_input_dim = 13
    patient_info_output_dim = args_cf_diabetic.arg_patidim
    with_encoder = args_cf_diabetic.arg_with_encoder
    iob_param = args_cf_diabetic.arg_iob_param
    delta_dist = args_cf_diabetic.arg_delta_dist
    patient_BW = 0#this_patient_BW
    epsilon = args_cf_diabetic.arg_epsilon
    cf_len = args_cf_diabetic.arg_cflen
    generate_train_trace_num = args_cf_diabetic.arg_generate_train_trace_num  # how many cf traces generated and pyshed to buffer each time conduct collect_rollout
    total_test_trace_num = args_cf_diabetic.arg_total_test_trace_num  # how many cf traces generated for generating test result
    #test_interval = args_cf_diabetic.arg_test_interval
    gradient_steps = args_cf_diabetic.arg_gradient_steps
    #total_train_steps = args_cf_diabetic.arg_total_train_steps
    log_interval = args_cf_diabetic.arg_log_interval
    dist_func = args_cf_diabetic.arg_dist_func
    batch_size = args_cf_diabetic.arg_batch_size
    lambda_start_value = args_cf_diabetic.arg_lambda_start_value
    learning_rate = args_cf_diabetic.arg_learning_rate
    #train_round = args_cf_diabetic.arg_train_round  # total train round for DDPG
    #ENV_NAME = args_cf_diabetic.arg_ENV_NAME
    #total_timesteps_each_trace = args_cf_diabetic.arg_total_timesteps_each_trace

    # for saving training results
    all_test_cf_trace_list_ddpg = []
    all_test_result_list_ddpg = []
    all_test_statistic_list_ddpg = []
    all_test_cf_trace_list_baseline = []
    all_test_result_list_baseline = []
    all_test_statistic_list_baseline = []

    #env_CF_test = NormalizeObservation(gym.make(ENV_NAME))

    for item, row in all_test_index_df.iterrows():
        env_name = row['ENV_NAME']
        trace_eps = row['orig_episode']
        current_time_step = row['orig_end_time_index']
        patient_type = row['patient_type']
        patient_id = row['patient_id']
        print('Test: ', env_name, trace_eps, current_time_step)
        parameter_dict = {'patient_type':patient_type, 'patient_id':patient_id, 'patient_info_input_dim': patient_info_input_dim,
                              'patient_info_output_dim': patient_info_output_dim,
                              'with_encoder': with_encoder, 'iob_param': iob_param, 'delta_dist': delta_dist,
                              'patient_BW': patient_BW,
                              'epsilon': epsilon, 'cf_len': cf_len,
                              'generate_train_trace_num': generate_train_trace_num,
                              'total_test_trace_num': total_test_trace_num,
                              #'test_interval': test_interval,
                              'gradient_steps': gradient_steps,
                              #'total_train_steps': total_train_steps,
                              'log_interval': log_interval, 'dist_func': dist_func, 'batch_size': batch_size,
                              'lambda_start_value': lambda_start_value, 'learning_rate': learning_rate,
                              'train_round': this_train_round,
                                #'ENV_NAME': ENV_NAME,
                                }
        # fixed_trace_this_eps_df = trace_df[trace_df["episode"] == trace_eps]
        fixed_orig_past_trace_df = trace_df_all_patient_dict[env_name]
        fixed_trace_this_eps_df = fixed_orig_past_trace_df[fixed_orig_past_trace_df['episode'] == trace_eps]
        # # get the distance value and the total reward of this trace from baseline
        # baseline_file_path = '{baseline_result_folder}/accumulated_reward_test_baseline_trained_0_{patient_type}_{patient_id}_{trace_eps}_{current_time_step}.csv'.format(
        #     baseline_result_folder=baseline_result_folder, patient_type=patient_type, patient_id=patient_id, trace_eps=trace_eps,
        #     current_time_step=current_time_step)
        # baseline_df = pd.read_csv(baseline_file_path)
        # baseline_total_reward_mean = baseline_df['cf_accumulated_reward'].mean()
        # baseline_total_reward_difference_mean = baseline_df['difference'].mean()
        # baseline_distance_mean = baseline_df['cf_pairwise_distance'].mean()
        # baseline_result_dict = {'baseline_total_reward_mean': baseline_total_reward_mean,
        #                         'baseline_total_reward_difference_mean': baseline_total_reward_difference_mean,
        #                         'baseline_distance_mean': baseline_distance_mean}
        # #time_6 = time.perf_counter()
        # all_cf_trace_test_list_this_seg, all_test_result_this_seg = run_experiment_ddpg_sb3_single(model_CF, trace_eps,
        #                                                                 current_time_step, fixed_trace_this_eps_df, parameter_dict,env_name,
        #                                                                     this_train_round=this_train_round,mode='test',
        #                                                                                            baseline_result_dict=baseline_result_dict)
        #print('Test DDPG.')
        all_cf_trace_test_list_this_seg_ddpg, all_test_result_this_seg_ddpg, test_statistic_dict_this_seg_ddpg = run_experiment_ddpg_sb3_single(
            model_CF, trace_eps,
            current_time_step,
            fixed_trace_this_eps_df,
            parameter_dict,
            env_name,
            this_train_round=this_train_round,
            mode='test',
            baseline_result_dict=None, model_type='ddpg')
        #print('Test PPO.')
        all_cf_trace_test_list_this_seg_baseline, all_test_result_this_seg_baseline, test_statistic_dict_this_seg_baseline = run_experiment_ddpg_sb3_single(
            model_CF, trace_eps,
            current_time_step,
            fixed_trace_this_eps_df,
            parameter_dict,
            env_name,
            this_train_round=this_train_round,
            mode='test',
            baseline_result_dict=None, model_type='baseline')

        if all_cf_trace_test_list_this_seg_ddpg is not None:
            all_test_cf_trace_list_ddpg.extend(all_cf_trace_test_list_this_seg_ddpg)
            all_test_result_list_ddpg.append(all_test_result_this_seg_ddpg)
            all_test_statistic_list_ddpg.append(pd.DataFrame([test_statistic_dict_this_seg_ddpg]))
        if all_cf_trace_test_list_this_seg_baseline is not None:
            all_test_cf_trace_list_baseline.extend(all_cf_trace_test_list_this_seg_baseline)
            all_test_result_list_baseline.append(all_test_result_this_seg_baseline)
            all_test_statistic_list_baseline.append(pd.DataFrame([test_statistic_dict_this_seg_baseline]))
        # #time_7 = time.perf_counter()
        # #print('Diabetic, time for testing 1 original trace: ', time_7 - time_6)
        # if all_cf_trace_test_list_this_seg is not None:
        #     #all_cf_trace_test_this_seg = pd.concat(all_cf_trace_test_list_this_seg)
        #     all_test_cf_trace_list.extend(all_cf_trace_test_list_this_seg)
        #     all_test_result_list.append(all_test_result_this_seg)
    all_test_cf_trace_df_ddpg = pd.concat(all_test_cf_trace_list_ddpg)
    all_test_result_df_ddpg = pd.concat(all_test_result_list_ddpg)
    all_test_statistic_df_ddpg = pd.concat(all_test_statistic_list_ddpg)
    all_test_cf_trace_df_ddpg.to_pickle(
        '{save_folder_test}/cf_traces_test_ddpg_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
    all_test_result_df_ddpg.to_pickle(
        '{save_folder_test}/test_result_ddpg_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
    all_test_statistic_df_ddpg.to_pickle(
        '{save_folder_test}/test_statistic_ddpg_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
    all_test_cf_trace_df_baseline = pd.concat(all_test_cf_trace_list_baseline)
    all_test_result_df_baseline = pd.concat(all_test_result_list_baseline)
    all_test_statistic_df_baseline = pd.concat(all_test_statistic_list_baseline)
    all_test_cf_trace_df_baseline.to_pickle(
        '{save_folder_test}/cf_traces_test_baseline_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
    all_test_result_df_baseline.to_pickle(
        '{save_folder_test}/test_result_baseline_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
    all_test_statistic_df_baseline.to_pickle(
        '{save_folder_test}/test_statistic_baseline_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
    del all_test_cf_trace_df_ddpg, all_test_result_df_ddpg, all_test_statistic_df_ddpg
    del all_cf_trace_test_list_this_seg_ddpg, all_test_result_this_seg_ddpg, test_statistic_dict_this_seg_ddpg
    del all_test_cf_trace_df_baseline, all_test_result_df_baseline, all_test_statistic_df_baseline
    del all_cf_trace_test_list_this_seg_baseline, all_test_result_this_seg_baseline, test_statistic_dict_this_seg_baseline
    # del baseline_df, baseline_result_dict
    gc.collect()
    # all_test_cf_trace_df = pd.concat(all_test_cf_trace_list)
    # all_test_result_df = pd.concat(all_test_result_list)
    # all_test_cf_trace_df.to_pickle('{save_folder_test}/cf_traces_test_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
    # all_test_result_df.to_pickle('{save_folder_test}/test_result_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
    # del all_test_cf_trace_df
    # del all_test_result_df
    # del baseline_df, baseline_result_dict
    gc.collect()
    # all_test_cf_trace_df.to_csv('{save_folder}/all_cf_traces_test.csv'.format(save_folder=save_folder))
    # all_test_result_df.to_csv('{save_folder}/all_test_result.csv'.format(save_folder=save_folder))

    return #all_test_cf_trace_df, all_test_result_df

# train and test for training with 1 trace from 1 patient
def run_experiment_ddpg_sb3_1_trace_1_patient(model_CF, trace_eps, current_time_step, fixed_trace_this_eps_df, parameter_dict, env_name,
                                   this_train_round, mode='train'):
    # parameters
    orig_trace_episode = trace_eps
    CF_start_step = current_time_step - args_cf_diabetic.arg_cflen + 1
    iob_param = parameter_dict['iob_param']
    patient_BW = parameter_dict['patient_BW']
    cf_len = parameter_dict['cf_len']
    patient_type = parameter_dict['patient_type']
    patient_id = parameter_dict['patient_id']
    #train_round = parameter_dict['train_round']
    total_test_trace_num = parameter_dict['total_test_trace_num']
    #ENV_NAME = parameter_dict['ENV_NAME']
    if args_cf_diabetic.arg_if_user_assign_action == 1:
        user_input_folder = '' # undecided yet
    else:
        user_input_folder = None
    reset_env_results = get_trace_info_for_reset_env(past_trace_this_eps=fixed_trace_this_eps_df, current_time_index=current_time_step,
                                    check_len=cf_len, patient_id=patient_id, patient_type=patient_type, user_input_folder=user_input_folder)
    if mode=='train':
        if reset_env_results is not None:
            (kwargs_cf, orig_trace_for_CF, orig_action_trace, orig_state_trace, orig_start_action_effect, J0,
             thre_fixed_step_index_list, user_fixed_step_index_list, user_input_this_segment_df) = reset_env_results
            _,all_cf_trace_training_time,all_train_result = model_CF.learn(
                log_interval=parameter_dict['log_interval'],
                kwargs_cf=kwargs_cf,
                train_round=this_train_round,
                patient_BW=patient_BW,
                iob_param=iob_param,
                CF_start_step=CF_start_step,
                current_time_index=current_time_step,
                orig_trace_episode=orig_trace_episode,
                J0=J0,
                orig_action_trace=orig_action_trace,
                orig_state_trace=orig_state_trace,
                ENV_ID=env_name,
                fixed_step_index_list=thre_fixed_step_index_list,
                user_fixed_step_index_list=user_fixed_step_index_list,
                user_input_this_segment_df=user_input_this_segment_df,
            )
        else:
            all_cf_trace_training_time, all_train_result = None, None
        return all_cf_trace_training_time, all_train_result
    else:
        if reset_env_results is not None:
            (kwargs_cf, orig_trace_for_CF, orig_action_trace, orig_state_trace, orig_start_action_effect, J0,
             thre_fixed_step_index_list, user_fixed_step_index_list, user_input_this_segment_df) = reset_env_results
            all_cf_trace_test, all_test_result = model_CF.test(
                train_round=this_train_round,
                patient_BW=patient_BW,
                iob_param=iob_param,
                kwargs_cf=kwargs_cf,
                CF_start_step=CF_start_step,
                current_time_index=current_time_step,
                orig_trace_episode=orig_trace_episode,
                J0=J0,
                orig_action_trace=orig_action_trace,
                orig_state_trace=orig_state_trace,
                total_test_trace_num=total_test_trace_num,
                #ENV_NAME=ENV_NAME,
                ENV_ID=env_name,
                fixed_step_index_list=thre_fixed_step_index_list,
                user_fixed_step_index_list=user_fixed_step_index_list,
                user_input_this_segment_df=user_input_this_segment_df,
            )
        else:
            all_cf_trace_test, all_test_result = None, None
        return all_cf_trace_test, all_test_result

def train_ddpg_sb3_all_1_trace_1_patient(all_train_index_df, ENV_NAME_dummy, save_folder, callback,
                       trained_controller_dict, all_env_dict, trace_df_all_patient_dict):
    # for server ver
    patient_info_input_dim = 13
    patient_info_output_dim = args_cf_diabetic.arg_patidim
    with_encoder = args_cf_diabetic.arg_with_encoder
    iob_param = args_cf_diabetic.arg_iob_param
    delta_dist = args_cf_diabetic.arg_delta_dist
    patient_BW = 0#this_patient_BW
    epsilon = args_cf_diabetic.arg_epsilon
    cf_len = args_cf_diabetic.arg_cflen
    generate_train_trace_num = args_cf_diabetic.arg_generate_train_trace_num  # how many cf traces generated and pyshed to buffer each time conduct collect_rollout
    total_test_trace_num = args_cf_diabetic.arg_total_test_trace_num  # how many cf traces generated for generating test result
    #test_interval = args_cf_diabetic.arg_test_interval
    gradient_steps = args_cf_diabetic.arg_gradient_steps
    #total_train_steps = args_cf_diabetic.arg_total_train_steps
    log_interval = args_cf_diabetic.arg_log_interval
    dist_func = args_cf_diabetic.arg_dist_func
    batch_size = args_cf_diabetic.arg_batch_size
    lambda_start_value = args_cf_diabetic.arg_lambda_start_value
    learning_rate = args_cf_diabetic.arg_learning_rate
    total_train_round = args_cf_diabetic.arg_train_round # total train round for DDPG
    total_timesteps_each_trace = args_cf_diabetic.arg_total_timesteps_each_trace
    #ENV_NAME = args_cf_diabetic.arg_ENV_NAME

    # for saving training results
    all_train_cf_trace_list = []
    all_train_result_list = []
    test_trace_df_list = []
    test_result_df_list = []
    for item, row in all_train_index_df.iterrows():
    #for r in range(1, total_train_round+1): # train with all orig traces a few rounds
        counter = 0
        #all_train_index_df = all_train_index_df.sample(frac=1)
        env_name = row['ENV_NAME']
        trace_eps = row['orig_episode']
        current_time_step = row['orig_end_time_index']
        patient_type = row['patient_type']
        patient_id = row['patient_id']
        test_index_df = all_train_index_df[item:item+1]
        print('test_index_df: ', test_index_df)

        # fixed_trace_this_eps_df = trace_df[trace_df["episode"] == trace_eps]
        fixed_orig_past_trace_df = trace_df_all_patient_dict[env_name]
        fixed_trace_this_eps_df = fixed_orig_past_trace_df[fixed_orig_past_trace_df['episode'] == trace_eps]
        # set up a new model for this trace
        # set training log
        tmp_path = "{save_folder}/logs/train_log".format(save_folder=save_folder)
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        env_CF_train = NormalizeObservation(gym.make(ENV_NAME_dummy))
        # The noise objects for DDPG
        n_actions = env_CF_train.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        # set params
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 64], qf=[256, 64]))
        # policy_kwargs = dict(net_arch=dict(pi=[256, 128], qf=[256, 128]))
        model_CF = TD3_CF(policy="MlpPolicy", env=env_CF_train, action_noise=action_noise, verbose=1,
                           learning_rate=learning_rate,
                           gradient_steps=gradient_steps,
                           batch_size=batch_size,
                           total_timesteps_each_trace=total_timesteps_each_trace,
                           callback=callback,
                           cf_len=cf_len,
                           generate_train_trace_num=generate_train_trace_num,
                           epsilon=epsilon,
                           delta_dist=delta_dist,
                           patient_info_input_dim=patient_info_input_dim,
                           patient_info_output_dim=patient_info_output_dim,
                           with_encoder=with_encoder,
                           dist_func=dist_func,
                           lambda_start_value=lambda_start_value,
                           policy_kwargs=policy_kwargs,
                           # ENV_NAME=ENV_NAME,
                           trained_controller_dict=trained_controller_dict,
                           all_env_dict=all_env_dict,
                           )
        model_CF.set_logger(new_logger)
        for r in range(1, total_train_round+1): # train with 1 orig traces a few rounds
            print('Train: ', r, env_name, trace_eps, current_time_step)
            parameter_dict = {'patient_type': patient_type, 'patient_id': patient_id,
                              'patient_info_input_dim': patient_info_input_dim,
                              'patient_info_output_dim': patient_info_output_dim,
                              'with_encoder': with_encoder, 'iob_param': iob_param, 'delta_dist': delta_dist,
                              'patient_BW': patient_BW,
                              'epsilon': epsilon, 'cf_len': cf_len,
                              'generate_train_trace_num': generate_train_trace_num,
                              'total_test_trace_num': total_test_trace_num,
                              # 'test_interval': test_interval,
                              'gradient_steps': gradient_steps,
                              # 'total_train_steps': total_train_steps,
                              'log_interval': log_interval, 'dist_func': dist_func, 'batch_size': batch_size,
                              'lambda_start_value': lambda_start_value, 'learning_rate': learning_rate,
                              'train_round': r,
                              'ENV_NAME': env_name,
                              }
            all_cf_trace_training_time_this_seg,all_train_result_this_seg = run_experiment_ddpg_sb3_1_trace_1_patient(model_CF, trace_eps,
                                                                    current_time_step, fixed_trace_this_eps_df, parameter_dict,
                                                                    env_name, this_train_round=r, mode='train')
            if all_cf_trace_training_time_this_seg is not None:
                all_train_cf_trace_list.append(all_cf_trace_training_time_this_seg)
                all_train_result_list.append(all_train_result_this_seg)
                all_train_cf_trace_df = pd.concat(all_train_cf_trace_list)
                all_train_result_df = pd.concat(all_train_result_list)
                all_train_cf_trace_df.to_csv('{save_folder}/all_cf_traces_train.csv'.format(save_folder=save_folder))
                all_train_result_df.to_csv('{save_folder}/all_train_result.csv'.format(save_folder=save_folder))
            counter += 1
            all_test_cf_trace_df_this_round, all_test_result_df_this_round = test_ddpg_sb3_all_1_trace_1_patient(model_CF, test_index_df,
                                                                                               save_folder, trace_df_all_patient_dict, this_train_round=r)
            test_trace_df_list.append(all_test_cf_trace_df_this_round)
            test_result_df_list.append(all_test_result_df_this_round)
        all_test_cf_trace_df = pd.concat(test_trace_df_list)
        all_test_result_df = pd.concat(test_result_df_list)
        all_test_cf_trace_df.to_csv('{save_folder}/all_cf_traces_test.csv'.format(save_folder=save_folder))
        all_test_result_df.to_csv('{save_folder}/all_test_result.csv'.format(save_folder=save_folder))

    return

def test_ddpg_sb3_all_1_trace_1_patient(model_CF, all_test_index_df, save_folder, trace_df_all_patient_dict, this_train_round):
    # for server ver
    patient_info_input_dim = 13
    patient_info_output_dim = args_cf_diabetic.arg_patidim
    with_encoder = args_cf_diabetic.arg_with_encoder
    iob_param = args_cf_diabetic.arg_iob_param
    delta_dist = args_cf_diabetic.arg_delta_dist
    patient_BW = 0#this_patient_BW
    epsilon = args_cf_diabetic.arg_epsilon
    cf_len = args_cf_diabetic.arg_cflen
    generate_train_trace_num = args_cf_diabetic.arg_generate_train_trace_num  # how many cf traces generated and pyshed to buffer each time conduct collect_rollout
    total_test_trace_num = args_cf_diabetic.arg_total_test_trace_num  # how many cf traces generated for generating test result
    #test_interval = args_cf_diabetic.arg_test_interval
    gradient_steps = args_cf_diabetic.arg_gradient_steps
    #total_train_steps = args_cf_diabetic.arg_total_train_steps
    log_interval = args_cf_diabetic.arg_log_interval
    dist_func = args_cf_diabetic.arg_dist_func
    batch_size = args_cf_diabetic.arg_batch_size
    lambda_start_value = args_cf_diabetic.arg_lambda_start_value
    learning_rate = args_cf_diabetic.arg_learning_rate
    #train_round = args_cf_diabetic.arg_train_round  # total train round for DDPG
    #ENV_NAME = args_cf_diabetic.arg_ENV_NAME
    #total_timesteps_each_trace = args_cf_diabetic.arg_total_timesteps_each_trace

    # for saving training results
    all_test_cf_trace_list = []
    all_test_result_list = []

    #env_CF_test = NormalizeObservation(gym.make(ENV_NAME))

    for item, row in all_test_index_df.iterrows():
        env_name = row['ENV_NAME']
        trace_eps = row['orig_episode']
        current_time_step = row['orig_end_time_index']
        patient_type = row['patient_type']
        patient_id = row['patient_id']
        print('Test: ', env_name, trace_eps, current_time_step)
        parameter_dict = {'patient_type':patient_type, 'patient_id':patient_id, 'patient_info_input_dim': patient_info_input_dim,
                              'patient_info_output_dim': patient_info_output_dim,
                              'with_encoder': with_encoder, 'iob_param': iob_param, 'delta_dist': delta_dist,
                              'patient_BW': patient_BW,
                              'epsilon': epsilon, 'cf_len': cf_len,
                              'generate_train_trace_num': generate_train_trace_num,
                              'total_test_trace_num': total_test_trace_num,
                              #'test_interval': test_interval,
                              'gradient_steps': gradient_steps,
                              #'total_train_steps': total_train_steps,
                              'log_interval': log_interval, 'dist_func': dist_func, 'batch_size': batch_size,
                              'lambda_start_value': lambda_start_value, 'learning_rate': learning_rate,
                              'train_round': this_train_round,
                                #'ENV_NAME': ENV_NAME,
                                }
        # fixed_trace_this_eps_df = trace_df[trace_df["episode"] == trace_eps]
        fixed_orig_past_trace_df = trace_df_all_patient_dict[env_name]
        fixed_trace_this_eps_df = fixed_orig_past_trace_df[fixed_orig_past_trace_df['episode'] == trace_eps]
        all_cf_trace_test_this_seg, all_test_result_this_seg = run_experiment_ddpg_sb3_1_trace_1_patient(model_CF, trace_eps,
                                                                        current_time_step, fixed_trace_this_eps_df, parameter_dict,env_name,
                                                                            this_train_round=this_train_round,mode='test')
        if all_cf_trace_test_this_seg is not None:
            all_test_cf_trace_list.append(all_cf_trace_test_this_seg)
            all_test_result_list.append(all_test_result_this_seg)

    all_test_cf_trace_df = pd.concat(all_test_cf_trace_list)
    all_test_result_df = pd.concat(all_test_result_list)
    # all_test_cf_trace_df.to_csv('{save_folder}/all_cf_traces_test.csv'.format(save_folder=save_folder))
    # all_test_result_df.to_csv('{save_folder}/all_test_result.csv'.format(save_folder=save_folder))

    return all_test_cf_trace_df, all_test_result_df


all_train_index_df = train_index_df
all_test_index_df = test_index_df
counter = 0

# Hyperparameters
render = False
#trace_file_folder = '{folder}/save_file/save_trace'.format(folder=save_folder)
# Set Callback to save evaluation results
callback_freq=100
train_step = 100
lr = 0
# new best model is saved according to the mean_reward
#save_folder = 'D:/ShuyangDongDocument/UVA/UVA-Research/Project/Counterfactual_Explanation/code/diabetic_example/save_file'
best_result_folder="{save_folder}/logs/{noise_type}/{train_step}_{lr}".format(save_folder=save_folder,
                                                            noise_type=noise_type,train_step=train_step,lr=lr)

# Save a checkpoint every k steps
checkpoint_callback = CheckpointCallback(
  save_freq=callback_freq,
  save_path="{save_folder}/logs/{noise_type}/{train_step}_{lr}/checkpoint".format(save_folder=save_folder,
                                                                                            noise_type=noise_type,
                                                                                 train_step=train_step,lr=lr),
  name_prefix="ddpg_cf_diabetic_model_{noise_type}".format(noise_type=noise_type),
  save_replay_buffer=False,
  save_vecnormalize=True,
)
# Create the callback list
callback = None #CallbackList([checkpoint_callback])
if args_cf_diabetic.arg_train_one_trace==0:# train with multiple trace for each model, generalization
    # # run baseline PPO
    # mem_before_baseline = get_mem_use()
    # for item, row in all_test_index_df.iterrows():
    # #for orig_trace in train_index:
    #     # parameters
    #     # for key, value in orig_trace.items():
    #     #     trace_eps = key#row['orig_episode']
    #     #     current_time_step = value#row['orig_end_time_index']
    #     env_name = row['ENV_NAME']
    #     trace_eps = row['orig_episode']
    #     current_time_step = row['orig_end_time_index']
    #     patient_type = row['patient_type']
    #     patient_id = row['patient_id']
    #     CF_start_step = current_time_step - args_cf_diabetic.arg_cflen + 1
    #     #print(trace_eps, current_time_step)
    #     orig_trace_episode = trace_eps
    #
    #     delta_dist = args_cf_diabetic.arg_delta_dist
    #     patient_BW = 0
    #     fixed_trace_this_eps_df = trace_df_all_patient_dict[env_name]
    #     fixed_trace_this_eps_df = fixed_trace_this_eps_df[fixed_trace_this_eps_df['episode'] == trace_eps]
    #     baseline_path = trained_controller_dict[env_name]
    #     # run baseline test
    #     #print('Run baseline PPO.')
    #     #test_index = {trace_eps: current_time_step}
    #     test_index = row
    #     #time_1 = time.perf_counter()
    #     test_baseline(rl_model=None, patient_BW=patient_BW, delta_dist=delta_dist, test_index=test_index, train_counter=train_counter,
    #                     fixed_trace_this_eps_df=fixed_trace_this_eps_df, mode='test_baseline', baseline_path=baseline_path)
    #     #time_2 = time.perf_counter()
    #     #print('Time for test 1 trace by Baseline: ', time_2 - time_1)

    # mem_after_baseline = get_mem_use()
    # print('Mem usage, Run all Baseline: ', mem_after_baseline - mem_before_baseline)
    time_3 = time.perf_counter()
    mem_before_1_exp = get_mem_use()
    ###
    # train DDPG_CF and test
    for trial in range(1, args_cf_diabetic.arg_total_trial_num + 1):
        print('Start trial: ', trial)
        ddpg_cf_results_folder = '{save_folder}/td3_cf_results/trial_{t}'.format(save_folder=save_folder, t=trial)
        mkdir(ddpg_cf_results_folder)
        ENV_NAME_dummy = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type='adult', patient_id='7')
        for name in ENV_ID_list:
            all_env_dict[name] = NormalizeObservation(gym.make(name))
        train_ddpg_sb3_all(all_train_index_df, all_test_index_df, ENV_NAME_dummy, ddpg_cf_results_folder, callback,
                           trained_controller_dict, all_env_dict, trace_df_all_patient_dict,ENV_ID_list,
                           baseline_result_folder='{save_folder}/baseline_results'.format(save_folder=save_folder),
                           trial_index=trial)
        time_4 = time.perf_counter()
        print('Time for 1 Full Exp by DDPG: ', time_4 - time_3)
        mem_after_exp = get_mem_use()
        print('Mem for 1 Full Exp by DDPG: ', mem_after_exp - mem_before_1_exp)
        gc.collect()
    time_5 = time.perf_counter()
    print('Time for all trials by DDPG: ', time_5 - time_3)
    mem_after_exp = get_mem_use()
    print('Mem for all trial by DDPG: ', mem_after_exp - mem_before_1_exp)
else: # only train for 1 trace for each model, no generalization
    # run baseline PPO
    for item, row in all_train_index_df.iterrows():
        # for orig_trace in train_index:
        # parameters
        # for key, value in orig_trace.items():
        #     trace_eps = key#row['orig_episode']
        #     current_time_step = value#row['orig_end_time_index']
        env_name = row['ENV_NAME']
        trace_eps = row['orig_episode']
        current_time_step = row['orig_end_time_index']
        patient_type = row['patient_type']
        patient_id = row['patient_id']
        CF_start_step = current_time_step - args_cf_diabetic.arg_cflen + 1
        # print(trace_eps, current_time_step)
        orig_trace_episode = trace_eps

        delta_dist = args_cf_diabetic.arg_delta_dist
        patient_BW = 0
        fixed_trace_this_eps_df = trace_df_all_patient_dict[env_name]
        fixed_trace_this_eps_df = fixed_trace_this_eps_df[fixed_trace_this_eps_df['episode'] == trace_eps]
        baseline_path = trained_controller_dict[env_name]
        # run baseline test
        # print('Run baseline PPO.')
        # test_index = {trace_eps: current_time_step}
        test_index = row
        test_baseline(rl_model=None, patient_BW=patient_BW, delta_dist=delta_dist, test_index=test_index,
                      train_counter=train_counter,
                      fixed_trace_this_eps_df=fixed_trace_this_eps_df, mode='test_baseline',
                      baseline_path=baseline_path)

    # train DDPG_CF and test
    ddpg_cf_results_folder = '{save_folder}/td3_cf_results'.format(save_folder=save_folder)
    mkdir(ddpg_cf_results_folder)
    ENV_NAME_dummy = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type='adult', patient_id='5')
    for name in ENV_ID_list:
        all_env_dict[name] = NormalizeObservation(gym.make(name))

    train_ddpg_sb3_all_1_trace_1_patient(all_train_index_df, ENV_NAME_dummy, ddpg_cf_results_folder, callback,
                                         trained_controller_dict, all_env_dict, trace_df_all_patient_dict,
                                         baseline_result_folder='{save_folder}/baseline_results'.format(save_folder=save_folder))



