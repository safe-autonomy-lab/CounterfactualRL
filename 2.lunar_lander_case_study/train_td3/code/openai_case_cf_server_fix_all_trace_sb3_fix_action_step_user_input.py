#!/usr/bin/env python
# coding: utf-8

# import package
import faulthandler
faulthandler.enable()
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MAIN_LOG_LEVEL'] = '2'
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

from typing import Optional
from datetime import datetime

from gym.envs.registration import register
#from gym.wrappers.normalize_cf import NormalizeObservation

pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)
import sys
sys.path.append("/.conda/envs/CF_lunarlander/lib/python3.8/site-packages/gym")

# modified the core.py in gym==0.21.0 to make it work
import gym
from gym.envs.registration import register
#from gym.wrappers.normalize_cf import NormalizeObservation
from gym import cf_generator
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from random import sample
from collections import namedtuple
import argparse
from memory_profiler import profile
#Observation = namedtuple('Observation', ['CGM', 'CHO', 'ROC','insulin', 'BG'])

# modified the core.py in gym==0.21.0 to make it work

def get_mem_use():
    mem=psutil.virtual_memory()
    mem_gb = mem.used/(1024*1024*1024)
    return mem_gb
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('New folder ok.')
    else:
        #print('There is this folder')
        pass

    return
# wrapper used
class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=350):
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

def store_time_index(time_index_list, save_file_path):
    # save {eps_idx: time_index} of training and testing
    df = pd.DataFrame(columns=['orig_episode', 'orig_end_time_index'])
    orig_episode_list = []
    orig_time_index_list = []
    for info_dict in time_index_list:
        for k, v in info_dict.items():
            trace_eps, current_time_step = k, v
        orig_episode_list.append(trace_eps)
        orig_time_index_list.append(current_time_step)
    df['orig_episode'] = orig_episode_list
    df['orig_end_time_index'] = orig_time_index_list
    df.to_csv(save_file_path)
    return

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


##################-LUNAR LANDER-################
# get the params of a b2Body
def get_b2Body_param(b2Body_obj, obj_type='lander'):
    dict = {'active': str(b2Body_obj.active),
            'angle':float(str(b2Body_obj.angle)),
            'angularDamping':float(str(b2Body_obj.angularDamping)),
            'angularVelocity':float(str(b2Body_obj.angularVelocity)),
            'awake':str(b2Body_obj.awake),
            'bullet':str(b2Body_obj.bullet),
            'contacts':str(b2Body_obj.contacts),
            'fixedRotation':str(b2Body_obj.fixedRotation), #'fixtures':str(b2Body_obj.fixtures),
            'fixtures_vertices': [x for x in b2Body_obj.fixtures[0].shape.vertices],
            'fixtures_density': float(str(b2Body_obj.fixtures[0].density)),
            'fixtures_friction': float(str(b2Body_obj.fixtures[0].friction)),
            'fixtures_restitution': float(str(b2Body_obj.fixtures[0].restitution)),
            'inertia':float(str(b2Body_obj.inertia)), 'joints':str(b2Body_obj.joints),
            'linearDamping':float(str(b2Body_obj.linearDamping)),
            #'linearVelocity':[float(str(b2Body_obj.linearVelocity.x)), float(str(b2Body_obj.linearVelocity.y))],
            'linearVelocity': [b2Body_obj.linearVelocity.x, b2Body_obj.linearVelocity.y],
            'localCenter':[float(str(b2Body_obj.localCenter.x)), float(str(b2Body_obj.localCenter.y))],
            'mass':float(str(b2Body_obj.mass)),
            'massData':{'I': float(str(b2Body_obj.massData.I)),
                        'center':[float(str(b2Body_obj.massData.center.x)),float(str(b2Body_obj.massData.center.y))],
                                   'mass':float(str(b2Body_obj.massData.mass))},
            'position':[float(str(b2Body_obj.position.x)), float(str(b2Body_obj.position.y))],
            'sleepingAllowed':str(b2Body_obj.sleepingAllowed), 'transform':str(b2Body_obj.transform),
            'type':b2Body_obj.type, 'userData':b2Body_obj.userData,
            'worldCenter':[float(str(b2Body_obj.worldCenter.x)), float(str(b2Body_obj.worldCenter.y))]}
    if obj_type=='lander':
        pass
    elif obj_type=='leg':
        dict['ground_contact'] = str(b2Body_obj.ground_contact)
    return dict

# get the params of a b2RevoluteJoint
def get_b2RevoluteJoint_param(b2RevoluteJoint_obj):
    '''#
    'b2RevoluteJoint': ['active', 'anchorA', 'anchorB', 'angle', 'bodyA',
     'bodyB', 'limitEnabled', 'limits', 'lowerLimit',
     'maxMotorTorque', 'motorEnabled', 'motorSpeed', 'speed',
     'type', 'upperLimit', 'userData']
     'b2RevoluteJointDef': ['anchor', 'bodyA', 'bodyB', 'collideConnected', 'enableLimit',
                                      'enableMotor', 'localAnchorA', 'localAnchorB', 'lowerAngle',
                                      'maxMotorTorque', 'motorSpeed', 'referenceAngle', 'type',
                                      'upperAngle', 'userData', ],
    #'''
    dict = {'active': str(b2RevoluteJoint_obj.active), 'angle': float(str(b2RevoluteJoint_obj.angle)),
            'anchorA': [float(str(b2RevoluteJoint_obj.anchorA.x)),float(str(b2RevoluteJoint_obj.anchorA.y))],
            'anchorB': [float(str(b2RevoluteJoint_obj.anchorB.x)),float(str(b2RevoluteJoint_obj.anchorB.y))],
            'bodyA': str(b2RevoluteJoint_obj.bodyA), 'bodyB': str(b2RevoluteJoint_obj.bodyB),
            'limitEnabled': str(b2RevoluteJoint_obj.limitEnabled),
            'limits': [float(str(b2RevoluteJoint_obj.limits[0])),float(str(b2RevoluteJoint_obj.limits[1]))],
            'lowerLimit': float(str(b2RevoluteJoint_obj.lowerLimit)),
            'maxMotorTorque': 40,
            'motorEnabled': str(b2RevoluteJoint_obj.motorEnabled),
            'motorSpeed': float(str(b2RevoluteJoint_obj.motorSpeed)),
            'speed': float(str(b2RevoluteJoint_obj.speed)),
            'type': b2RevoluteJoint_obj.type, 'upperLimit': float(str(b2RevoluteJoint_obj.upperLimit)), 'userData': b2RevoluteJoint_obj.userData}

    return dict

def get_rocket_trace(env, trace_df, step, current_episode, current_episode_step, current_action, current_states,
                      current_obs,
                      current_obs_new, current_reward, current_accumulated_reward,
                      current_done,gravity):
    # store the state info of patient, sensor, scenario for reseting while generating cf
    # Get CF traces and save in buffer
    # store past steps
    current_moon = env.moon
    current_sky_polys = env.sky_polys
    current_lander = env.lander
    current_legs = env.legs
    current_terrain_hight = env.terrain_hight
    '''#######################
    current_lander_position = [str(current_lander.position.x), str(current_lander.position.y)]
    current_lander_angle = str(current_lander.angle)
    current_lander_angularDamping = str(current_lander.angularDamping)
    current_lander_angularVelocity = str(current_lander.angularVelocity)
    current_lander_awake = str(current_lander.awake)
    current_lander_bullet = str(current_lander.bullet)
    current_lander_contacts = str(current_lander.contacts)
    current_lander_fixedRotation = str(current_lander.fixedRotation)
    current_lander_sleepingAllowed = str(current_lander.sleepingAllowed)
    current_lander_type = str(current_lander.type)
    current_lander_userData = str(current_lander.userData)
    current_lander_worldCenter = [str(current_lander.worldCenter.x),str(current_lander.worldCenter.y)]
    current_lander_fixtures_vertices = [str(x) for x in current_lander.fixtures[0].shape.vertices] #str(current_lander.fixtures[0].shape.vertices)
    current_lander_fixtures_density = str(current_lander.fixtures[0].density)
    current_lander_fixtures_friction = str(current_lander.fixtures[0].friction)
    current_lander_fixtures_restitution = str(current_lander.fixtures[0].restitution)
    current_lander_inertia = str(current_lander.inertia)
    current_lander_joints = current_lander.joints[0]
    current_lander_linearDamping = str(current_lander.linearDamping)
    current_lander_linearVelocity = [str(current_lander.linearVelocity.x), str(current_lander.linearVelocity.y)]
    current_lander_localCenter = [str(current_lander.localCenter.x), str(current_lander.localCenter.y)]
    current_lander_mass = str(current_lander.mass)
    current_lander_massData = {'I': str(current_lander.massData.I), 'center':[str(current_lander.massData.center.x),str(current_lander.massData.center.y)],
                               'mass':str(current_lander.massData.mass)} #str(current_lander.massData)

    print('current_lander_position: ', current_lander_position)
    print('current_lander_angle: ', current_lander_angle)
    print('current_lander_angularDamping: ', current_lander_angularDamping)
    print('current_lander_angularVelocity: ', current_lander_angularVelocity)
    print('current_lander_contacts: ', current_lander_contacts)
    print('current_lander_fixedRotation: ', current_lander_fixedRotation)
    print('current_lander_sleepingAllowed: ', current_lander_sleepingAllowed)
    print('current_lander_type: ', current_lander_type)
    print('current_lander_userData: ', current_lander_userData)
    print('current_lander_worldCenter: ', current_lander_worldCenter, current_lander.worldCenter.x, current_lander.worldCenter.y)
    print('current_lander_fixtures_vertices: ', current_lander_fixtures_vertices)
    print('current_lander_fixtures_density: ', current_lander_fixtures_density)
    print('current_lander_fixtures_friction: ', current_lander_fixtures_friction)
    print('current_lander_fixtures_restitution: ', current_lander_fixtures_restitution)
    print('current_lander_inertia: ', current_lander_inertia)
    print('current_lander_joints.joint: ', current_lander_joints.joint, type(current_lander_joints.joint))
    print('current_lander_joints.other: ', current_lander_joints.other, type(current_lander_joints.other))
    print('current_lander_linearDamping: ', current_lander_linearDamping)
    print('current_lander_linearVelocity: ', current_lander_linearVelocity)
    print('current_lander_localCenter: ', current_lander_localCenter)
    print('current_lander_mass: ', current_lander_mass)
    print('current_lander_massData: ', current_lander_massData)

    current_lander_feature_dict = {'active':env.lander.active, 'position':env.lander.position, 'angle':env.lander.angle,
                                   'angularDamping':env.lander.angularDamping, 'angularVelocity':env.lander.angularVelocity,
                                   'awake':env.lander.awake, 'bullet':env.lander.bullet, 'contacts':env.lander.contacts,
                                   'fixedRotation':env.lander.fixedRotation, 'sleepingAllowed':env.lander.sleepingAllowed,
                                   'type':env.lander.type, 'userData':env.lander.userData,
                                   'worldCenter':env.lander.worldCenter, 'fixtures':env.lander.fixtures,
                                   'fixtures_vertices': env.lander.fixtures[0].shape.vertices,'fixtures_density':env.lander.fixtures[0].density,
                                   'fixtures_friction':env.lander.fixtures[0].friction,'fixtures_restitution':env.lander.fixtures[0].restitution,
                                   'inertia':env.lander.inertia,
                                   'joints': env.lander.joints, 'linearDamping':env.lander.linearDamping, 'linearVelocity':env.lander.linearVelocity,
                                   'localCenter': env.lander.localCenter, 'mass':env.lander.mass, 'massData':env.lander.massData}
    current_leg_feature_dict_0 = {'active': env.legs[0].active, 'position': env.legs[0].position,
                                   'angle': env.legs[0].angle,
                                   'angularDamping': env.legs[0].angularDamping,
                                   'angularVelocity': env.legs[0].angularVelocity,
                                   'awake': env.legs[0].awake, 'bullet': env.legs[0].bullet,
                                   'contacts': env.legs[0].contacts,
                                   'fixedRotation': env.legs[0].fixedRotation,
                                   'sleepingAllowed': env.legs[0].sleepingAllowed,
                                   'type': env.legs[0].type, 'userData': env.legs[0].userData,
                                   'worldCenter': env.legs[0].worldCenter, 'fixtures': env.legs[0].fixtures,
                                  'ground_contact':env.legs[0].ground_contact,
                                  'inertia':env.legs[0].inertia,'joints': env.legs[0].joints, 'linearDamping':env.legs[0].linearDamping,
                                  'linearVelocity':env.legs[0].linearVelocity,
                                   'localCenter': env.legs[0].joints, 'mass':env.legs[0].mass, 'massData':env.legs[0].massData}
    current_leg_feature_dict_1 = {'active': env.legs[1].active, 'position': env.legs[1].position,
                                  'angle': env.legs[1].angle,
                                  'angularDamping': env.legs[1].angularDamping,
                                  'angularVelocity': env.legs[1].angularVelocity,
                                  'awake': env.legs[1].awake, 'bullet': env.legs[1].bullet,
                                  'contacts': env.legs[1].contacts,
                                  'fixedRotation': env.legs[1].fixedRotation,
                                  'sleepingAllowed': env.legs[1].sleepingAllowed,
                                  'type': env.legs[1].type, 'userData': env.legs[1].userData,
                                  'worldCenter': env.legs[1].worldCenter, 'fixtures': env.legs[1].fixtures,
                                  'ground_contact':env.legs[1].ground_contact,
                                  'inertia':env.legs[1].inertia,'joints': env.legs[1].joints, 'linearDamping':env.legs[1].linearDamping,
                                  'linearVelocity':env.legs[1].linearVelocity,
                                   'localCenter': env.legs[1].joints, 'mass':env.legs[1].mass, 'massData':env.legs[1].massData}
    #######################'''

    terrain_param_dict = {'terrain_hight':current_terrain_hight,'helipad_x1':env.helipad_x1,
                          'helipad_x2':env.helipad_x2, 'helipad_y':env.helipad_y}
    current_lander_feature_dict = get_b2Body_param(current_lander, 'lander')
    current_leg_feature_dict_0 = get_b2Body_param(current_legs[0], 'leg')
    current_leg_feature_dict_1 = get_b2Body_param(current_legs[1], 'leg')
    current_joint_feature_dict_0 = get_b2RevoluteJoint_param(current_lander.joints[0].joint)
    current_joint_feature_dict_1 = get_b2RevoluteJoint_param(current_lander.joints[1].joint)

    current_legs_dict_list = [current_leg_feature_dict_0, current_leg_feature_dict_1]
    current_joints_dict_list = [current_joint_feature_dict_0, current_joint_feature_dict_1]
    trace_dict = {'gravity':gravity,'step': step, 'episode': current_episode, 'episode_step': current_episode_step, 'action': current_action.tolist(),
                  'observation': list(current_obs),
                  'observation_new': list(current_obs_new),
                  'reward': current_reward,
                  'episode_return': current_accumulated_reward, 'done': current_done,
                  'moon': current_moon, 'sky_polys': current_sky_polys,
                  'lander': current_lander_feature_dict, 'legs': current_legs_dict_list,
                  'joints': current_joints_dict_list, #'terrain':terrain_param_dict,
                  'terrain_hight': list(current_terrain_hight),
                  'states':current_states}
    trace_dict = save_b2body_data(trace_dict)
    # print('current_moon: ', current_moon)
    # print('current_sky_polys: ', current_sky_polys)
    # print('current_lander_feature_dict: ', current_lander_feature_dict)
    # print('current_legs_dict_list: ', current_legs_dict_list[0], current_legs_dict_list[1])
    # print('current_joints_dict_list: ', current_joints_dict_list[0], current_joints_dict_list[1])
    # print('terrain_param_dict: ', terrain_param_dict)
    # print('current_states: ', current_states)
    trace_df = trace_df.append(trace_dict, ignore_index=True)
    return trace_df

def save_b2body_data(trace_dict):
    lander_dict = trace_dict['lander']
    current_legs_dict_0 = trace_dict['legs'][0]
    current_legs_dict_1 = trace_dict['legs'][1]
    current_joint_dict_0 = trace_dict['joints'][0]
    current_joint_dict_1 = trace_dict['joints'][1]
    # lander
    for key, value in lander_dict.items():
        if key not in ['contacts', 'joints', 'localCenter', 'transform', 'worldCenter']:
            if (key != 'massData') and (key != 'fixtures_vertices'):
                new_key = 'lander_{key}'.format(key=key)
                trace_dict[new_key] = lander_dict[key]
            elif key == 'massData':
                for k, v in lander_dict[key].items():
                    new_key = 'lander_massData_{k}'.format(k=k)
                    trace_dict[new_key] = lander_dict[key][k]
            elif key == 'fixtures_vertices':
                new_key = 'lander_{key}'.format(key=key)
                trace_dict[new_key] = lander_dict[key]
                veticle_list = lander_dict['fixtures_vertices']
                for i in range(0, len(veticle_list)):
                    new_key = 'lander_fixtures_vertices_{i}'.format(i=i)
                    trace_dict[new_key] = veticle_list[i]

    for key, value in current_legs_dict_0.items():
        if key not in ['contacts', 'joints', 'localCenter', 'transform', 'worldCenter']:
            # new_key = 'leg_0_{key}'.format(key=key)
            # trace_dict[new_key] = current_legs_dict_0[key]
            if (key != 'massData') and (key != 'fixtures_vertices'):
                new_key = 'leg_0_{key}'.format(key=key)
                trace_dict[new_key] = current_legs_dict_0[key]
            elif key == 'massData':
                for k, v in current_legs_dict_0[key].items():
                    new_key = 'leg_0_massData_{k}'.format(k=k)
                    trace_dict[new_key] = current_legs_dict_0[key][k]
            elif key == 'fixtures_vertices':
                new_key = 'leg_0_{key}'.format(key=key)
                trace_dict[new_key] = current_legs_dict_0[key]
                veticle_list = current_legs_dict_0['fixtures_vertices']
                for i in range(0, len(veticle_list)):
                    new_key = 'leg_0_fixtures_vertices_{i}'.format(i=i)
                    trace_dict[new_key] = veticle_list[i]
    for key, value in current_legs_dict_1.items():
        if key not in ['contacts', 'joints', 'localCenter', 'transform', 'worldCenter']:
            # new_key = 'leg_1_{key}'.format(key=key)
            # trace_dict[new_key] = current_legs_dict_1[key]
            if (key != 'massData') and (key != 'fixtures_vertices'):
                new_key = 'leg_1_{key}'.format(key=key)
                trace_dict[new_key] = current_legs_dict_1[key]
            elif key == 'massData':
                for k, v in current_legs_dict_1[key].items():
                    new_key = 'leg_1_massData_{k}'.format(k=k)
                    trace_dict[new_key] = current_legs_dict_1[key][k]
            elif key == 'fixtures_vertices':
                new_key = 'leg_1_{key}'.format(key=key)
                trace_dict[new_key] = current_legs_dict_1[key]
                veticle_list = current_legs_dict_1['fixtures_vertices']
                for i in range(0, len(veticle_list)):
                    new_key = 'leg_1_fixtures_vertices_{i}'.format(i=i)
                    trace_dict[new_key] = veticle_list[i]
    for key, value in current_joint_dict_0.items():
        if key in ['limitEnabled', 'limits', 'lowerLimit', 'maxMotorTorque', 'motorEnabled','motorSpeed']:
            new_key = 'joint_0_{key}'.format(key=key)
            trace_dict[new_key] = current_joint_dict_0[key]
    for key, value in current_joint_dict_1.items():
        if key in ['limitEnabled', 'limits', 'lowerLimit', 'maxMotorTorque', 'motorEnabled','motorSpeed']:
            new_key = 'joint_1_{key}'.format(key=key)
            trace_dict[new_key] = current_joint_dict_1[key]

    return trace_dict

def convert_str_to_list(str, split_type=","):
    # lander_fixtures_vertices, lander_linearVelocity, lander_mass_center, lander_position,
    # leg_0_fixtures_vertices, leg_0_linearVelocity, leg_0_mass_center, leg_0_position
    # leg_1_fixtures_vertices, leg_1_linearVelocity, leg_1_mass_center, leg_1_position
    # joint_0_limits, joint_1_limits,
    str_1 = str[1:-1]
    str_list = [float(x) for x in str_1.split(split_type)]
    #print('Before convert str: ', str_1, type(str_1))
    #print('After convert str: ', str_list, type(str_list))
    return str_list
def convert_to_b2body_data_dict(b2_body_info_name, df_row, b2body_type='lander'):
    # get lander data, leg data
    b2body_info_dict = {}
    up_n = 6 if b2body_type=='lander' else 3
    for info_name in b2_body_info_name:
        b2body_col_name = '{b2body_type}_{info_name}'.format(b2body_type=b2body_type, info_name=info_name)
        if info_name not in ['massData_I', 'massData_center','massData_mass']:
            if info_name in ['linearVelocity', 'position']:
                converted = convert_str_to_list(df_row[b2body_col_name])
                b2body_info_dict[info_name] = converted
                # print(b2body_col_name, df_row[b2body_col_name])
                # print(info_name, converted)
            elif info_name in ['fixtures_vertices']:
                fixtures_vertices_list = []
                for i in range(0,up_n):
                    fixtures_vertice_tuple_str = df_row['{b2body_col_name}_{i}'.format(b2body_col_name=b2body_col_name,i=i)]
                    fixtures_vertices_list.append(fixtures_vertice_tuple_str)
                b2body_info_dict[info_name] = fixtures_vertices_list
                # print(b2body_col_name, df_row[b2body_col_name])
                # print(info_name, fixtures_vertices_list)
            elif info_name in ['active', 'awake', 'bullet', 'fixedRotation', 'sleepingAllowed', 'ground_contact']:
                b2body_info_dict[info_name] = str(df_row[b2body_col_name])
                # print(info_name, b2body_col_name, df_row[b2body_col_name], b2body_info_dict[info_name], type(b2body_info_dict[info_name]))
            else:
                b2body_info_dict[info_name] = df_row[b2body_col_name]
                # print(info_name, b2body_col_name, df_row[b2body_col_name])
    mass_dict = {'I': df_row['{b2body_type}_massData_I'.format(b2body_type=b2body_type)],
                         'center': convert_str_to_list(df_row['{b2body_type}_massData_center'.format(b2body_type=b2body_type)]),
                         'mass': df_row['{b2body_type}_massData_mass'.format(b2body_type=b2body_type)]}
    b2body_info_dict['massData'] = mass_dict
    # print('mass_dict: ', mass_dict)
    # print(b2body_type, b2body_info_dict)
    return b2body_info_dict

def convert_to_b2joint_data_dict(b2_joint_info_name, df_row, b2joint_type='joint_0'):
    # get joint data
    b2joint_info_dict = {}
    for info_name in b2_joint_info_name:
        joint_col_name = '{b2joint_type}_{info_name}'.format(info_name=info_name,b2joint_type=b2joint_type)
        if info_name in ['limits']:
            converted = convert_str_to_list(df_row[joint_col_name])
            b2joint_info_dict[info_name] = converted
            # print(joint_col_name, df_row[joint_col_name], type(df_row[joint_col_name]))
            # print(info_name, converted, type(converted))
        elif info_name in ['limitEnabled', 'motorEnabled']:
            b2joint_info_dict[info_name] = str(df_row[joint_col_name])
            # print(info_name, joint_col_name, df_row[joint_col_name], b2joint_info_dict[info_name], type(b2joint_info_dict[info_name]))
        else:
            b2joint_info_dict[info_name] = df_row[joint_col_name]
            # print(info_name, joint_col_name, df_row[joint_col_name])
        b2joint_info_dict[info_name] = df_row[joint_col_name]
    # print(b2joint_type ,b2joint_info_dict)
    return b2joint_info_dict

def get_rocket_trace_from_saved_data(trace_df, save_path,orig_id,g):
    # dict = {'active', 'angle',
    #         'angularDamping',
    #         'angularVelocity', 'awake',
    #         'bullet', 'contacts',
    #         'fixedRotation',  # 'fixtures':str(b2Body_obj.fixtures),
    #         'fixtures_vertices',
    #         'fixtures_density',
    #         'fixtures_friction',
    #         'fixtures_restitution',
    #         'inertia', 'joints',
    #         'linearDamping',
    #         'linearVelocity',
    #         'localCenter',
    #         'mass',
    #         'massData',
    #         'position',
    #         'sleepingAllowed', 'transform',
    #         'type', 'userData',
    #         'worldCenter'}
    # dict = {'active': str(b2Body_obj.active), 'angle': str(b2Body_obj.angle),
    #         'angularDamping': str(b2Body_obj.angularDamping),
    #         'angularVelocity': str(b2Body_obj.angularVelocity),
    #         'awake': str(b2Body_obj.awake),
    #         'bullet': str(b2Body_obj.bullet),
    #         #'contacts': str(b2Body_obj.contacts),
    #         'fixedRotation': str(b2Body_obj.fixedRotation),  # 'fixtures':str(b2Body_obj.fixtures),
    #         'fixtures_vertices': [str(x) for x in b2Body_obj.fixtures[0].shape.vertices],
    #         'fixtures_density': str(b2Body_obj.fixtures[0].density),
    #         'fixtures_friction': str(b2Body_obj.fixtures[0].friction),
    #         'fixtures_restitution': str(b2Body_obj.fixtures[0].restitution),
    #         'inertia': str(b2Body_obj.inertia), #'joints': str(b2Body_obj.joints),
    #         'linearDamping': str(b2Body_obj.linearDamping),
    #         'linearVelocity': [str(b2Body_obj.linearVelocity.x), str(b2Body_obj.linearVelocity.y)],
    #         #'localCenter': [str(b2Body_obj.localCenter.x), str(b2Body_obj.localCenter.y)],
    #         'mass': str(b2Body_obj.mass),
    #         'massData': {'I': str(b2Body_obj.massData.I),
    #                      'center': [str(b2Body_obj.massData.center.x), str(b2Body_obj.massData.center.y)],
    #                      'mass': str(b2Body_obj.massData.mass)},
    #         'position': [str(b2Body_obj.position.x), str(b2Body_obj.position.y)],
    #         'sleepingAllowed': str(b2Body_obj.sleepingAllowed), #'transform': str(b2Body_obj.transform),
    #         'type': b2Body_obj.type, 'userData': b2Body_obj.userData,
    #         #'worldCenter': [str(b2Body_obj.worldCenter.x), str(b2Body_obj.worldCenter.y)]
    #         }
    # lander_info_name = ['contacts', 'joints', 'localCenter', 'transform', 'worldCenter']
    # leg_0_info_name = ['contacts', 'joints', 'localCenter', 'transform', 'worldCenter']
    # leg_1_info_name = ['contacts', 'joints', 'localCenter', 'transform', 'worldCenter']
    # joint_0_info_name = ['limitEnabled', 'limits', 'lowerLimit', 'maxMotorTorque', 'motorEnabled', 'motorSpeed']
    # joint_1_info_name = ['limitEnabled', 'limits', 'lowerLimit', 'maxMotorTorque', 'motorEnabled', 'motorSpeed']
    b2_body_info_name_lander = ['active', 'angle',
            'angularDamping',
            'angularVelocity', 'awake',
            'bullet',
            'fixedRotation',
            'fixtures_vertices',
            'fixtures_density',
            'fixtures_friction',
            'fixtures_restitution',
            'inertia',
            'linearDamping',
            'linearVelocity',
            'mass',
            #'massData',
            'position',
            'sleepingAllowed',
            'type', 'userData'
            ]
    b2_body_info_name_leg = ['active', 'angle',
                                'angularDamping',
                                'angularVelocity', 'awake',
                                'bullet',
                                'fixedRotation',
                                'fixtures_vertices',
                                'fixtures_density',
                                'fixtures_friction',
                                'fixtures_restitution',
                                'inertia',
                                'linearDamping',
                                'linearVelocity',
                                'mass',
                                # 'massData',
                                'position',
                                'sleepingAllowed',
                                'type', 'userData', 'ground_contact'
                                ]
    b2_joint_info_name = ['limitEnabled', 'limits', 'lowerLimit', 'maxMotorTorque', 'motorEnabled','motorSpeed']
    lander_data_list = []
    leg_data_list = []
    joint_data_list = []
    hight_list = []
    obs_list = []
    obs_new_list = []
    action_list = []
    for item, row in trace_df.iterrows():
        # get lander data
        # lander_info_dict = {}
        # for info_name in b2_body_info_name:
        #     lander_col_name = 'lander_{info_name}'.format(info_name=info_name)
        #     if info_name not in ['massData']:
        #         lander_info_dict[info_name] = row[lander_col_name]
        #     else:
        #         mass_dict = {'I': row['lander_I'],
        #                  'center': row['lander_center'],
        #                  'mass': row['lander_mass']}
        #         lander_info_dict[info_name] = mass_dict
        lander_info_dict = convert_to_b2body_data_dict(b2_body_info_name_lander, row, b2body_type='lander')
        # get leg data
        # leg_0_info_dict = {}
        # for info_name in b2_body_info_name:
        #     leg_0_col_name = 'leg_0_{info_name}'.format(info_name=info_name)
        #     if info_name not in ['massData']:
        #         leg_0_info_dict[info_name] = row[leg_0_col_name]
        #     else:
        #         mass_dict = {'I': row['leg_0_I'],
        #                      'center': row['leg_0_center'],
        #                      'mass': row['leg_0_mass']}
        #         leg_0_info_dict[info_name] = mass_dict
        # leg_1_info_dict = {}
        # for info_name in b2_body_info_name:
        #     leg_1_col_name = 'leg_1_{info_name}'.format(info_name=info_name)
        #     if info_name not in ['massData']:
        #         leg_1_info_dict[info_name] = row[leg_1_col_name]
        #     else:
        #         mass_dict = {'I': row['leg_1_I'],
        #                      'center': row['leg_1_center'],
        #                      'mass': row['leg_1_mass']}
        #         leg_1_info_dict[info_name] = mass_dict
        leg_0_info_dict = convert_to_b2body_data_dict(b2_body_info_name_leg, row, b2body_type='leg_0')
        leg_1_info_dict = convert_to_b2body_data_dict(b2_body_info_name_leg, row, b2body_type='leg_1')
        # # get joint data
        # joint_0_info_dict = {}
        # for info_name in b2_joint_info_name:
        #     joint_0_col_name = 'joint_0_{info_name}'.format(info_name=info_name)
        #     joint_0_info_dict[info_name] = row[joint_0_col_name]
        # joint_1_info_dict = {}
        # for info_name in b2_joint_info_name:
        #     joint_1_col_name = 'joint_1_{info_name}'.format(info_name=info_name)
        #     joint_1_info_dict[info_name] = row[joint_1_col_name]
        joint_0_info_dict = convert_to_b2joint_data_dict(b2_joint_info_name, row, b2joint_type='joint_0')
        joint_1_info_dict = convert_to_b2joint_data_dict(b2_joint_info_name, row, b2joint_type='joint_1')

        lander_data_list.append(lander_info_dict)
        current_legs_dict_list = [leg_0_info_dict, leg_1_info_dict]
        leg_data_list.append(current_legs_dict_list)
        current_joints_dict_list = [joint_0_info_dict, joint_1_info_dict]
        joint_data_list.append(current_joints_dict_list)

        # terrian hight
        #print('Convert terrain_hight list:')
        terrain_hight_str = row['terrain_hight']
        terrain_hight_list = convert_str_to_list(terrain_hight_str)
        hight_list.append(terrain_hight_list)
        # obs, obs_new, action
        obs_str = row['observation']
        obs_float = convert_str_to_list(obs_str)
        obs_list.append(obs_float)
        obs_new_str = row['observation_new']
        obs_new_float = convert_str_to_list(obs_new_str)
        obs_new_list.append(obs_new_float)
        action_str = row['action']
        action_float = convert_str_to_list(action_str)
        action_list.append(action_float)

    trace_df['lander'] = lander_data_list
    trace_df['legs'] = leg_data_list
    trace_df['joints'] = joint_data_list
    trace_df['terrain_hight'] = hight_list

    trace_df['observation'] = obs_list
    trace_df['observation_new'] = obs_new_list
    trace_df['action'] = action_list
    trace_df.to_csv('{save_path}/original_trace_convert_{orig_id}_g_{g}.csv'.format(save_path=save_path,orig_id=orig_id,g=g))
    return trace_df


# get parameters
parser = argparse.ArgumentParser(description='cf lunar lander train DDPG argparse')
parser.add_argument('--arg_id', '-arg_id', help='experiment id', default=0, type=int)
parser.add_argument('--arg_whole_trace_step', '-arg_wts', help='total length of the original traces', default=1000, type=int)
parser.add_argument('--arg_cuda', '-arg_cuda', help='cuda id', default=0, type=int)
parser.add_argument('--arg_cflen', '-arg_cflen', help='length of CF traces', default=20, type=int)
parser.add_argument('--arg_dist_func', '-arg_dist_func', help='assign distance loss function', default='dist_pairwise', type=str)
parser.add_argument('--arg_total_used_num', '-arg_total_used_num', help='total used number of traces for training and testing', default=5, type=int)
parser.add_argument('--arg_train_split', '-arg_train_split', help='split for training set', default=1.0, type=float)
# parser.add_argument('--arg_train_eps', '-arg_train_eps', help='training eps after sample a new original trace', default=10, type=int)
# parser.add_argument('--arg_test_eps', '-arg_test_eps', help='testing eps after sample a new original trace', default=10, type=int)
# parser.add_argument('--arg_update_iteration', '-arg_update_iteration', help='ddpg update iterration', default=10, type=int)
# parser.add_argument('--arg_actor_lr', '-arg_actor_lr', help='actor learning rate', default=0.0001, type=float)
# parser.add_argument('--arg_critic_lr', '-arg_critic_lr', help='critic learning rate', default=0.001, type=float)
# parser.add_argument('--arg_lambda_lr', '-arg_lambda_lr', help='lambda learning rate', default=0.001, type=float)
# parser.add_argument('--arg_patidim', '-arg_patidim', help='encoded patient state dim', default=3, type=int)
# parser.add_argument('--arg_improve_percentage_thre', '-arg_improve_percentage_thre', help='improve percentage threshold for better cf', default=0.1, type=float)
# parser.add_argument('--arg_train_sample_num', '-arg_train_sample_num', help='number of samples generated in training to push in buffer for each orig trace', default=150, type=int)
parser.add_argument('--arg_train_round', '-arg_train_round', help='train round for all training index', default=3, type=int)
# parser.add_argument('--arg_train_mark', '-arg_train_mark', help='train mark for updating model', default=3, type=int)
parser.add_argument('--arg_with_encoder', '-arg_with_encoder', help='if using encoder in training', default=0, type=int)
# parser.add_argument('--arg_iob_param', '-arg_iob_param', help='param for calculate iob', default=0.15, type=float)
# parser.add_argument('--arg_clip_value', '-arg_clip_value', help='gradient clip value for actor', default=0, type=float)
# parser.add_argument('--arg_delta_dist', '-arg_delta_dist', help='delta value used in distance function, how much is CF action different from orig action at 1 step', default=0.5, type=float)
# parser.add_argument('--arg_test_mark', '-arg_test_mark', help='marker for test', default=30, type=int)
parser.add_argument('--arg_train_one_trace', '-arg_train_one_trace', help='flag for training for 1 trace', default=0, type=int)
parser.add_argument('--arg_lambda_start_value', '-arg_lambda_start_value', help='lambda_start_value', default=1.0, type=float)
parser.add_argument('--arg_generate_train_trace_num', '-arg_generate_train_trace_num', help='generate_train_trace_num', default=150, type=int)
parser.add_argument('--arg_total_test_trace_num', '-arg_total_test_trace_num', help='total_test_trace_num', default=100, type=int)
parser.add_argument('--arg_total_train_steps', '-arg_total_train_steps', help='total_train_steps', default=3000, type=int)
parser.add_argument('--arg_log_interval', '-arg_log_interval', help='log_interval', default=50, type=int)
parser.add_argument('--arg_batch_size', '-arg_batch_size', help='batch_size', default=256, type=int)
parser.add_argument('--arg_gradient_steps', '-arg_gradient_steps', help='arg_gradient_steps', default=200, type=int)
parser.add_argument('--arg_epsilon', '-arg_epsilon', help='epsilon', default=0.1, type=float)
parser.add_argument('--arg_run_baseline', '-arg_run_baseline', help='run_baseline', default=0, type=int)
parser.add_argument('--arg_learning_rate', '-arg_learning_rate', help='learning_rate for SB3 actor and critic', default=0.00001, type=float)
parser.add_argument('--arg_total_timesteps_each_trace', '-arg_total_timesteps_each_trace', help='train time steps after sampling a new orig trace', default=500, type=int)
parser.add_argument('--arg_thre_for_fix_a', '-arg_thre_for_fix_a', help='threshold for fixing action', default=0.0, type=float)
parser.add_argument('--arg_if_user_assign_action', '-arg_if_user_assign_action', help='if use the action assigned by user', default=0, type=int)
parser.add_argument('--arg_if_single_env', '-arg_if_single_env', help='if only use 1 env to generate traces', default=1, type=int)
parser.add_argument('--arg_assigned_gravity', '-arg_assigned_gravity', help='assigned g for 1 env case', default=-10.0, type=float)
parser.add_argument('--arg_thread', '-arg_thread', help='thread for running exp', default=1, type=int)
parser.add_argument('--arg_total_trial_num', '-arg_total_trial_num', help='total_trial_num', default=1, type=int)
parser.add_argument('--arg_reward_weight', '-arg_reward_weight', help='reward weight', default=-1, type=float)
parser.add_argument('--arg_test_param', '-arg_test_param', help='test param', default=20, type=int)
parser.add_argument('--arg_ppo_train_time', '-arg_ppo_train_time', help='ppo training time', default=10, type=int)
parser.add_argument('--arg_use_exist_trace', '-arg_use_exist_trace', help='if use exist trace for exp', default=0, type=int)
parser.add_argument('--arg_exist_trace_id', '-arg_exist_trace_id', help='exist trace id', default=1, type=int)
parser.add_argument('--arg_exp_type', '-arg_exp_type', help='experiment type (1, 2, 3)', default=1, type=int)
parser.add_argument('--arg_if_constrain_on_s', '-arg_if_constrain_on_s', help='if set any constraint on state (RP 2)', default=0, type=int)
parser.add_argument('--arg_thre_for_s', '-arg_thre_for_s', help='threshold for state (RP 2)', default=-999999, type=float)
parser.add_argument('--arg_s_index', '-arg_s_index', help='index of state vector for the one to be constrained (RP 2)', default=2, type=int)
parser.add_argument('--arg_if_use_user_input', '-arg_if_use_user_input', help='if use user input action (RP 3)', default=0, type=int)
parser.add_argument('--arg_user_input_action', '-arg_user_input_action', help='value of user input action (RP 3), main,lateral', default='-1_0', type=str)


args_cf_diabetic = parser.parse_args()
torch.set_num_threads(args_cf_diabetic.arg_thread)

# add cf generator during testing
# test the outside model (e.g. PPO) in test_env
ENV_NAME = 'LunarLanderContinuous-v2'
#reward_fun_id = 0
noise_type = 'normal_noise'
#lr = 0.001
#train_step = 300
render = False
reward = 0
done = False
ENV_ID_list = [ENV_NAME]
gravity_list = [-11, -9, -5, -10, -8, -6] # generalize: first 3 to train, last 3 to test
trained_controller_dict = {}
all_env_dict = {}
for id in ENV_ID_list:
    for g in gravity_list:
        all_env_dict[g] = gym.make(id, gravity=g)
        #all_env_dict[id] = gym.make(id)
        ######
        if args_cf_diabetic.arg_if_single_env == 1:
            print('Load personalized PPO model.')
            # Load the saved statistics, env and model of outside controller, trained for each patient
            ppo_folder = "/p/citypm/AIFairness/Counterfactual_Explanation/lunarlander_example/trained_ppo_lunarlander_personalize"
            #ppo_folder = "/home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander/trained_ppo_lunarlander_personalize"
            # ppo_model_path = '{ppo_folder}/ppo_lunarlander_personalize_trained_{arg_ppo_train_time}_{g}'.format(
            #     ppo_folder=ppo_folder, g=g, arg_ppo_train_time=args_cf_diabetic.arg_ppo_train_time)
            # if g==-11 or g==-5:
            #     ppo_model_path = '{ppo_folder}/ppo_lunarlander_personalize_trained_{arg_ppo_train_time}_{g}'.format(ppo_folder=ppo_folder,g=g,arg_ppo_train_time=args_cf_diabetic.arg_ppo_train_time)
            # else:
            ppo_model_path = '{ppo_folder}/ppo_lunarlander_personalize_999_{g}'.format(ppo_folder=ppo_folder,g=g)
            # For RP1, ppo for all g are 999
            #ppo_model_path = '{ppo_folder}/ppo_lunarlander_personalize_999_{g}'.format(ppo_folder=ppo_folder, g=g)
            #stats_path = '{best_result_folder}/vecnormalize.pkl'.format(best_result_folder=ppo_folder)
            # Load the agent and save in dict
            trained_controller_dict[g] = PPO.load(ppo_model_path, env=all_env_dict[g],
                                                   custom_objects={
                                                       'observation_space': all_env_dict[g].observation_space,
                                                       'action_space': all_env_dict[g].action_space})
        else:
            print('Load generalized PPO model.')
            # Load the saved statistics, env and model of outside controller, trained for each patient
            ppo_folder = "/p/citypm/AIFairness/Counterfactual_Explanation/lunarlander_example/trained_ppo_lunarlander_generalize"
            #ppo_folder = "/home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander/trained_ppo_lunarlander_generalize"
            ppo_model_path = '{ppo_folder}/ppo_{ENV_NAME}_generalize_g_trained_{arg_ppo_train_time}'.format(
                ppo_folder=ppo_folder, ENV_NAME=id, arg_ppo_train_time=args_cf_diabetic.arg_ppo_train_time)
            # ppo_model_path = '{ppo_folder}/ppo_model_{ENV_NAME}_generalize_g_trained_780'.format(
            #     ppo_folder=ppo_folder, ENV_NAME=id)
            #stats_path = '{best_result_folder}/vecnormalize.pkl'.format(best_result_folder=ppo_folder)
            # Load the agent and save in dict
            trained_controller_dict[g] = PPO.load(ppo_model_path, env=all_env_dict[g],
                                                   custom_objects={
                                                       'observation_space': all_env_dict[g].observation_space,
                                                       'action_space': all_env_dict[g].action_space})
        # ######
        # # Load the saved statistics, env and model of outside controller
        # ppo_folder = "/home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander/trained_ppo_lunarlander"
        # ppo_model_path = '{ppo_folder}/ppo_model_{ENV_NAME}'.format(
        #     ppo_folder=ppo_folder,
        #     ENV_NAME=id)
        # stats_path = '{best_result_folder}/vecnormalize.pkl'.format(best_result_folder=ppo_folder)
        # # Load the agent and save in dict
        # trained_controller_dict[id] = PPO.load(ppo_model_path, env=all_env_dict[id],
        #                       custom_objects={'observation_space': all_env_dict[id].observation_space,
        #                                       'action_space': all_env_dict[id].action_space})


#device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:{cuda_id}".format(cuda_id=args_cf_diabetic.arg_cuda))
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
print(device, ' Thread: ', args_cf_diabetic.arg_thread)
print('Exp id: ', args_cf_diabetic.arg_id, 'whole_trace_step: ', args_cf_diabetic.arg_whole_trace_step, ' check_len: ', args_cf_diabetic.arg_cflen)
# print('whole_trace_step: ', args_cf_diabetic.arg_whole_trace_step, ' check_len: ', args_cf_diabetic.arg_cflen,
#       ' train_eps: ', args_cf_diabetic.arg_train_eps, ' test_eps: ', args_cf_diabetic.arg_test_eps,
#       ' update_iteration: ', args_cf_diabetic.arg_update_iteration, ' dist_func: ', args_cf_diabetic.arg_dist_func,
#       ' arg_clip_value: ', args_cf_diabetic.arg_clip_value, ' arg_delta_dist: ', args_cf_diabetic.arg_delta_dist,
#       ' arg_test_mark: ', args_cf_diabetic.arg_test_mark, ' arg_train_one_trace: ', args_cf_diabetic.arg_train_one_trace)

# print(type(args_cf_diabetic.arg_whole_trace_step), type(args_cf_diabetic.arg_cuda), type(args_cf_diabetic.arg_cflen),
#         type(args_cf_diabetic.arg_train_eps), type(args_cf_diabetic.arg_test_eps), type(args_cf_diabetic.arg_update_iteration),
#       type(args_cf_diabetic.arg_train_split))
test_step = args_cf_diabetic.arg_whole_trace_step #100000
# save_folder = '/p/citypm/AIFairness/Counterfactual_Explanation/lunarlander_example/DDPG_results_slurm/all_trace_len_{test_step}_{dist_func}_{arg_exp_id}'.format(test_step=test_step,
#                                                                 arg_exp_id=args_cf_diabetic.arg_id,dist_func=args_cf_diabetic.arg_dist_func)
#mkdir(path='/p/citypm/AIFairness/Counterfactual_Explanation/lunarlander_example/DDPG_results_slurm')

# save_folder = '/home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander/results/all_trace_len_{test_step}_{dist_func}_lr_{arg_learning_rate}_grad_{arg_gradient_steps}_{arg_exp_id}'.format(test_step=test_step,
#                                                                  arg_exp_id=args_cf_diabetic.arg_id,dist_func=args_cf_diabetic.arg_dist_func, arg_learning_rate=args_cf_diabetic.arg_learning_rate,
#                                                                 arg_gradient_steps=args_cf_diabetic.arg_gradient_steps)
lunar_lander_file_folder = '/p/citypm/AIFairness/Counterfactual_Explanation/lunarlander_example'
#lunar_lander_file_folder = '/home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander/results'
all_folder = '{lunar_lander_file_folder}/TD3_results_slurm_e{arg_exp_type}_RP3_S_UET'.format(arg_exp_type=args_cf_diabetic.arg_exp_type,lunar_lander_file_folder=lunar_lander_file_folder)
#all_folder = '/p/citypm/AIFairness/Counterfactual_Explanation/lunarlander_example/DDPG_results_slurm_e{arg_exp_type}'.format(arg_exp_type=args_cf_diabetic.arg_exp_type)
if os.path.exists(all_folder):
    pass
else:
    mkdir(path=all_folder)
save_folder = '{all_folder}/all_trace_len_{test_step}_{dist_func}_lr_{arg_learning_rate}_grad_{arg_gradient_steps}_{arg_exp_id}'.format(test_step=test_step,all_folder=all_folder,
                                                                                                                                        arg_exp_id=args_cf_diabetic.arg_id,dist_func=args_cf_diabetic.arg_dist_func, arg_learning_rate=args_cf_diabetic.arg_learning_rate,
                                                                                                                                        arg_gradient_steps=args_cf_diabetic.arg_gradient_steps)
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
    print('DEL EXIST Folder.')
mkdir(save_folder)

# save parameters of this exp
argsDict = args_cf_diabetic.__dict__
with open('{save_folder}/parameters.txt'.format(save_folder=save_folder),'w') as f:
    f.writelines('-----------------------start---------------------------'+'\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg+':'+str(value)+'\n')
    f.writelines('-----------------------end---------------------------')

#best_result_folder = "/p/citypm/AIFairness/Counterfactual_Explanation"
#best_result_folder = '/home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander'
trace_file_folder = save_folder
print('Begin running outside controller.')

# # Load the saved statistics, env and model of outside controller
# model_train_step = 100000
# ppo_model_path = '{best_result_folder}/lunar_lander_ppo_model_normal_noise_{model_train_step}_steps'.format(
#     best_result_folder=best_result_folder,
#     model_train_step=model_train_step)
# #stats_path = '{best_result_folder}/vecnormalize.pkl'.format(best_result_folder=best_result_folder)
#
# test_env = gym.make(ENV_NAME)#Monitor(gym.make(ENV_NAME))
# #  do not update them at test time
# test_env.training = False
# # reward normalization is not needed at test time
# test_env.norm_reward = False
# # Load the agent
# test_model = PPO.load(ppo_model_path, env=test_env,
#                       custom_objects={'observation_space': test_env.observation_space,
#                                       'action_space': test_env.action_space})
# observation = test_env.reset()
# print('observation: ', observation, type(observation))

train_index_all_env_df_list = []
test_index_all_env_df_list = []
trace_df_all_env_dict = {}

def generate_original_traces_for_each_env(trace_file_folder, gravity, ENV_NAME, all_env_dict, trained_controller_dict):
    trace_file_path = '{folder}/lunar_lander_trace_gravity_{g}.csv'.format(folder=trace_file_folder, g=gravity)
    # Iterate over the number of epochs for generating the original trace
    epochs = 1
    # store the original trace in df
    columns = ['gravity','step', 'episode', 'episode_step', 'action', 'observation', 'observation_new',
                                     'reward', 'episode_return', 'done']
    exclude_col_name = ['moon', 'sky_polys', 'lander', 'legs', 'joints']
    trace_df = pd.DataFrame(columns=columns, dtype=object)

    num_episodes = 0
    exit_flag = False
    test_env = all_env_dict[gravity]
    test_model = trained_controller_dict[gravity]
    print('test_env: ', test_env, ' test_model: ', test_model, ' gravity: ', gravity)
    ##############################
    with torch.no_grad():
        for epoch in range(epochs):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            # num_episodes = 0
            accumulated_reward = 0
            episode_length = 0
            observation = test_env.reset()
            # Iterate over the steps of each epoch
            for t in range(test_step):
                # print('current time step in test env: ', t)
                if render:
                    test_env.render()
                # Get the logits, action, and take one step in the environment
                action, _states = test_model.predict(observation)
                observation_new, reward, done, info = test_env.step(action)
                accumulated_reward += reward
                episode_length += 1
                trace_df = get_rocket_trace(test_env, trace_df, t, num_episodes, episode_length, action, _states,
                                            observation, observation_new, reward, accumulated_reward, done, gravity)

                # Update the observation
                observation = observation_new
                # Finish trajectory if reached to a terminal state
                terminal = done
                if (t == test_step - 1):
                    print('Trace is long enough.')
                    sum_return += accumulated_reward
                    sum_length += episode_length
                    test_env.close()
                    #observation, episode_return, episode_length = test_env.reset(), 0, 0
                    # trace_df = add_action_effect_to_trace(trace_df)
                    new_columns = [i for i in trace_df.columns if i not in exclude_col_name]
                    saved_trace = trace_df[new_columns]
                    #saved_trace = trace_df
                    #print(saved_trace)
                    saved_trace.to_csv(trace_file_path)
                    #print('Save trace file.')
                    exit_flag = True
                    break
                else:
                    if terminal or episode_length == 350:
                        num_episodes += 1
                        sum_return += accumulated_reward
                        sum_length += episode_length
                        observation, episode_return, episode_length = test_env.reset(), 0, 0
                    else:
                        # print('Trace not long enough.')
                        continue

                if exit_flag:
                    print('Finish generating all original traces.')
                    break

    check_len = args_cf_diabetic.arg_cflen  # 20 # length of CF traces
    interval = args_cf_diabetic.arg_cflen  # length of sliding window among original trace segments
    start_time_index = args_cf_diabetic.arg_cflen + 10  # the earlist time step in an original trace for training/testing
    time_index_checkpoint_list = [start_time_index + i * interval for i in range(800)]  # all possible time index for finding the original traces used in training/testing
    trace_df_per_eps_list = []  # store traces for each episode in trace_df
    eps_time_index_checkpoint_list = []  # store the {eps_idx: time_index} pairs for loading the parameters of original traces
    max_total_reward = 150
    for eps in range(num_episodes + 1):
        past_trace_this_eps = trace_df[trace_df["episode"] == eps]
        #print('len(past_trace_this_eps): ', len(past_trace_this_eps), ' len(trace_df): ', len(trace_df))
        for t in time_index_checkpoint_list:
            #print('t: ', t, ' int(len(past_trace_this_eps) * 0.75): ', int(len(past_trace_this_eps) * 0.75))
            if (t <= int(len(past_trace_this_eps) * 0.75)):
                this_segment = past_trace_this_eps[t - check_len:t]
                #print('this_segment index, Generate Orig: ', this_segment['episode'].tolist()[0], t, t - check_len, this_segment['episode_step'].tolist(), len(this_segment))
                total_reward_this_segment = this_segment['reward'].sum()
                # print('t: ', t, ' this_segment: ', this_segment, ' total_reward_this_segment: ', total_reward_this_segment)
                if total_reward_this_segment < max_total_reward:
                    #eps_time_index_checkpoint_list.append({eps: t})
                    eps_time_index_checkpoint_list.append({'ENV_NAME':ENV_NAME,'gravity':gravity,
                                                           'orig_episode':eps, 'orig_end_time_index':t})
                else:
                    #print('achieve max reward.')
                    pass
        trace_df_per_eps_list.append(past_trace_this_eps)
    # train DDPG with original traces selected randomly from eps_time_index_checkpoint_list
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]
    total_used_number = args_cf_diabetic.arg_total_used_num

    print('total generated index number: ', len(eps_time_index_checkpoint_list), ' total_used_number: ', total_used_number)
    eps_time_index_checkpoint_list = sample(eps_time_index_checkpoint_list, total_used_number)
    train_index = sample(eps_time_index_checkpoint_list, round(args_cf_diabetic.arg_train_split * len(eps_time_index_checkpoint_list)))  # {eps_idx: time_index} used for finding the original traces used to train DDPG
    test_index = [info_dict for info_dict in eps_time_index_checkpoint_list if info_dict not in train_index]  # {eps_idx: time_index} used for finding the original traces used to test DDPG

    train_index_df_this_patient = pd.DataFrame(columns=['ENV_NAME','gravity', 'orig_episode', 'orig_end_time_index'])
    test_index_df_this_patient = pd.DataFrame(columns=['ENV_NAME','gravity', 'orig_episode', 'orig_end_time_index'])
    for ix in train_index:
        #print('ix in train index: ', ix)
        train_index_df_this_patient = train_index_df_this_patient.append(ix, ignore_index=True)
    for ix in test_index:
        #print('ix in test_index: ', ix)
        test_index_df_this_patient = test_index_df_this_patient.append(ix, ignore_index=True)
    return train_index_df_this_patient, test_index_df_this_patient, trace_df

mem_before_generate_orig_trace = get_mem_use()
if args_cf_diabetic.arg_if_single_env == 0:# train and test with different envs
    if args_cf_diabetic.arg_use_exist_trace == 0:
        for g in gravity_list:
            ENV_NAME = ENV_ID_list[0]#'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type, patient_id=id)
            train_index_df_this_patient, test_index_df_this_patient, trace_df = generate_original_traces_for_each_env(trace_file_folder, g,
                                                                                              ENV_NAME,
                                                                                              all_env_dict,
                                                                                              trained_controller_dict)
            train_index_all_env_df_list.append(train_index_df_this_patient)
            test_index_all_env_df_list.append(test_index_df_this_patient)
            trace_df_all_env_dict[g] = trace_df
            #trace_file_path = '{folder}/lunar_lander_trace_gravity_{g}.csv'.format(folder=trace_file_folder, g=g)
            #print('trace_df: ', trace_df)
            #trace_df.to_csv(trace_file_path)
            #print('Save trace.', len(trace_df))
        if args_cf_diabetic.arg_exp_type==2:
            train_index_df = pd.concat(train_index_all_env_df_list[:3])  # use first 3 to train
            train_index_df = train_index_df.reset_index(drop=True)
            test_index_df = pd.concat(test_index_all_env_df_list[:3])  #  use the same 3 env for train to test, different traces
            test_index_df = test_index_df.reset_index(drop=True)
        else:
            train_index_df = pd.concat(train_index_all_env_df_list[:3])  # use first 3 to train
            train_index_df = train_index_df.reset_index(drop=True)
            test_index_df = pd.concat(test_index_all_env_df_list[3:])  # use the same 3 env for train to test, different traces
            test_index_df = test_index_df.reset_index(drop=True)
        # test_index_df = pd.concat(test_index_all_env_df_list[3:])#use last 3 to test
    else:
        exist_trace_folder = '{lunar_lander_file_folder}/exist_trace/exp_{arg_exp_type}/ppo_{arg_ppo_train_time}_generalize_group_{arg_exist_trace_id}'.format(
            lunar_lander_file_folder=lunar_lander_file_folder,
            arg_exist_trace_id=args_cf_diabetic.arg_exist_trace_id,
            arg_exp_type=args_cf_diabetic.arg_exp_type,
            arg_ppo_train_time=args_cf_diabetic.arg_ppo_train_time)
        exist_train_index_file_path = '{exist_trace_folder}/train_index_file.csv'.format(exist_trace_folder=exist_trace_folder)
        exist_test_index_file_path = '{exist_trace_folder}/test_index_file.csv'.format(exist_trace_folder=exist_trace_folder)
        for g in gravity_list:
            exist_trace_file_path = '{exist_trace_folder}/lunar_lander_trace_gravity_{g}.csv'.format(exist_trace_folder=exist_trace_folder, g=g)
            trace_df = pd.read_csv(exist_trace_file_path)
            trace_df = get_rocket_trace_from_saved_data(trace_df,save_path=trace_file_folder,orig_id=args_cf_diabetic.arg_exist_trace_id, g=g)
            trace_df_all_env_dict[g] = trace_df
        train_index_df = pd.read_csv(exist_train_index_file_path)
        test_index_df = pd.read_csv(exist_test_index_file_path)
else:# train and test with 1 env
    if args_cf_diabetic.arg_use_exist_trace == 0:
        #id = args_cf_diabetic.arg_assigned_gravity
        ENV_NAME = ENV_ID_list[0]#'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type, patient_id=id)
        train_index_df_this_patient, test_index_df_this_patient, trace_df = generate_original_traces_for_each_env(trace_file_folder,
                                                                                        args_cf_diabetic.arg_assigned_gravity,
                                                                                          ENV_NAME,
                                                                                          all_env_dict,
                                                                                          trained_controller_dict)
        train_index_all_env_df_list.append(train_index_df_this_patient)
        test_index_all_env_df_list.append(test_index_df_this_patient)
        trace_df_all_env_dict[args_cf_diabetic.arg_assigned_gravity] = trace_df
        train_index_df = pd.concat(train_index_all_env_df_list)
        test_index_df = pd.concat(test_index_all_env_df_list)
        #test_index_df = train_index_all_env_df_list[0]
        #trace_file_path = '{folder}/lunar_lander_trace_gravity_{g}.csv'.format(folder=trace_file_folder, g=args_cf_diabetic.arg_assigned_gravity)
        #trace_df.to_csv(trace_file_path)
    else:
        exist_trace_folder = '{lunar_lander_file_folder}/exist_trace/exp_{arg_exp_type}/ppo_{arg_ppo_train_time}_g_{arg_assigned_gravity}_group_{arg_exist_trace_id}'.format(
            lunar_lander_file_folder=lunar_lander_file_folder,
            arg_exist_trace_id=args_cf_diabetic.arg_exist_trace_id,
            arg_exp_type=args_cf_diabetic.arg_exp_type,
            arg_ppo_train_time=args_cf_diabetic.arg_ppo_train_time,
        arg_assigned_gravity=args_cf_diabetic.arg_assigned_gravity)
        exist_trace_file_path = '{exist_trace_folder}/lunar_lander_trace_gravity_{arg_assigned_gravity}.csv'.format(exist_trace_folder=exist_trace_folder,
                                                                                                              arg_assigned_gravity=args_cf_diabetic.arg_assigned_gravity)
        exist_train_index_file_path = '{exist_trace_folder}/train_index_file.csv'.format(exist_trace_folder=exist_trace_folder)
        exist_test_index_file_path = '{exist_trace_folder}/test_index_file.csv'.format(exist_trace_folder=exist_trace_folder)
        trace_df = pd.read_csv(exist_trace_file_path)
        trace_df = get_rocket_trace_from_saved_data(trace_df,save_path=trace_file_folder,orig_id=args_cf_diabetic.arg_exist_trace_id,g=args_cf_diabetic.arg_assigned_gravity)
        trace_df_all_env_dict[args_cf_diabetic.arg_assigned_gravity] = trace_df
        train_index_df = pd.read_csv(exist_train_index_file_path)
        test_index_df = pd.read_csv(exist_test_index_file_path)

gc.collect()
mem_after_generate_orig_trace = get_mem_use()
print('Mem usage, Generate Orig Trace: ', mem_after_generate_orig_trace-mem_before_generate_orig_trace)
train_index_path = '{save_folder}/train_index_file.csv'.format(save_folder=save_folder)
test_index_path = '{save_folder}/test_index_file.csv'.format(save_folder=save_folder)
train_index_df.to_csv(train_index_path)
test_index_df.to_csv(test_index_path)
print('train_index_df: ', len(train_index_df))
print('test_index_df: ', len(test_index_df))

train_counter = 0
test_counter = 0
test_eps_each_trace = args_cf_diabetic.arg_total_test_trace_num
trained_timestep_list = []
test_timestep_list = []
patient_info_input_dim = 13 # len of input patient state vector
patient_info_output_dim = 0#args_cf_diabetic.arg_patidim # len of compressed patient state vector
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
# ###########################################################
# print('Finish generating whole original trace.')
# #print('trace_df: ', len(trace_df))
# check_len = args_cf_diabetic.arg_cflen #length of CF traces
# interval = args_cf_diabetic.arg_cflen # length of sliding window among original trace segments
# start_time_index = args_cf_diabetic.arg_cflen + 10 # the earlist time step in an original trace for training/testing
# time_index_checkpoint_list = [start_time_index + i * interval for i in range(800)]  # all possible time index for finding the original traces used in training/testing
# #time_index_checkpoint_list = [start_time_index+i*interval for i in range(500)] # all possible time index for finding the original traces used in training/testing
# #time_index_checkpoint_list = [t for t in time_index_checkpoint_list if t <= int(len(past_trace_this_eps)*0.9)]
# trace_df_per_eps_list = [] # store traces for each episode in trace_df
# eps_time_index_checkpoint_list = [] # store the {eps_idx: time_index} pairs for loading the parameters of original traces
# max_total_reward = 150
# for eps in range(num_episodes+1):
#     past_trace_this_eps = trace_df[trace_df["episode"] == eps]
#     for t in time_index_checkpoint_list:
#         if (t <= int(len(past_trace_this_eps))*0.75):
#             this_segment = past_trace_this_eps[t-check_len:t]
#             total_reward_this_segment = this_segment['reward'].sum()
#             #print('t: ', t, ' this_segment: ', this_segment, ' total_reward_this_segment: ', total_reward_this_segment)
#             if total_reward_this_segment < max_total_reward:
#                 eps_time_index_checkpoint_list.append({eps: t})
#             else:
#                 pass
#     trace_df_per_eps_list.append(past_trace_this_eps)
#
#
# # train DDPG with original traces selected randomly from eps_time_index_checkpoint_list
# state_dim = test_env.observation_space.shape[0]
# action_dim = test_env.action_space.shape[0]
#
# total_used_number = args_cf_diabetic.arg_total_used_num # training+testing
# print('total generated index number: ', len(eps_time_index_checkpoint_list))
# eps_time_index_checkpoint_list = sample(eps_time_index_checkpoint_list, total_used_number)
# train_index = sample(eps_time_index_checkpoint_list, round(args_cf_diabetic.arg_train_split * len(eps_time_index_checkpoint_list))) # {eps_idx: time_index} used for finding the original traces used to train DDPG
# test_index = [info_dict for info_dict in eps_time_index_checkpoint_list if info_dict not in train_index] # {eps_idx: time_index} used for finding the original traces used to test DDPG
# if args_cf_diabetic.arg_train_one_trace==1:
#     train_index = [train_index[0]]*len(train_index)
#     test_index = [train_index[0]]*len(test_index)
# print('train_index: ', len(train_index))
# print('test_index: ', len(test_index))
#
#
# train_counter = 0
# test_counter = 0
# #train_eps_each_trace = args_cf_diabetic.arg_train_eps #10
# test_eps_each_trace = args_cf_diabetic.arg_total_test_trace_num #10
# #update_iteration = args_cf_diabetic.arg_update_iteration #10
# trained_timestep_list = []
# test_timestep_list = []
# train_result_path = '{save_folder}/train_result'.format(save_folder=save_folder)
# mkdir(train_result_path)
# ddpg_save_directory = '{train_result_path}/trained_model'.format(train_result_path=train_result_path)
# mkdir(ddpg_save_directory)
# train_index_path = '{train_result_path}/train_index_file.csv'.format(train_result_path=train_result_path)
# test_index_path = '{train_result_path}/test_index_file.csv'.format(train_result_path=train_result_path)
# store_time_index(train_index, train_index_path)
# store_time_index(test_index, test_index_path)
#
#
# # total_train_round = args_cf_diabetic.arg_train_round # total train round for DDPG
# # train_mark = args_cf_diabetic.arg_train_mark
#
#
# index_already_test_list = [] # store the {eps:time_index} pairs that have beed tested
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

def convert_scenario(scenario_str):
    # {'meal': {'time': [472.0, 767.0, 1121.0], 'amount': [66, 66, 75]}}
    #print('scenario_str: ', scenario_str)
    dict = {}
    t_list = []
    amount_list = []
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

def set_fixed_orig_trace(orig_trace_df):
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

    return orig_trace_df

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
            #print('info_dict_idx: ', info_dict_idx)
            # info_dict = train_index[info_dict_idx]
            # for k, v in info_dict.items():
            #     trace_eps, current_time_step = k, v
            # print('Train DDPG with traces end at: ', current_time_step, ' trace_eps: ', trace_eps)
            #TODO: set a fixed original trace for training
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

def test_ver_0(rl_model, patient_BW, delta_dist, test_index, each_test_round, trace_df_per_eps_list, index_already_test_list,improve_percentage_thre,train_counter,
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
            # print('info_dict.items(): ', info_dict.items())
            # for k, v in info_dict.items():
            #     trace_eps, current_time_step = k, v
            #print('Test DDPG with traces end at: ', current_time_step, ' trace_eps: ', trace_eps)
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
    #print('accumulated_reward_test_set_df: ', len(accumulated_reward_test_set_df))
    #(better_difference_perc, max_difference_perc, improve_percentage) = static_accumulate_reward(accumulated_reward_test_set_df,improve_percentage_thre)
    # print('Among all testing traces, {a}% gain better accumulated reward, {b}% improve over {improve_percentage_thre}%'.format(
    #     a=better_difference_perc*100, b=improve_percentage*100, improve_percentage_thre=improve_percentage_thre*100))
    # print('Trained {train_counter}, Among all testing traces, {a}% gain better accumulated reward, {b}% gain better max accumulated reward.'.format(train_counter=train_counter,
    #         a=better_difference_perc * 100, b=max_difference_perc * 100))
    #print('All test index used.')
    #print('Test finished.')
    cf_trace_test_file_path = '{save_folder}/cf_trace_file_{mode}_trained_{train_counter}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline,
                                                                                    mode=mode, train_counter=train_counter, trace_eps=trace_eps, current_time_step=current_time_step)
    # cf_accumulated_reward_test_file_path = '{train_result_path}/cf_test_accumulated_reward_difference_{mode}.csv'.format(train_result_path=train_result_path,mode=mode)
    if len(CF_trace_test_list)!=0:
        CF_test_trace_total = pd.concat(CF_trace_test_list)
        CF_test_trace_total.to_csv(cf_trace_test_file_path)
        #accumulated_reward_difference_total_df = pd.concat(accumulated_reward_difference_total_list)
        #print('accumulated_reward_difference_total_df: ', accumulated_reward_difference_total_df)
        #accumulated_reward_difference_total_df.to_csv(cf_accumulated_reward_test_file_path)
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
         fixed_trace_this_eps_df, mode='test_cf', baseline_path=None, trial_index=1):
    save_folder_baseline = '{save_folder}/baseline_results/trial_{trial_index}'.format(save_folder=save_folder, trial_index=trial_index)
    mkdir(save_folder_baseline)
    # test with a set of original traces
    test_counter = 0
    #index_not_test = [info_dict for info_dict in test_index if info_dict not in index_already_test_list]
    index_not_test = [info_dict for info_dict in test_index]
    accumulated_reward_difference_total_list = []
    orig_trace_df_list = []
    CF_trace_test_list = []
    accumulated_reward_test_set_df = pd.DataFrame(columns=['id','gravity','orig_trace_episode', 'orig_end_step', 'orig_accumulated_reward', 'cf_accumulated_reward',
                                                'difference', 'percentage', 'cf_count_distance','cf_pairwise_distance',
                                                'effect_distance', 'orig_effect', 'cf_effect', 'cf_iob_distance','orig_iob_distance', 'orig_start_action_effect','cf_aveg_cgm','if_generate_cf'])
    # info_dict = test_index
    # for k, v in info_dict.items():
    #     trace_eps, current_time_step = k, v
    env_name, trace_eps, current_time_step, gravity = test_index['ENV_NAME'], test_index['orig_episode'], \
                                    test_index['orig_end_time_index'], test_index['gravity']
    ##
    # print('info_dict.items(): ', info_dict.items())
    # for k, v in info_dict.items():
    #     trace_eps, current_time_step = k, v
    #print('Test Baseline PPO with traces end at: ', current_time_step, ' trace_eps: ', trace_eps, ' gravity: ', gravity, ' model: ', baseline_path)
    past_trace_this_eps = fixed_trace_this_eps_df  # trace_df_per_eps_list[trace_eps]
    #index_already_test_list.append(info_dict)
    results = cf_generator.cf_generator_Lunar_Lander(past_trace_this_eps, current_time_step, args_cf_diabetic.arg_cflen,
                                                     env_name,gravity,
                                                     rl_model, mode=mode,
                                                     run_eps_each_trace=test_eps_each_trace,
                                                     orig_trace_episode=trace_eps, patient_BW=patient_BW,
                                                     delta_dist=delta_dist,
                                                     iob_param=0,
                                                     train_episodes_mark_test=train_counter,
                                                     baseline_path=baseline_path)

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
            accumulated_reward_dict = {'gravity':gravity,'id': test_counter, 'orig_trace_episode': trace_eps,
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
                cf_aveg_cgm = 0#cf_aveg_cgm_list[accumulated_reward_list.index(item)]
                cf_iob_distance = cf_IOB_distance_list[accumulated_reward_list.index(item)]
                orig_iob_distance = orig_IOB_distance_list[accumulated_reward_list.index(item)]
                orig_start_action_effect = orig_start_action_effect_list[accumulated_reward_list.index(item)]
                cf_count_distance = cf_distance_count_list[accumulated_reward_list.index(item)]
                cf_pairwise_distance = cf_pairwise_distance_list[accumulated_reward_list.index(item)]
                accumulated_reward_dict = {'id': test_counter, 'gravity':gravity,'orig_trace_episode': trace_eps,
                                               'orig_end_step': current_time_step,
                                               'orig_accumulated_reward': orig_accu_r,
                                               'cf_accumulated_reward': cf_accu_r,
                                               'difference': cf_accu_r - orig_accu_r,
                                                'percentage': perc,
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
        accumulated_reward_dict = {'id': test_counter, 'gravity':gravity, 'orig_trace_episode': trace_eps,
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

    #print('accumulated_reward_test_set_df: ', len(accumulated_reward_test_set_df))
    #(better_difference_perc, max_difference_perc, improve_percentage) = static_accumulate_reward(accumulated_reward_test_set_df,improve_percentage_thre)
    # print('Among all testing traces, {a}% gain better accumulated reward, {b}% improve over {improve_percentage_thre}%'.format(
    #     a=better_difference_perc*100, b=improve_percentage*100, improve_percentage_thre=improve_percentage_thre*100))
    # print('Trained {train_counter}, Among all testing traces, {a}% gain better accumulated reward, {b}% gain better max accumulated reward.'.format(train_counter=train_counter,
    #         a=better_difference_perc * 100, b=max_difference_perc * 100))
    #print('All test index used.')
    #print('Test finished.')
    cf_trace_test_file_path = '{save_folder}/cf_trace_file_{mode}_trained_{train_counter}_g_{gravity}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline,
                                                        mode=mode, gravity=gravity, train_counter=train_counter, trace_eps=trace_eps, current_time_step=current_time_step)
    # cf_accumulated_reward_test_file_path = '{train_result_path}/cf_test_accumulated_reward_difference_{mode}.csv'.format(train_result_path=train_result_path,mode=mode)
    columns_cf = ['gravity', 'orig_trace_episode','orig_end_step', 'step', 'episode', 'episode_step', 'action', 'cf_start', 'lambda_value',
               'if_user_input','observation', 'observation_new','reward', 'accumulated_reward', 'done']
    columns_orig = ['gravity', 'step', 'episode', 'episode_step', 'action', 'observation', 'observation_new',
               'reward', 'episode_return', 'done']
    if len(CF_trace_test_list)!=0:
        CF_test_trace_total = pd.concat(CF_trace_test_list)
        CF_test_trace_total = CF_test_trace_total[columns_cf]
        CF_test_trace_total.to_csv(cf_trace_test_file_path)
        #accumulated_reward_difference_total_df = pd.concat(accumulated_reward_difference_total_list)
        #print('accumulated_reward_difference_total_df: ', accumulated_reward_difference_total_df)
        #accumulated_reward_difference_total_df.to_csv(cf_accumulated_reward_test_file_path)
        #actual_test_index_path = '{save_folder}/actual_test_index_file_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline, trace_eps=trace_eps, current_time_step=current_time_step)
        #store_time_index(test_timestep_list, actual_test_index_path)
        total_orig_trace_test = pd.concat(orig_trace_df_list)
        total_orig_trace_test=total_orig_trace_test[columns_orig]
        total_orig_trace_test.to_csv('{save_folder}/orig_trace_test_g_{gravity}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline,
                                                                                        gravity=gravity, trace_eps=trace_eps, current_time_step=current_time_step))
    else:
        CF_test_trace_total = None
    accumulated_reward_test_set_df.to_csv('{save_folder}/accumulated_reward_{mode}_trained_{train_counter}_g_{gravity}_{trace_eps}_{current_time_step}.csv'.format(save_folder=save_folder_baseline,
                                                            mode=mode,gravity=gravity,
                                                            train_counter=train_counter, trace_eps=trace_eps, current_time_step=current_time_step))
    del total_orig_trace_test, accumulated_reward_test_set_df
    gc.collect()
    return #index_already_test_list, CF_test_trace_total, accumulated_reward_test_set_df

def get_trace_info_for_reset_env(past_trace_this_eps, current_time_index, check_len, ENV_NAME, gravity, user_input_folder=None):
    # get the initial state for env_CF, which is the check_len step in this eps
    max_reward_per_step = 150
    CF_start_step = current_time_index - check_len + 1
    #print('CF_start_step: ', CF_start_step, type(CF_start_step))
    #reward_CF = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step + 1]['reward'].values[0]
    #print('reset env_CF')
    # if want to change actions starting from CF_start_step, then should reset env to state at CF_start_step-1
    #initial_state_obs = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step + 1]['observation'].values[0]
    initial_state_obs = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['observation'].values[0]
    #initial_state_obs_new_CF = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step + 1]['observation_new'].values[0]
    initial_state_moon = None #past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['moon'].values[0]
    initial_state_sky_polys = None #past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['sky_polys'].values[0]
    initial_state_lander = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['lander'].values[0]
    #print('initial_state_lander: ', initial_state_lander, type(initial_state_lander))
    initial_state_legs = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['legs'].values[0]
    initial_state_joints = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['joints'].values[0]
    #print('initial_state_legs: ', initial_state_legs, type(initial_state_legs))
    #print('initial_state_joints: ', initial_state_joints, type(initial_state_joints))
    initial_state_terrain_hight = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['terrain_hight'].values[0]
    #episode_return_CF = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step + 1]['episode_return'].values[0]
    #episode_length_CF = CF_start_step - 1
    #past_action_str = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step + 1]['action'].values[0][1: -1].split(',')
    #initial_state_action = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step + 1]['action'].values[0]#[float(x) for x in past_action_str]
    initial_state_action = past_trace_this_eps[past_trace_this_eps['episode_step'] == CF_start_step]['action'].values[0]
    env_CF_train = gym.make(ENV_NAME, gravity=gravity)
    #print('env_CF_train.action_space: ', env_CF_train.action_space, env_CF_train.action_space.shape[0])
    num_actions = env_CF_train.action_space.shape[0]
    # reset states for CF env
    kwargs_cf = {'if_determain':1, 'moon':initial_state_moon, 'sky_polys':initial_state_sky_polys,
                          'lander_dict':initial_state_lander, 'legs_dict_list':initial_state_legs,
                          'joints_dict_list':initial_state_joints,
                          'initial_state_action':initial_state_action, 'initial_state_terrain_hight':initial_state_terrain_hight,
                 'initial_state_obs': initial_state_obs, 'gravity':gravity}
    obs_CF = env_CF_train.reset(**kwargs_cf)
    # obs_CF = env_CF_train.reset(if_determain=1, moon=initial_state_moon, sky_polys=initial_state_sky_polys,
    #                       lander_dict=initial_state_lander, legs_dict_list=initial_state_legs,
    #                       joints_dict_list=initial_state_joints,
    #                       initial_state_action=initial_state_action, initial_state_terrain=initial_state_terrain)
    # state_dim = env_CF_train.observation_space.shape[0]
    # action_dim = env_CF_train.action_space.shape[0]
    orig_trace_for_CF = past_trace_this_eps[CF_start_step-1:current_time_index]
    #print('orig_trace_for_CF index Train&Test: ', orig_trace_for_CF['episode'].tolist()[0], orig_trace_for_CF['episode_step'].tolist(), len(orig_trace_for_CF), current_time_index,' reset to: ', CF_start_step)
    # orig_trace_for_CF.to_csv(original_trace_file_path)
    orig_action_trace = orig_trace_for_CF['action'].tolist()
    orig_state_trace = orig_trace_for_CF['observation'].tolist()
    # print('orig_state_trace: ', orig_state_trace)
    # if args_cf_diabetic.arg_use_exist_trace==1: # convert str to float for the original action and state trace
    #     for a_idx in range(0, len(orig_action_trace)):
    #         action_str = orig_action_trace[a_idx]
    #         action_float = convert_str_to_list(action_str,split_type=",")
    #         orig_action_trace[a_idx] = action_float
    #         state_str = orig_state_trace[a_idx]
    #         state_float = convert_str_to_list(action_str,split_type=",")
    #         orig_state_trace[a_idx] = state_float
    orig_start_action_effect = None#orig_trace_for_CF['action_effect'].tolist()[0]

    thre_fixed_step_index_list = []
    thre_ddpg_action_index_list = []
    for k in range(CF_start_step, current_time_index + 1):  # set some step as fixed action step
        #print(k, list(range(CF_start_step, current_time_index + 1)))
        if orig_action_trace[k - CF_start_step][0]< args_cf_diabetic.arg_thre_for_fix_a:
            thre_fixed_step_index_list.append([1]*num_actions)
            thre_ddpg_action_index_list.append([0]*num_actions)
        else:
            thre_fixed_step_index_list.append([0]*num_actions)
            thre_ddpg_action_index_list.append([1]*num_actions)
    if args_cf_diabetic.arg_if_user_assign_action == 1:
        this_eps = past_trace_this_eps['episode'].tolist()[0]
        user_input_this_segment_path = '{user_input_folder}/user_input_lunarlander_g_{gravity}_{this_eps}_{current_time_index}.csv'.format(
            user_input_folder=user_input_folder,gravity=gravity,
            this_eps=this_eps, current_time_index=current_time_index)
        user_input_this_segment_df = pd.read_csv(user_input_this_segment_path)
        user_fixed_step_index_list = user_input_this_segment_df['step'].tolist()
        user_fixed_step_index_list = [(x - CF_start_step) for x in user_fixed_step_index_list]
    else:
        user_fixed_step_index_list = []
        user_input_this_segment_df = None

    # print('orig_trace_for_CF: ', orig_trace_for_CF)
    J0 = sum(orig_trace_for_CF['reward'].tolist())  # the accumulated reward of the original trace
    if J0 < max_reward_per_step * len(orig_trace_for_CF['reward'].tolist()):  # only seek CF when orig trace doen not have the max reward, else there won't be any improvement
        return kwargs_cf, orig_trace_for_CF, orig_action_trace, orig_state_trace, orig_start_action_effect, J0,\
                thre_fixed_step_index_list, thre_ddpg_action_index_list, user_fixed_step_index_list, user_input_this_segment_df
    else:
        print('No better CF to find.')
        return None

# train and test for training with multiple trace from multiple/single env
#@profile(precision=4,stream=open("//home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander/results/memory_profiler_Run_Single_Exp.log","w+"))
def run_experiment_ddpg_sb3_single(model_CF, trace_eps, current_time_step, fixed_trace_this_eps_df, parameter_dict,
                                   this_train_round,mode='train',baseline_result_dict=None,model_type='ddpg'):
    # parameters
    orig_trace_episode = trace_eps
    CF_start_step = current_time_step - args_cf_diabetic.arg_cflen + 1
    iob_param = parameter_dict['iob_param']
    patient_BW = parameter_dict['patient_BW']
    cf_len = parameter_dict['cf_len']
    #train_round = parameter_dict['train_round']
    total_test_trace_num = parameter_dict['total_test_trace_num']
    ENV_NAME = parameter_dict['ENV_NAME']
    gravity = parameter_dict['gravity']
    if args_cf_diabetic.arg_if_user_assign_action == 1:
        user_input_folder = '' # undecided yet
    else:
        user_input_folder = None
    reset_env_results = get_trace_info_for_reset_env(past_trace_this_eps=fixed_trace_this_eps_df, current_time_index=current_time_step,
                                                     check_len=cf_len, ENV_NAME=ENV_NAME, user_input_folder=user_input_folder,gravity=gravity)
    if mode=='train':
        if reset_env_results is not None:
            (kwargs_cf, orig_trace_for_CF, orig_action_trace, orig_state_trace, orig_start_action_effect,
             J0, thre_fixed_step_index_list, thre_ddpg_action_index_list, user_fixed_step_index_list, user_input_this_segment_df) = reset_env_results
            #_,all_cf_trace_training_time,all_train_result = \
            #print('orig_action_trace: ', orig_action_trace[0], type(orig_action_trace[0]), type(orig_action_trace[0][0]))
            #print('orig_state_trace: ', orig_state_trace[0], type(orig_state_trace[0]), type(orig_state_trace[0][0]))
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
                ENV_ID=ENV_NAME,
                gravity=gravity,
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
            (kwargs_cf, orig_trace_for_CF, orig_action_trace, orig_state_trace, orig_start_action_effect,
             J0, thre_fixed_step_index_list, thre_ddpg_action_index_list, user_fixed_step_index_list, user_input_this_segment_df) = reset_env_results
            all_cf_trace_test, all_test_result, all_test_statistic = model_CF.test(
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
                ENV_ID=ENV_NAME,
                gravity=gravity,
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
            all_cf_trace_test, all_test_result, all_test_statistic = None, None, None
        return all_cf_trace_test, all_test_result, all_test_statistic

#@profile(precision=4,stream=open("/home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander/results/memory_profiler_Train_All.log","w+"))
def train_ddpg_sb3_all(all_train_index_df, all_test_index_df, ENV_NAME_dummy, save_folder, callback,
                       trained_controller_dict, all_env_dict,trace_df_all_env_dict, baseline_result_folder,trial_index):
    # for server ver
    save_folder_train = '{save_folder}/train'.format(save_folder=save_folder)
    save_folder_test = '{save_folder}/test'.format(save_folder=save_folder)
    mkdir(save_folder_train)
    mkdir(save_folder_test)
    patient_info_input_dim = 13
    patient_info_output_dim = 0#args_cf_diabetic.arg_patidim
    with_encoder = args_cf_diabetic.arg_with_encoder
    iob_param = 0#args_cf_diabetic.arg_iob_param
    delta_dist = 0#args_cf_diabetic.arg_delta_dist
    patient_BW = 0
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

    env_CF_train = gym.make(ENV_NAME_dummy, gravity=-10.0)
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
                       ENV_NAME=ENV_NAME_dummy,
                       trained_controller_dict=trained_controller_dict,
                       all_env_dict=all_env_dict,
                       )
    model_CF.set_logger(new_logger)
    test_trace_df_list = []
    test_result_df_list = []
    columns_cf = ['gravity', 'orig_trace_episode', 'step', 'episode', 'episode_step', 'action', 'cf_start',
                  'lambda_value',
                  'if_user_input',
                  'observation', 'observation_new',
                  'reward', 'accumulated_reward', 'done']
    #time_6 = time.perf_counter()
    for r in range(1, total_train_round+1): # train with all orig traces a few rounds
        counter = 0
        #all_train_index_df = all_train_index_df.sample(frac=1)
        time_3 = time.perf_counter()
        mem_before_train_1_trace = get_mem_use()
        random_train_trace_index = np.random.randint(0, len(all_train_index_df))
        for item, row in all_train_index_df.loc[[random_train_trace_index]].iterrows():
            env_name = row['ENV_NAME']
            trace_eps = row['orig_episode']
            current_time_step = row['orig_end_time_index']
            gravity = row['gravity']
            print('Train: ', r, counter, env_name, gravity, trace_eps, current_time_step)
            # trace_eps = row['orig_episode']
            # current_time_step = row['orig_end_time_index']
            # print('Train: ', r, trace_eps, current_time_step)
            parameter_dict = {'gravity':gravity,'patient_info_input_dim': patient_info_input_dim,
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
            fixed_orig_past_trace_df = trace_df_all_env_dict[gravity]
            fixed_trace_this_eps_df = fixed_orig_past_trace_df[fixed_orig_past_trace_df['episode'] == trace_eps]
            time_1 = time.perf_counter()
            # # get the distance value and the total reward of this trace from baseline
            # baseline_file_path = '{baseline_result_folder}/accumulated_reward_test_baseline_trained_0_g_{gravity}_{trace_eps}_{current_time_step}.csv'.format(
            #     baseline_result_folder=baseline_result_folder, gravity=gravity, trace_eps=trace_eps,
            #     current_time_step=current_time_step)
            # baseline_df = pd.read_csv(baseline_file_path)
            # baseline_total_reward_mean = baseline_df['cf_accumulated_reward'].mean()
            # baseline_total_reward_difference_mean = baseline_df['difference'].mean()
            # baseline_distance_mean = baseline_df['cf_pairwise_distance'].mean()
            # baseline_result_dict = {'baseline_total_reward_mean': baseline_total_reward_mean,
            #                         'baseline_total_reward_difference_mean': baseline_total_reward_difference_mean,
            #                         'baseline_distance_mean': baseline_distance_mean}
            # all_cf_trace_training_time_this_seg, all_train_result_this_seg = \
            run_experiment_ddpg_sb3_single(model_CF, trace_eps, current_time_step, fixed_trace_this_eps_df,
                                                                                                            parameter_dict,
                                                                                                            this_train_round=r,
                                                                                                            mode='train',
                                           baseline_result_dict=None)
            #del baseline_df, baseline_result_dict
            gc.collect()
            time_2 = time.perf_counter()
            #print('Lunar Lander, time for training 1 original trace: ', time_2 - time_1)

            # if all_cf_trace_training_time_this_seg is not None:
            #     all_cf_trace_training_time_this_seg.to_pickle(
            #         '{save_folder_train}/cf_traces_train_{gravity}_{trace_eps}_{current_time_step}_r_{r}.pkl'.format(
            #             save_folder_train=save_folder_train,
            #             gravity=gravity, trace_eps=trace_eps, current_time_step=current_time_step, r=r))
            #     all_train_result_this_seg.to_pickle(
            #         '{save_folder_train}/train_result_{gravity}_{trace_eps}_{current_time_step}_r_{r}.pkl'.format(
            #             save_folder_train=save_folder_train,
            #             gravity=gravity, trace_eps=trace_eps, current_time_step=current_time_step, r=r))
            #     # all_train_cf_trace_list.append(all_cf_trace_training_time_this_seg)
            #     # all_train_result_list.append(all_train_result_this_seg)
            #     #
            #     # all_train_cf_trace_df = pd.concat(all_train_cf_trace_list)
            #     # all_train_cf_trace_df = all_train_cf_trace_df[columns_cf]
            #     # all_train_result_df = pd.concat(all_train_result_list)
            #     # all_train_cf_trace_df.to_csv('{save_folder}/all_cf_traces_train.csv'.format(save_folder=save_folder))
            #     # all_train_result_df.to_csv('{save_folder}/all_train_result.csv'.format(save_folder=save_folder))
            # if counter==int(len(all_train_index_df)*0.5):
            #     all_test_cf_trace_df_this_round, all_test_result_df_this_round = test_ddpg_sb3_all(model_CF,
            #                                                                                        all_test_index_df,
            #                                                                                        save_folder,
            #                                                                                        trace_df_all_env_dict,
            #                                                                                        this_train_round=(r-0.5))
            #     test_trace_df_list.append(all_test_cf_trace_df_this_round)
            #     test_result_df_list.append(all_test_result_df_this_round)
            #     all_test_cf_trace_df = pd.concat(test_trace_df_list)
            #     all_test_cf_trace_df = all_test_cf_trace_df[columns_cf]
            #     all_test_result_df = pd.concat(test_result_df_list)
            #     all_test_cf_trace_df.to_csv('{save_folder}/all_cf_traces_test.csv'.format(save_folder=save_folder))
            #     all_test_result_df.to_csv('{save_folder}/all_test_result.csv'.format(save_folder=save_folder))
            counter += 1
        mem_after_train_1_trace = get_mem_use()
        #print('Mem usage, Train 1 round: ', mem_after_train_1_trace - mem_before_train_1_trace, ' Total: ', mem_after_train_1_trace)
        time_4 = time.perf_counter()
        #print('Lunar Lander, time for Training One Full Round: ', time_4 - time_3)
        mem_before_test_1_trace = get_mem_use()
        #all_test_cf_trace_df_this_round, all_test_result_df_this_round = \
        if r%args_cf_diabetic.arg_test_param==0 and r>=20:
            print('Test.')
            test_ddpg_sb3_all(model_CF, all_test_index_df, save_folder_test, trace_df_all_env_dict,
                              this_train_round=r,baseline_result_folder=baseline_result_folder,trial_index=trial_index)
            test_ddpg_sb3_all(model_CF, all_train_index_df, save_folder_train, trace_df_all_env_dict,
                              this_train_round=r, baseline_result_folder=baseline_result_folder,
                              trial_index=trial_index)
            mem_after_test_1_trace = get_mem_use()
            #print('Mem usage, Test 1 round: ', mem_after_test_1_trace - mem_before_test_1_trace, ' Total: ',mem_after_test_1_trace)
            time_5 = time.perf_counter()
            #print('Lunar Lander, time for Testing One Full Round: ', time_5 - time_4)
        # test_trace_df_list.append(all_test_cf_trace_df_this_round)
        # test_result_df_list.append(all_test_result_df_this_round)
        # all_test_cf_trace_df = pd.concat(test_trace_df_list)
        # all_test_cf_trace_df = all_test_cf_trace_df[columns_cf]
        # all_test_result_df = pd.concat(test_result_df_list)
        # all_test_cf_trace_df.to_csv('{save_folder}/all_cf_traces_test.csv'.format(save_folder=save_folder))
        # all_test_result_df.to_csv('{save_folder}/all_test_result.csv'.format(save_folder=save_folder))
        # all_test_cf_trace_df_this_round.to_pickle('{save_folder_test}/cf_traces_test_r_{r}.pkl'.format(save_folder_test=save_folder_test, r=r))
        # all_test_result_df_this_round.to_pickle('{save_folder_test}/test_result_r_{r}.pkl'.format(save_folder_test=save_folder_test, r=r))
    #time_7 = time.perf_counter()
    #print('Lunar Lander, time for One Full Experiment: ', time_7 - time_6)
    return

#@profile(precision=4,stream=open("/home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander/results/memory_profiler_Test_All.log","w+"))
def test_ddpg_sb3_all(model_CF, all_test_index_df, save_folder, trace_df_all_env_dict, this_train_round, baseline_result_folder,trial_index):
    # for server ver
    patient_info_input_dim = 13
    patient_info_output_dim = 0#args_cf_diabetic.arg_patidim
    with_encoder = args_cf_diabetic.arg_with_encoder
    iob_param = 0#args_cf_diabetic.arg_iob_param
    delta_dist = 0#args_cf_diabetic.arg_delta_dist
    patient_BW = 0
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
        gravity = row['gravity']
        #print('Test: ', gravity, trace_eps, current_time_step)
        parameter_dict = {'gravity':gravity,'patient_info_input_dim': patient_info_input_dim,
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
                                'ENV_NAME': env_name,
                                }
        fixed_orig_past_trace_df = trace_df_all_env_dict[gravity]
        fixed_trace_this_eps_df = fixed_orig_past_trace_df[fixed_orig_past_trace_df['episode'] == trace_eps]
        # #
        # baseline_path = trained_controller_dict[gravity]
        # test_baseline(rl_model=None, patient_BW=patient_BW, delta_dist=delta_dist, test_index=row,
        #               train_counter=this_train_round,
        #               fixed_trace_this_eps_df=fixed_trace_this_eps_df, mode='test_baseline',trial_index=trial_index,
        #               baseline_path=baseline_path)
        # #get the distance value and the total reward of this trace from baseline
        # baseline_file_path = '{baseline_result_folder}/accumulated_reward_test_baseline_trained_0_g_{gravity}_{trace_eps}_{current_time_step}.csv'.format(
        #     baseline_result_folder=baseline_result_folder,gravity=gravity,trace_eps=trace_eps,current_time_step=current_time_step)
        # baseline_df = pd.read_csv(baseline_file_path)
        # baseline_total_reward_mean = baseline_df['cf_accumulated_reward'].mean()
        # baseline_total_reward_difference_mean = baseline_df['difference'].mean()
        # baseline_distance_mean = baseline_df['cf_pairwise_distance'].mean()
        # baseline_result_dict = {'baseline_total_reward_mean':baseline_total_reward_mean,
        #                         'baseline_total_reward_difference_mean':baseline_total_reward_difference_mean,
        #                         'baseline_distance_mean':baseline_distance_mean}
        # time_6 = time.perf_counter()
        print('Test DDPG.')
        all_cf_trace_test_list_this_seg_ddpg, all_test_result_this_seg_ddpg, test_statistic_dict_this_seg_ddpg = run_experiment_ddpg_sb3_single(model_CF, trace_eps,
                                                                                              current_time_step,
                                                                                              fixed_trace_this_eps_df,
                                                                                              parameter_dict,
                                                                                              this_train_round=this_train_round,
                                                                                              mode='test',
                                                                                            baseline_result_dict=None, model_type='ddpg')
        print('Test PPO.')
        all_cf_trace_test_list_this_seg_baseline, all_test_result_this_seg_baseline, test_statistic_dict_this_seg_baseline = run_experiment_ddpg_sb3_single(
            model_CF, trace_eps,
            current_time_step,
            fixed_trace_this_eps_df,
            parameter_dict,
            this_train_round=this_train_round,
            mode='test',
            baseline_result_dict=None, model_type='baseline')
        # time_7 = time.perf_counter()
        # print('Lunar Lander, time for testing 1 original trace: ', time_7 - time_6)

        if all_cf_trace_test_list_this_seg_ddpg is not None:
            all_test_cf_trace_list_ddpg.extend(all_cf_trace_test_list_this_seg_ddpg)
            all_test_result_list_ddpg.append(all_test_result_this_seg_ddpg)
            all_test_statistic_list_ddpg.append(pd.DataFrame([test_statistic_dict_this_seg_ddpg]))
        if all_cf_trace_test_list_this_seg_baseline is not None:
            all_test_cf_trace_list_baseline.extend(all_cf_trace_test_list_this_seg_baseline)
            all_test_result_list_baseline.append(all_test_result_this_seg_baseline)
            all_test_statistic_list_baseline.append(pd.DataFrame([test_statistic_dict_this_seg_baseline]))

    all_test_cf_trace_df_ddpg = pd.concat(all_test_cf_trace_list_ddpg)
    all_test_result_df_ddpg = pd.concat(all_test_result_list_ddpg)
    all_test_statistic_df_ddpg = pd.concat(all_test_statistic_list_ddpg)
    all_test_cf_trace_df_ddpg.to_pickle('{save_folder_test}/cf_traces_test_ddpg_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
    all_test_result_df_ddpg.to_pickle('{save_folder_test}/test_result_ddpg_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
    all_test_statistic_df_ddpg.to_pickle('{save_folder_test}/test_statistic_ddpg_r_{r}.pkl'.format(save_folder_test=save_folder, r=this_train_round))
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
    #del baseline_df, baseline_result_dict
    gc.collect()
    # all_test_cf_trace_df = pd.concat(all_test_cf_trace_list)
    # all_test_result_df = pd.concat(all_test_result_list)
    # all_test_cf_trace_df.to_csv('{save_folder}/all_cf_traces_test.csv'.format(save_folder=save_folder))
    # all_test_result_df.to_csv('{save_folder}/all_test_result.csv'.format(save_folder=save_folder))

    return #all_test_cf_trace_df, all_test_result_df

# train and test for training with 1 trace from 1 patient
def run_experiment_ddpg_sb3_1_trace_1_env(model_CF, trace_eps, current_time_step, fixed_trace_this_eps_df, parameter_dict,
                                   this_train_round, mode='train'):
    # parameters
    orig_trace_episode = trace_eps
    CF_start_step = current_time_step - args_cf_diabetic.arg_cflen + 1
    iob_param = parameter_dict['iob_param']
    patient_BW = parameter_dict['patient_BW']
    cf_len = parameter_dict['cf_len']
    ENV_NAME = parameter_dict['ENV_NAME']
    gravity = parameter_dict['gravity']
    #train_round = parameter_dict['train_round']
    total_test_trace_num = parameter_dict['total_test_trace_num']
    #ENV_NAME = parameter_dict['ENV_NAME']
    if args_cf_diabetic.arg_if_user_assign_action == 1:
        user_input_folder = '' # undecided yet
    else:
        user_input_folder = None
    reset_env_results = get_trace_info_for_reset_env(past_trace_this_eps=fixed_trace_this_eps_df, current_time_index=current_time_step,
                                    check_len=cf_len,ENV_NAME=ENV_NAME, gravity=gravity, user_input_folder=user_input_folder)
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
                ENV_ID=ENV_NAME,
                gravity=gravity,
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
                ENV_ID=ENV_NAME,
                gravity=gravity,
                fixed_step_index_list=thre_fixed_step_index_list,
                user_fixed_step_index_list=user_fixed_step_index_list,
                user_input_this_segment_df=user_input_this_segment_df,
            )
        else:
            all_cf_trace_test, all_test_result = None, None
        return all_cf_trace_test, all_test_result

def train_ddpg_sb3_all_1_trace_1_env(all_train_index_df, ENV_NAME_dummy, save_folder, callback,
                       trained_controller_dict, all_env_dict, trace_df_all_env_dict):
    # for server ver
    patient_info_input_dim = 13
    patient_info_output_dim = 0#args_cf_diabetic.arg_patidim
    with_encoder = args_cf_diabetic.arg_with_encoder
    iob_param = 0#args_cf_diabetic.arg_iob_param
    delta_dist = 0#args_cf_diabetic.arg_delta_dist
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
        gravity = row['gravity']
        test_index_df = all_train_index_df[item:item+1]
        print('test_index_df: ', test_index_df)

        # fixed_trace_this_eps_df = trace_df[trace_df["episode"] == trace_eps]
        fixed_orig_past_trace_df = trace_df_all_env_dict[gravity]
        fixed_trace_this_eps_df = fixed_orig_past_trace_df[fixed_orig_past_trace_df['episode'] == trace_eps]
        # set up a new model for this trace
        # set training log
        tmp_path = "{save_folder}/logs/train_log".format(save_folder=save_folder)
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        env_CF_train = gym.make(ENV_NAME_dummy, gravity=gravity)
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
            print('Train: ', r, env_name, trace_eps, current_time_step, gravity)
            parameter_dict = {'gravity': gravity,
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
            all_cf_trace_training_time_this_seg,all_train_result_this_seg = run_experiment_ddpg_sb3_1_trace_1_env(model_CF, trace_eps,
                                                                    current_time_step, fixed_trace_this_eps_df, parameter_dict,
                                                                    this_train_round=r, mode='train')

            if all_cf_trace_training_time_this_seg is not None:
                all_train_cf_trace_list.append(all_cf_trace_training_time_this_seg)
                all_train_result_list.append(all_train_result_this_seg)
                all_train_cf_trace_df = pd.concat(all_train_cf_trace_list)
                all_train_result_df = pd.concat(all_train_result_list)
                all_train_cf_trace_df.to_csv('{save_folder}/all_cf_traces_train.csv'.format(save_folder=save_folder))
                all_train_result_df.to_csv('{save_folder}/all_train_result.csv'.format(save_folder=save_folder))
            counter += 1
            all_test_cf_trace_df_this_round, all_test_result_df_this_round = test_ddpg_sb3_all_1_trace_1_env(model_CF, test_index_df,
                                                                                               save_folder, trace_df_all_env_dict, this_train_round=r)
            test_trace_df_list.append(all_test_cf_trace_df_this_round)
            test_result_df_list.append(all_test_result_df_this_round)
        all_test_cf_trace_df = pd.concat(test_trace_df_list)
        all_test_result_df = pd.concat(test_result_df_list)
        all_test_cf_trace_df.to_csv('{save_folder}/all_cf_traces_test.csv'.format(save_folder=save_folder))
        all_test_result_df.to_csv('{save_folder}/all_test_result.csv'.format(save_folder=save_folder))

    return


def test_ddpg_sb3_all_1_trace_1_env(model_CF, all_test_index_df, save_folder, trace_df_all_env_dict, this_train_round):
    # for server ver
    patient_info_input_dim = 13
    patient_info_output_dim = 0#args_cf_diabetic.arg_patidim
    with_encoder = args_cf_diabetic.arg_with_encoder
    iob_param = 0#args_cf_diabetic.arg_iob_param
    delta_dist = 0#args_cf_diabetic.arg_delta_dist
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
        gravity = row['gravity']
        print('Test: ', env_name, trace_eps, current_time_step, gravity)
        parameter_dict = {'gravity':gravity, 'patient_info_input_dim': patient_info_input_dim,
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
                                'ENV_NAME': env_name,
                                }
        # fixed_trace_this_eps_df = trace_df[trace_df["episode"] == trace_eps]
        fixed_orig_past_trace_df = trace_df_all_env_dict[gravity]
        fixed_trace_this_eps_df = fixed_orig_past_trace_df[fixed_orig_past_trace_df['episode'] == trace_eps]
        all_cf_trace_test_this_seg, all_test_result_this_seg = run_experiment_ddpg_sb3_1_trace_1_env(model_CF, trace_eps,
                                                                        current_time_step, fixed_trace_this_eps_df, parameter_dict,
                                                                            this_train_round=this_train_round,mode='test')
        if all_cf_trace_test_this_seg is not None:
            all_test_cf_trace_list.append(all_cf_trace_test_this_seg)
            all_test_result_list.append(all_test_result_this_seg)

    all_test_cf_trace_df = pd.concat(all_test_cf_trace_list)
    all_test_result_df = pd.concat(all_test_result_list)
    # all_test_cf_trace_df.to_csv('{save_folder}/all_cf_traces_test.csv'.format(save_folder=save_folder))
    # all_test_result_df.to_csv('{save_folder}/all_test_result.csv'.format(save_folder=save_folder))

    return all_test_cf_trace_df, all_test_result_df
##########################################
all_train_index_df = train_index_df
all_test_index_df = test_index_df
counter = 0

# Hyperparameters
render = False
# Set Callback to save evaluation results
callback_freq=100
train_step = 100
lr = 0
# new best model is saved according to the mean_reward
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
mem_before_exp = get_mem_use()
callback = None #CallbackList([checkpoint_callback])
if args_cf_diabetic.arg_train_one_trace==0:# train with multiple trace for each model, generalization
    # # run baseline PPO
    # for item, row in all_test_index_df.iterrows():
    # #for orig_trace in train_index:
    #     # parameters
    #     # for key, value in orig_trace.items():
    #     #     trace_eps = key#row['orig_episode']
    #     #     current_time_step = value#row['orig_end_time_index']
    #     env_name = row['ENV_NAME']
    #     trace_eps = row['orig_episode']
    #     current_time_step = row['orig_end_time_index']
    #     gravity = row['gravity']
    #     CF_start_step = current_time_step - args_cf_diabetic.arg_cflen + 1
    #     #print(trace_eps, current_time_step)
    #     orig_trace_episode = trace_eps
    #
    #     delta_dist = 0#args_cf_diabetic.arg_delta_dist
    #     patient_BW = 0
    #     fixed_trace_this_eps_df = trace_df_all_env_dict[gravity]
    #     fixed_trace_this_eps_df = fixed_trace_this_eps_df[fixed_trace_this_eps_df['episode'] == trace_eps]
    #     baseline_path = trained_controller_dict[gravity]
    #     # run baseline test
    #     #print('Run baseline PPO.')
    #     #test_index = {trace_eps: current_time_step}
    #     test_index = row
    #     time_1 = time.perf_counter()
    #     test_baseline(rl_model=None, patient_BW=patient_BW, delta_dist=delta_dist, test_index=test_index, train_counter=train_counter,
    #                     fixed_trace_this_eps_df=fixed_trace_this_eps_df, mode='test_baseline', baseline_path=baseline_path)
    #     time_2 = time.perf_counter()
    #     print('Time for test 1 trace by Baseline: ', time_2-time_1)
    time_3 = time.perf_counter()
    # train DDPG_CF and test
    for trial in range(1, args_cf_diabetic.arg_total_trial_num+1):
        print('Start trial: ', trial)
        ddpg_cf_results_folder = '{save_folder}/td3_cf_results/trial_{t}'.format(save_folder=save_folder, t=trial)
        mkdir(ddpg_cf_results_folder)
        ENV_NAME_dummy = ENV_ID_list[0]#'simglucose-{patient_type}{patient_id}-v0'.format(patient_type='adult', patient_id='5')
        for id in ENV_ID_list:
            for g in gravity_list:
                all_env_dict[g] = gym.make(id, gravity=g)

        train_ddpg_sb3_all(all_train_index_df, all_test_index_df, ENV_NAME_dummy, ddpg_cf_results_folder, callback,
                                            trained_controller_dict, all_env_dict, trace_df_all_env_dict,
                           baseline_result_folder='{save_folder}/baseline_results'.format(save_folder=save_folder), trial_index=trial)
        time_4 = time.perf_counter()
        print('Time for 1 Full Exp by DDPG: ', time_4 - time_3)
        mem_after_exp = get_mem_use()
        print('Mem for 1 Full Exp by DDPG: ', mem_after_exp - mem_before_exp)
        gc.collect()
    time_5 = time.perf_counter()
    print('Time for all trials by DDPG: ', time_5 - time_3)
    mem_after_exp = get_mem_use()
    print('Mem for all trial by DDPG: ', mem_after_exp - mem_before_exp)
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
        gravity = row['gravity']
        CF_start_step = current_time_step - args_cf_diabetic.arg_cflen + 1
        # print(trace_eps, current_time_step)
        orig_trace_episode = trace_eps

        delta_dist = 0  # args_cf_diabetic.arg_delta_dist
        patient_BW = 0
        fixed_trace_this_eps_df = trace_df_all_env_dict[gravity]
        fixed_trace_this_eps_df = fixed_trace_this_eps_df[fixed_trace_this_eps_df['episode'] == trace_eps]
        baseline_path = trained_controller_dict[gravity]
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
    ENV_NAME_dummy = ENV_ID_list[0]
    for id in ENV_ID_list:
        for g in gravity_list:
            all_env_dict[g] = gym.make(id, gravity=g)

    train_ddpg_sb3_all_1_trace_1_env(all_train_index_df, ENV_NAME_dummy, ddpg_cf_results_folder, callback,
                                         trained_controller_dict, all_env_dict, trace_df_all_env_dict,
                                     baseline_result_folder='{save_folder}/baseline_result'.format(save_folder=save_folder))
