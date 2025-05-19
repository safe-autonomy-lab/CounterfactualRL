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
sys.path.append("/home/cpsgroup/anaconda3/envs/CF_LunarLander_train_ppo_generalize/lib/python3.8/site-packages/gym")
#import cf_generator
import matplotlib.pyplot as plt
from random import sample
import warnings
warnings.filterwarnings('ignore')
import argparse
# modified the core.py in gym==0.21.0 to make it work

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
    dict = {'active': str(b2Body_obj.active), 'angle':str(b2Body_obj.angle), 'angularDamping':str(b2Body_obj.angularDamping),
            'angularVelocity':str(b2Body_obj.angularVelocity), 'awake':str(b2Body_obj.awake),
            'bullet':str(b2Body_obj.bullet), 'contacts':str(b2Body_obj.contacts), 'fixedRotation':str(b2Body_obj.fixedRotation), #'fixtures':str(b2Body_obj.fixtures),
            'fixtures_vertices': [str(x) for x in b2Body_obj.fixtures[0].shape.vertices],
            'fixtures_density': str(b2Body_obj.fixtures[0].density),
            'fixtures_friction': str(b2Body_obj.fixtures[0].friction),
            'fixtures_restitution': str(b2Body_obj.fixtures[0].restitution),
            'inertia':str(b2Body_obj.inertia), 'joints':str(b2Body_obj.joints), 'linearDamping':str(b2Body_obj.linearDamping),
            'linearVelocity':[str(b2Body_obj.linearVelocity.x), str(b2Body_obj.linearVelocity.y)],
            'localCenter':[str(b2Body_obj.localCenter.x), str(b2Body_obj.localCenter.y)],
            'mass':str(b2Body_obj.mass),
            'massData':{'I': str(b2Body_obj.massData.I), 'center':[str(b2Body_obj.massData.center.x),str(b2Body_obj.massData.center.y)],
                                   'mass':str(b2Body_obj.massData.mass)},
            'position':[str(b2Body_obj.position.x), str(b2Body_obj.position.y)],
            'sleepingAllowed':str(b2Body_obj.sleepingAllowed), 'transform':str(b2Body_obj.transform), 'type':b2Body_obj.type, 'userData':b2Body_obj.userData,
            'worldCenter':[str(b2Body_obj.worldCenter.x), str(b2Body_obj.worldCenter.y)]}
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
    dict = {'active': str(b2RevoluteJoint_obj.active), 'angle': str(b2RevoluteJoint_obj.angle),
            'anchorA': [str(b2RevoluteJoint_obj.anchorA.x),str(b2RevoluteJoint_obj.anchorA.y)],
            'anchorB': [str(b2RevoluteJoint_obj.anchorB.x),str(b2RevoluteJoint_obj.anchorB.y)],
            'bodyA': str(b2RevoluteJoint_obj.bodyA), 'bodyB': str(b2RevoluteJoint_obj.bodyB),
            'limitEnabled': str(b2RevoluteJoint_obj.limitEnabled), 'limits': [str(b2RevoluteJoint_obj.limits[0]),str(b2RevoluteJoint_obj.limits[1])],
            'lowerLimit': str(b2RevoluteJoint_obj.lowerLimit),
            'maxMotorTorque': 40, 'motorEnabled': str(b2RevoluteJoint_obj.motorEnabled), 'motorSpeed': str(b2RevoluteJoint_obj.motorSpeed),
            'speed': str(b2RevoluteJoint_obj.speed),
            'type': b2RevoluteJoint_obj.type, 'upperLimit': str(b2RevoluteJoint_obj.upperLimit), 'userData': b2RevoluteJoint_obj.userData}

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

    terrain_param_dict = {'helipad_x1':env.helipad_x1, 'helipad_x2':env.helipad_x2, 'helipad_y':env.helipad_y}
    current_lander_feature_dict = get_b2Body_param(current_lander, 'lander')
    current_leg_feature_dict_0 = get_b2Body_param(current_legs[0], 'leg')
    current_leg_feature_dict_1 = get_b2Body_param(current_legs[1], 'leg')
    current_joint_feature_dict_0 = get_b2RevoluteJoint_param(current_lander.joints[0].joint)
    current_joint_feature_dict_1 = get_b2RevoluteJoint_param(current_lander.joints[1].joint)

    current_legs_dict_list = [current_leg_feature_dict_0, current_leg_feature_dict_1]
    current_joints_dict_list = [current_joint_feature_dict_0, current_joint_feature_dict_1]
    trace_dict = {'gravity':gravity,'step': step, 'episode': current_episode, 'episode_step': current_episode_step, 'action': current_action.tolist(),
                  'observation': current_obs,
                  'observation_new': current_obs_new,
                  'reward': current_reward,
                  'episode_return': current_accumulated_reward, 'done': current_done,
                  'moon': current_moon, 'sky_polys': current_sky_polys,
                  'lander': current_lander_feature_dict, 'legs': current_legs_dict_list,
                  'joints': current_joints_dict_list, 'terrain':terrain_param_dict,'states':current_states}
    trace_df = trace_df.append(trace_dict, ignore_index=True)
    return trace_df


# In[ ]:

# get parameters
parser = argparse.ArgumentParser(description='cf diabetic train ppo argparse')
parser.add_argument('--arg_exp_id', '-arg_exp_id', help='exp_id', default=0, type=int)
#parser.add_argument('--arg_gravity', '-arg_gravity', help='gravity', default=-10.0, type=float)
parser.add_argument('--arg_cuda', '-arg_cuda', help='cuda id', default=1, type=int)
parser.add_argument('--arg_train_step_each_env', '-arg_train_step_each_env', help='train step', default=5000, type=int)
parser.add_argument('--arg_callback_step', '-arg_callback_step', help='callback step', default=1000, type=int)
parser.add_argument('--arg_lr', '-arg_lr', help='learn rate', default=0.001, type=float)
#parser.add_argument('--arg_test_step_each_env', '-arg_test_step', help='test step', default=3400, type=int)
parser.add_argument('--arg_train_round', '-arg_train_round', help='train_round', default=5, type=int)
parser.add_argument('--arg_test_epochs_each_env', '-arg_test_epochs_each_env', help='test_epochs_each_env', default=5, type=int)
parser.add_argument('--arg_max_test_time_each_env', '-arg_max_test_time_each_env', help='max_test_time_each_env', default=3400, type=int)
parser.add_argument('--arg_if_train_personalize', '-arg_if_train_personalize', help='if_train_personalize', default=0, type=int)
parser.add_argument('--arg_assigned_gravity', '-arg_assigned_gravity', help='assigned_gravity for training single env', default=-10.0, type=float)

args_ppo_diabetic = parser.parse_args()

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:{cuda_id}".format(cuda_id=args_ppo_diabetic.arg_cuda))
print(device)
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
gravity_list = [-11, -9, -5, -10, -8, -6]
trained_controller_dict = {}
all_env_dict = {}
for id in ENV_ID_list:
    for g in gravity_list:
        all_env_dict[g] = gym.make(id, gravity=g)
if args_ppo_diabetic.arg_if_train_personalize==0:
    trace_file_folder = '/home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander/trained_ppo_lunarlander_generalize/lunar_lander_generalize_ppo_exp_{arg_exp_id}'.format(arg_exp_id=args_ppo_diabetic.arg_exp_id)
    mkdir(trace_file_folder)
else:
    trace_file_folder = '/home/cpsgroup/Counterfactual_Explanation/OpenAI_Example/lunar_lander/trained_ppo_lunarlander_personalize/lunar_lander_personalize_ppo_exp_{arg_exp_id}'.format(
        arg_exp_id=args_ppo_diabetic.arg_exp_id)
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


def train_ppo_generalize(train_round, ENV_ID_list, gravity_list, all_env_dict, train_step_each_env, save_folder, test_trace_result_path):
    first_train_flag = 1
    # The noise objects for DDPG
    noise_type = 'normal_noise'
    # set training log
    tmp_path = "{save_folder}/logs/{noise_type}/lr_{lr}/train_log".format(save_folder=save_folder,noise_type=noise_type, lr=lr)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    env_id = ENV_ID_list[0]
    for i in range(train_round):
        for g in gravity_list[0:3]:
            print('train ppo round: ', i, ' env_id: ', env_id, ' g: ', g)
            g_idx = gravity_list.index(g)
            # make env for this patient
            # all_env_dict[ENV_NAME] = gym.make(ENV_NAME)
            all_env_dict[g] = Monitor(gym.make(env_id, gravity=g))
            env = all_env_dict[g]
            env.reset()
            if first_train_flag==1:
                check_point_mark_0 = '{i}_{idx}'.format(i=i, idx=g_idx)
                model_save_path_0 = '{save_folder}/ppo_lunarlander_generalize_{check_point_mark}'.format(
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
                check_point_mark_1 = '{i}_{idx}'.format(i=i, idx=g_idx)
                model_save_path_1 = '{save_folder}/ppo_lunarlander_generalize_{check_point_mark}'.format(
                    save_folder=save_folder,
                    check_point_mark=check_point_mark_1)
                model.save(model_save_path_1)
            env.close()
        test_ppo_generalize(model_save_path_1, ENV_ID_list,gravity_list, all_env_dict,
                            test_epochs_each_env=args_ppo_diabetic.arg_test_epochs_each_env,
                            max_test_time_each_env=args_ppo_diabetic.arg_max_test_time_each_env,
                            train_round=i, results_folder=test_trace_result_path)

    return

def test_ppo_generalize(ppo_model_path, ENV_ID_list, gravity_list, all_env_dict, test_epochs_each_env, max_test_time_each_env, train_round, results_folder):
    trace_file_path = '{folder}/test_LL_trace_round_{train_round}.csv'.format(folder=results_folder, train_round=train_round)
    trace_df_list = []
    env_id = ENV_ID_list[0]
    columns = ['gravity', 'step', 'episode', 'episode_step', 'action', 'observation', 'observation_new',
               'reward', 'episode_return', 'done']
    for g in gravity_list[3:]:
        trace_len = 0
        aveg_accumulated_reward = 0
        #test_env = all_env_dict[g]
        all_env_dict[g] = Monitor(gym.make(env_id, gravity=g))
        test_env = all_env_dict[g]
        test_env.reset()
        # test the best model in a new env
        print('Begin testing: ', train_round, env_id, ' g: ', g)
        # result_col = ['id', 'hypoPercent', 'hyperPercent', 'TIR', 'step', 'reward_func']
        # results_path = '{folder}/medical_metrics_results_{env_id}_train_{train_round}.csv'.format(folder=results_folder,
        #                                                                                           env_id=env_id,train_round=train_round)
        # result_df = pd.DataFrame(columns=result_col)
        test_env.training = False
        test_env.norm_reward = False
        test_model = PPO.load(ppo_model_path, env=test_env)
        observation = test_env.reset()
        # store the past trace of this epoch

        trace_df = pd.DataFrame(columns=columns, dtype=object)

        num_episodes = 0
        for epoch in range(test_epochs_each_env):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            # num_episodes = 0
            accumulated_reward = 0
            episode_length = 0

            # Iterate over the steps of each epoch
            for t in range(max_test_time_each_env):
                # print('current time step in test env: ', t)
                if render:
                    test_env.render()
                # Get the logits, action, and take one step in the environment
                action, _states = test_model.predict(observation)
                observation_new, reward, done, info = test_env.step(action)
                accumulated_reward += reward
                episode_length += 1
                trace_df = get_rocket_trace(test_env, trace_df, t, num_episodes, episode_length, action, _states,
                                            observation, observation_new, reward, accumulated_reward, done, gravity=g)

                # Update the observation
                observation = observation_new
                # Finish trajectory if reached to a terminal state
                terminal = done
                if (t == max_test_time_each_env - 1) or terminal:
                    sum_return += accumulated_reward
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = test_env.reset(), 0, 0
                # if terminal:
                #     test_env.close()
                #     #print('Trace is long enough.')
                #     sum_return += accumulated_reward
                #     sum_length += episode_length
                #     num_episodes += 1
                #     all_env_dict[g] = Monitor(gym.make(env_id, gravity=g))
                #     test_env = all_env_dict[g]
                #     observation, episode_return, episode_length = test_env.reset(), 0, 0
                # elif (t == max_test_time_each_env - 1):
                #     sum_return += accumulated_reward
                #     sum_length += episode_length
                #     num_episodes += 1
                #     test_env.close()
                    # trace_df = add_action_effect_to_trace(trace_df)
                    #trace_df.to_csv(trace_file_path)
                    # exit_flag = True
                    # break
                # else:
                #     if terminal or episode_length == 350:
                #         num_episodes += 1
                #         sum_return += accumulated_reward
                #         sum_length += episode_length
                #         observation, episode_return, episode_length = test_env.reset(), 0, 0
                #     else:
                #         # print('Trace not long enough.')
                #         continue

            trace_df_list.append(trace_df)
            trace_len += len(trace_df)
            aveg_accumulated_reward += accumulated_reward
        trace_len /= num_episodes
        print('Average accumulated_reward this env: ', aveg_accumulated_reward/num_episodes, ' Aveg trace len this env: ', trace_len)
        test_env.close()
        print('close env')
    trace_df_all = pd.concat(trace_df_list)
    trace_df_all = trace_df_all[columns]
    #print('trace_df_all: ', len(trace_df_all))
    trace_df_all.to_csv(trace_file_path)
    #print('save trace')
    return

def train_ppo_personalize(train_round, ENV_ID_list, assigned_gravity, all_env_dict, train_step_each_env, save_folder, test_trace_result_path):
    first_train_flag = 1
    # The noise objects for DDPG
    noise_type = 'normal_noise'
    # set training log
    tmp_path = "{save_folder}/logs/{noise_type}/lr_{lr}/train_log".format(save_folder=save_folder,noise_type=noise_type, lr=lr)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    env_id = ENV_ID_list[0]
    # gravity_list = [assigned_gravity]
    # g = gravity_list[0]
    # g_idx = g#gravity_list.index(g)
    # # make env for this patient
    # # all_env_dict[ENV_NAME] = gym.make(ENV_NAME)
    # all_env_dict[g] = Monitor(gym.make(env_id, gravity=g))
    # env = all_env_dict[g]
    # env.reset()
    for i in range(train_round):

        gravity_list = [assigned_gravity]
        g = gravity_list[0]
        g_idx = g  # gravity_list.index(g)
        # make env for this patient
        # all_env_dict[ENV_NAME] = gym.make(ENV_NAME)
        all_env_dict[g] = Monitor(gym.make(env_id, gravity=g))
        env = all_env_dict[g]
        env.reset()
        print('train personalized ppo round: ', i, ' env_id: ', env_id, ' g: ', g)
        if first_train_flag == 1:
            check_point_mark_0 = '{i}_{idx}'.format(i=i, idx=g_idx)
            model_save_path_0 = '{save_folder}/ppo_lunarlander_personalize_{check_point_mark}'.format(
                save_folder=save_folder,
                check_point_mark=check_point_mark_0)
            model = PPO("MlpPolicy", env, verbose=10, learning_rate=lr, tensorboard_log=tmp_path)
            # model.set_logger(new_logger)
            model.learn(total_timesteps=train_step_each_env, callback=callback, reset_num_timesteps=False,
                        tb_log_name='PPO_personalize')
            model.save(model_save_path_0)
            first_train_flag = 0
            model_save_path_1 = model_save_path_0
        else:
            # print('Load model from: ', model_save_path_1)
            model = PPO.load(model_save_path_1)
            model.set_env(env, force_reset=True)
            # model.set_logger(new_logger)
            model.learn(total_timesteps=train_step_each_env, callback=callback, reset_num_timesteps=False,
                        tb_log_name='PPO_personalize')
            check_point_mark_1 = '{i}_{idx}'.format(i=i, idx=g_idx)
            model_save_path_1 = '{save_folder}/ppo_lunarlander_personalize_{check_point_mark}'.format(
                save_folder=save_folder,
                check_point_mark=check_point_mark_1)
            model.save(model_save_path_1)


        test_ppo_personalize(model_save_path_1, ENV_ID_list,assigned_gravity, all_env_dict,
                            test_epochs_each_env=args_ppo_diabetic.arg_test_epochs_each_env,
                            max_test_time_each_env=args_ppo_diabetic.arg_max_test_time_each_env,
                            train_round=i, results_folder=test_trace_result_path)
        env.close()

    return

def test_ppo_personalize(ppo_model_path, ENV_ID_list, assigned_gravity, all_env_dict, test_epochs_each_env, max_test_time_each_env, train_round, results_folder):
    trace_file_path = '{folder}/test_LL_trace_round_{train_round}.csv'.format(folder=results_folder, train_round=train_round)
    trace_df_list = []
    env_id = ENV_ID_list[0]
    columns = ['gravity', 'step', 'episode', 'episode_step', 'action', 'observation', 'observation_new',
               'reward', 'episode_return', 'done']
    gravity_list = [assigned_gravity]
    for g in gravity_list:
        trace_len = 0
        aveg_accumulated_reward = 0
        #test_env = all_env_dict[g]
        all_env_dict[g] = Monitor(gym.make(env_id, gravity=g))
        test_env = all_env_dict[g]
        test_env.reset()
        # test the best model in a new env
        print('Personalize model. Begin testing: ', train_round, env_id, ' g: ', g)
        # result_col = ['id', 'hypoPercent', 'hyperPercent', 'TIR', 'step', 'reward_func']
        # results_path = '{folder}/medical_metrics_results_{env_id}_train_{train_round}.csv'.format(folder=results_folder,
        #                                                                                           env_id=env_id,train_round=train_round)
        # result_df = pd.DataFrame(columns=result_col)
        test_env.training = False
        test_env.norm_reward = False
        test_model = PPO.load(ppo_model_path, env=test_env)
        observation = test_env.reset()
        # store the past trace of this epoch

        trace_df = pd.DataFrame(columns=columns, dtype=object)

        num_episodes = 0
        for epoch in range(test_epochs_each_env):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            # num_episodes = 0
            accumulated_reward = 0
            episode_length = 0

            # Iterate over the steps of each epoch
            for t in range(max_test_time_each_env):
                # print('current time step in test env: ', t)
                if render:
                    test_env.render()
                # Get the logits, action, and take one step in the environment
                action, _states = test_model.predict(observation)
                observation_new, reward, done, info = test_env.step(action)
                accumulated_reward += reward
                episode_length += 1
                trace_df = get_rocket_trace(test_env, trace_df, t, num_episodes, episode_length, action, _states,
                                            observation, observation_new, reward, accumulated_reward, done, gravity=g)

                # Update the observation
                observation = observation_new
                # Finish trajectory if reached to a terminal state
                terminal = done
                if (t == max_test_time_each_env - 1) or terminal:
                    sum_return += accumulated_reward
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = test_env.reset(), 0, 0
            trace_df_list.append(trace_df)
            trace_len += len(trace_df)
            aveg_accumulated_reward += accumulated_reward
        trace_len /= num_episodes
        print('Average accumulated_reward this env: ', aveg_accumulated_reward/num_episodes, ' Aveg trace len this env: ', trace_len)
        test_env.close()
        print('close env')
    trace_df_all = pd.concat(trace_df_list)
    trace_df_all = trace_df_all[columns]
    #print('trace_df_all: ', len(trace_df_all))
    trace_df_all.to_csv(trace_file_path)
    #print('save trace')
    return

if args_ppo_diabetic.arg_if_train_personalize==0:
    train_ppo_generalize(args_ppo_diabetic.arg_train_round, ENV_ID_list, gravity_list, all_env_dict, train_step_each_env, save_folder, test_trace_result_path)
else:
    train_ppo_personalize(args_ppo_diabetic.arg_train_round, ENV_ID_list, args_ppo_diabetic.arg_assigned_gravity, all_env_dict,
                          train_step_each_env, save_folder, test_trace_result_path)


# check on training log
# in terminal
# tensorboard  --logdir /home/cpsgroup/Counterfactual_Explanation/diabetic_example/trained_ppo_generalize/patient_adult_lr_0.001_exp_0/save_model/logs/normal_noise/lr_0.001/train_log/PPO_generalize_0
# open link in browser
# http://localhost:6006/