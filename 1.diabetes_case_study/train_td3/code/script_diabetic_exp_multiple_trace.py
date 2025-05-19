import subprocess
from itertools import product
import sys
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

exp_id = list(range(1,8)) # trial id: 1-7
arg_wts = [1000] # whole trace length for the original traces, this is the total length, and then the data will be sliced by sliding-window
train_split = [0.5] # split the trace segment dataset into training and testing
total_used_num = [12] # total number of trace segments provide by 1 patient, including both training and testing
train_round = [2500] # total training round of td3
total_timesteps_each_trace = [1]
log_interval = [5]
gradient_steps = [50]
arg_generate_train_trace_num = [1] # each time in training when sampling an original trace, generate 1 CF trace for it
total_test_trace_num = [10] # each time in test when sampling an original trace, generate 10 CF trace for it
learning_rate = [0.00001]

if_single_env = [0] # run experiment for single-env or multi-env
assigned_patient_id = [7] # patient id for single-env exps

if_meal_restrict = [0]
arg_total_r_thre = [20.0]
arg_min_cgm = [65.0]
arg_total_trial_num = [1]
arg_reward_weight = [-1.0]
arg_test_param = [20]
arg_ppo_train_time = [179] #single-env: 1; multi-env: 179
arg_use_exist_trace = [1] # if using existing original traces, if not switch to 0
arg_exist_trace_id = [6] #assign the existing trace id, single-env trace id: 11, 14; multi-env trace id: 3,6, this is the UET id in the file name

#param for P2-base
if_start_state_restrict = [0]
start_state_id = [0]
arg_if_constrain_on_s = [1]
arg_thre_for_s = [100] # state threshold value for switching between td3 and baseline, also used in P2-fixed
arg_s_index = [0] # state index of the one we set threshold on
thre_for_fix_a = [-1.0] # not used
#param for P2-fixed
arg_if_use_user_input = [1] # if run code for P2-fixed experiment
arg_user_input_action = [0.03] # user input action value

arg_exp_type = [3] #single-env:1, multi-env:3

df = pd.DataFrame()

total_comb = list(product(arg_wts, train_split, train_round, total_timesteps_each_trace, gradient_steps, learning_rate, thre_for_fix_a,
                            if_single_env, assigned_patient_id, total_used_num,
                          if_start_state_restrict, start_state_id, if_meal_restrict, arg_total_r_thre, arg_generate_train_trace_num,
                          total_test_trace_num, arg_min_cgm, log_interval, arg_total_trial_num, arg_reward_weight, arg_test_param,
                          arg_exp_type, arg_ppo_train_time, arg_use_exist_trace, arg_exist_trace_id,
                          arg_if_constrain_on_s, arg_thre_for_s, arg_s_index, arg_if_use_user_input, arg_user_input_action, exp_id))
with open(r'yourFolder\temp_experiments.txt', "w") as f:
    f.write("TaskID, arg_wts, train_split, train_round, total_timesteps_each_trace, gradient_steps, learning_rate, thre_for_fix_a,\
                if_single_env, assigned_patient_id, total_used_num, if_start_state_restrict, start_state_id, \
            if_meal_restrict, arg_total_r_thre, arg_generate_train_trace_num, total_test_trace_num, arg_min_cgm, "
            "log_interval, arg_total_trial_num, arg_reward_weight, arg_test_param, "
            "arg_exp_type, arg_ppo_train_time, arg_use_exist_trace, arg_exist_trace_id, "
            "arg_if_constrain_on_s, arg_thre_for_s, arg_s_index, "
            "arg_if_use_user_input, arg_user_input_action, "
            "exp_id\n")
    idx = 0
    for element in total_comb:
        idx += 1
        f.write(f"{idx},{element[0]},{element[1]},{element[2]},{element[3]},{element[4]},{element[5]},{element[6]},\
                            {element[7]},{element[8]},{element[9]},{element[10]},{element[11]},\
                {element[12]},{element[13]},{element[14]},{element[15]},{element[16]},{element[17]},{element[18]},{element[19]},\
                {element[20]},{element[21]},{element[22]},{element[23]},{element[24]},{element[25]},"
                f"{element[26]}, {element[27]},{element[28]},{element[29]},{element[30]},\n")
        dict = {'TaskID': idx, 'arg_wts': element[0], 'train_split': element[1], 'train_round': element[2],
                'total_timesteps_each_trace': element[3], 'gradient_steps': element[4],
                'learning_rate': element[5], 'thre_for_fix_a': element[6],
                'if_single_env': element[7], 'assigned_patient_id': element[8],'total_used_num':element[9],
                'if_start_state_restrict':element[10], 'start_state_id':element[11], 'if_meal_restrict':element[12],
                'arg_total_r_thre': element[13],'arg_generate_train_trace_num':element[14],
                'total_test_trace_num': element[15], 'arg_min_cgm':element[16],'log_interval':element[17], 'arg_total_trial_num':element[18],
                'arg_reward_weight': element[19], 'arg_test_param': element[20],
                'arg_exp_type': element[21],'arg_ppo_train_time':element[22], 'arg_use_exist_trace':element[23], 'arg_exist_trace_id':element[24],
                'arg_if_constrain_on_s':element[25], 'arg_thre_for_s':element[26], 'arg_s_index':element[27],
                'arg_if_use_user_input':element[28], 'arg_user_input_action':element[29],
                'exp_id':element[30]}
        df = df.append(dict, ignore_index=True)
    
command = ['awk', 'BEGIN {FS=OFS=","} {gsub(/,/, "\t"); print}', r'yourFolder\temp_experiments.txt']
output_file = r"yourFolder\diabetic_experiments_exp3_state_RP3_UET6.txt"
output_file_1 = r"yourFolder\diabetic_experiments_exp3_state_RP3_UET6.csv"
df.to_csv(output_file_1)
subprocess.run(command, stdout=open(output_file, 'w'), text=True)
del_command = ['rm', r'yourFolder\temp_experiments.txt']
subprocess.run(del_command, stdout=subprocess.PIPE, text=True)

