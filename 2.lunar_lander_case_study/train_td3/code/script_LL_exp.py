import subprocess
from itertools import product
import sys
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

exp_id = list(range(1, 8))
arg_wts = [100]
train_split = [0.5] #!!
total_used_num = [8] #!!
train_round = [4000] #!!
total_timesteps_each_trace = [1]
log_interval = [5]
gradient_steps = [20]
arg_generate_train_trace_num = [1]
total_test_trace_num = [10]
learning_rate = [0.00001]

if_single_env = [0] #!!
assigned_gravity = [-9.0]

arg_total_trial_num = [1]
arg_reward_weight = [-1.0]
arg_test_param = [20]
arg_ppo_train_time = [3] #!!! EXP 3: 3, EXP1: 5
arg_use_exist_trace = [1]
arg_exist_trace_id = [6] #!!! RP2-EXP1 UET id: 1,2,3,11, RP2-EXP3 UET-rp1 id: 1,6,, RP2-EXP3 UET-rp2 id: 7
#RP2
arg_if_constrain_on_s = [1]
arg_thre_for_s = [0.18, 0.13, 0.08]
arg_s_index = [2]
thre_for_fix_a = [-99]
#RP3
arg_if_use_user_input = [1] #!!!
arg_user_input_action = ['0_0', '0.1_0', '0.3_0'] #!!! '0,-0.6', '0,0.6'
arg_exp_type = [3] #!!!


df = pd.DataFrame()
total_comb = list(product(arg_wts, train_split, train_round, total_timesteps_each_trace, gradient_steps, learning_rate, thre_for_fix_a,
                            if_single_env, assigned_gravity, total_used_num, arg_generate_train_trace_num,
                          total_test_trace_num, log_interval, arg_total_trial_num, arg_reward_weight, arg_test_param,
                          arg_exp_type, arg_ppo_train_time, arg_use_exist_trace, arg_exist_trace_id,
                          arg_if_constrain_on_s, arg_thre_for_s, arg_s_index, arg_if_use_user_input, arg_user_input_action, exp_id))
with open(r'Project\Counterfactual_Explanation\code\OpenAI_example\lunar_lander\slurm\temp_experiments.txt', "w") as f:
    f.write("TaskID, arg_wts, train_split, train_round, total_timesteps_each_trace, gradient_steps, learning_rate, thre_for_fix_a,\
                if_single_env, assigned_gravity, total_used_num, arg_generate_train_trace_num, total_test_trace_num, log_interval, "
            "arg_total_trial_num, arg_reward_weight, arg_test_param, arg_exp_type, arg_ppo_train_time, "
            "arg_use_exist_trace, arg_exist_trace_id, arg_if_constrain_on_s, arg_thre_for_s, arg_s_index, arg_if_use_user_input, arg_user_input_action, exp_id\n")
    idx = 0
    for element in total_comb:
        idx += 1
        f.write(f"{idx},{element[0]},{element[1]},{element[2]},{element[3]},{element[4]},{element[5]},{element[6]},\
                            {element[7]},{element[8]},{element[9]},{element[10]},{element[11]},\
                {element[12]},{element[13]}, {element[14]}, {element[15]},{element[16]},{element[17]},{element[18]},"
                f"{element[19]},{element[20]},{element[21]},{element[22]},{element[23]},{element[24]},{element[25]}\n")
        dict = {'TaskID': idx, 'arg_wts': element[0], 'train_split': element[1], 'train_round': element[2],
                'total_timesteps_each_trace': element[3], 'gradient_steps': element[4],
                'learning_rate': element[5], 'thre_for_fix_a': element[6],
                'if_single_env': element[7], 'assigned_gravity': element[8],'total_used_num':element[9],
                'arg_generate_train_trace_num':element[10],
                'total_test_trace_num': element[11], 'log_interval':element[12],'arg_total_trial_num':element[13],
                'arg_reward_weight':element[14],'arg_test_param':element[15], 'arg_exp_type':element[16], 'arg_ppo_train_time':element[17],
                'arg_use_exist_trace':element[18], 'arg_exist_trace_id':element[19],
                'arg_if_constrain_on_s':element[20], 'arg_thre_for_s':element[21], 'arg_s_index':element[22],
                'arg_if_use_user_input': element[23], 'arg_user_input_action': element[24],
                'exp_id':element[25]}
        df = df.append(dict, ignore_index=True)
    
command = ['awk', 'BEGIN {FS=OFS=","} {gsub(/,/, "\t"); print}', r'Project\Counterfactual_Explanation\code\OpenAI_example\lunar_lander\slurm\temp_experiments.txt']
output_file = r"Project\Counterfactual_Explanation\code\OpenAI_example\lunar_lander\slurm\LL_experiments_exp3_RP3_state_UET6.txt"
output_file_1 = r"Project\Counterfactual_Explanation\code\OpenAI_example\lunar_lander\slurm\LL_experiments_exp3_RP3_state_UET6.csv"
df.to_csv(output_file_1)
subprocess.run(command, stdout=open(output_file, 'w'), text=True)
del_command = ['rm', r'Project\Counterfactual_Explanation\code\OpenAI_example\lunar_lander\slurm\temp_experiments.txt']
subprocess.run(del_command, stdout=subprocess.PIPE, text=True)

