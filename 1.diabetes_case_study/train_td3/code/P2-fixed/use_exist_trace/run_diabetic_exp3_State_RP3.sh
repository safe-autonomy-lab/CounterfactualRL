#!/bin/bash
#SBATCH --job-name=diabetic_exp3_RP3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --array=1-42
#SBATCH --partition=gpu
#SBATCH --error="/p/citypm/AIFairness/Counterfactual_Explanation/diabetic_example/slurm/error_%A_%a.err"
#SBATCH --output="/p/citypm/AIFairness/Counterfactual_Explanation/diabetic_example/slurm/output_%A_%a.output"
config=/p/citypm/AIFairness/Counterfactual_Explanation/diabetic_experiments_exp3_state_RP3_UET6.txt
task_id=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $1}' $config)
arg_wts=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
train_split=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
train_round=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
total_timesteps_each_trace=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
gradient_steps=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
learning_rate=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)
thre_for_fix_a=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)
if_single_env=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $9}' $config)
assigned_patient_id=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $10}' $config)
total_used_num=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $11}' $config)
if_start_state_restrict=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $12}' $config)
start_state_id=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $13}' $config)
if_meal_restrict=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $14}' $config)
arg_total_r_thre=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $15}' $config)
arg_generate_train_trace_num=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $16}' $config)
arg_total_test_trace_num=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $17}' $config)
arg_min_cgm=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $18}' $config)
arg_log_interval=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $19}' $config)
arg_total_trial_num=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $20}' $config)
arg_reward_weight=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $21}' $config)
arg_test_param=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $22}' $config)
arg_exp_type=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $23}' $config)
arg_ppo_train_time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $24}' $config)
arg_use_exist_trace=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $25}' $config)
arg_exist_trace_id=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $26}' $config)
arg_if_constrain_on_s=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $27}' $config)
arg_thre_for_s=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $28}' $config)
arg_s_index=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $29}' $config)
arg_if_use_user_input=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $30}' $config)
arg_user_input_action=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $31}' $config)
python diabetic_case_cf_server_fix_all_trace_sb3_fix_action_step_user_input.py --arg_exp_id ${task_id} --arg_whole_trace_step ${arg_wts} --arg_train_split ${train_split} --arg_total_used_num ${total_used_num} --arg_train_round ${train_round} --arg_total_timesteps_each_trace ${total_timesteps_each_trace} --arg_gradient_steps ${gradient_steps} --arg_learning_rate ${learning_rate} --arg_thre_for_fix_a ${thre_for_fix_a} --arg_if_single_env ${if_single_env} --arg_assigned_patient_id ${assigned_patient_id} --arg_if_start_state_restrict ${if_start_state_restrict} --arg_start_state_id ${start_state_id} --arg_if_meal_restrict ${if_meal_restrict} --arg_total_r_thre ${arg_total_r_thre} --arg_generate_train_trace_num ${arg_generate_train_trace_num} --arg_total_test_trace_num ${arg_total_test_trace_num} --arg_log_interval ${arg_log_interval} --arg_min_cgm ${arg_min_cgm} --arg_total_trial_num ${arg_total_trial_num} --arg_reward_weight ${arg_reward_weight} --arg_test_param ${arg_test_param}  --arg_exp_type ${arg_exp_type} --arg_ppo_train_time ${arg_ppo_train_time} --arg_use_exist_trace ${arg_use_exist_trace} --arg_exist_trace_id ${arg_exist_trace_id} --arg_if_constrain_on_s ${arg_if_constrain_on_s} --arg_thre_for_s ${arg_thre_for_s} --arg_s_index ${arg_s_index} --arg_if_use_user_input ${arg_if_use_user_input} --arg_user_input_action ${arg_user_input_action}