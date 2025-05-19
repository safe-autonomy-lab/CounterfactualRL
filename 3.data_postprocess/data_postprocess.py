from typing import List, Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import torch
#pd.set_option('display.max_rows', None)
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('New folder ok.')
    else:
        print('There is this folder')

    return

def get_result_each_round(result_df, distance_func, total_train_round_list, type='baseline'):
    # for 1 trace segment, get the results at each round of training
    # for draw training & testing performance
    lowest_distance_list = []
    best_result_for_lowest_distance_list = []
    best_outcome_list = []
    lowest_dist_for_best_outcome_list = []
    if type=='baseline' or type=='ddpg':
        if len(result_df)!=0:
            J0 = result_df['orig_accumulated_reward'].tolist()[0]
            train_round_list = list(set(result_df['train_round'].tolist()))
            #train_round_list = list(set(result_df['trained_time_step'].tolist()))
            train_round_list.sort()
            #print('train_round_list: ', len(train_round_list), len(total_train_round_list))
            for r in total_train_round_list:
                if r in train_round_list:
                    #print('Is in train round list: ', r)
                    result_df_this_round = result_df[result_df['train_round'] == r]
                    # result_df_this_round = result_df[result_df['trained_time_step'] == r]
                    result_df_with_better_outcome = result_df_this_round[result_df_this_round['difference'] > 0]
                    if len(result_df_with_better_outcome) > 0:
                        lowest_distance = min(result_df_this_round[distance_func].tolist())
                        best_result_for_lowest_distance = max(
                            result_df_this_round[result_df_this_round[distance_func] == lowest_distance][
                                'difference'].tolist())
                        best_outcome = max(result_df_this_round['difference'].tolist())
                        lowest_dist_for_best_outcome = min(
                            result_df_this_round[result_df_this_round['difference'] == best_outcome][
                                distance_func].tolist())
                    else:
                        lowest_distance, best_result_for_lowest_distance, best_outcome, lowest_dist_for_best_outcome = 99, -99, -99, 99
                else: # in some train round, ddpg cannot generate 20-step cf trace for some orig traces, thus mark them as no better cf in this round
                    #print('Not in train round list: ', r)
                    lowest_distance, best_result_for_lowest_distance, best_outcome, lowest_dist_for_best_outcome = 99, -99, -99, 99
                # print(lowest_distance, best_result_for_lowest_distance, best_outcome, lowest_dist_for_best_outcome)
                lowest_distance_list.append(lowest_distance)
                best_result_for_lowest_distance_list.append(round(best_result_for_lowest_distance, 4))
                best_outcome_list.append(best_outcome)
                lowest_dist_for_best_outcome_list.append(lowest_dist_for_best_outcome)
            train_round_list = total_train_round_list
            #print(len(train_round_list), len(lowest_distance_list))
        else:
            train_round_list, lowest_distance_list, best_result_for_lowest_distance_list,\
            best_outcome_list, lowest_dist_for_best_outcome_list, J0 = None, None, None, None, None, None
    else:
        #print('Baseline')
        result_df_with_better_outcome = result_df[result_df['difference'] > 0]
        if len(result_df_with_better_outcome) > 0:
            lowest_distance = min(result_df[distance_func].tolist())
            best_result_for_lowest_distance = max(result_df[result_df[distance_func] == lowest_distance]['difference'].tolist())
            best_outcome = max(result_df['difference'].tolist())
            lowest_dist_for_best_outcome = min(result_df[result_df['difference'] == best_outcome][distance_func].tolist())
        else:
            lowest_distance, best_result_for_lowest_distance, best_outcome, lowest_dist_for_best_outcome = 99, -99, -99, 99
        lowest_distance_list.append(lowest_distance)
        best_result_for_lowest_distance_list.append(round(best_result_for_lowest_distance,4))
        best_outcome_list.append(best_outcome)
        lowest_dist_for_best_outcome_list.append(lowest_dist_for_best_outcome)
        train_round_list = []
        J0 = result_df['orig_accumulated_reward'].tolist()[0]
    return [train_round_list,lowest_distance_list, best_result_for_lowest_distance_list,
            best_outcome_list,lowest_dist_for_best_outcome_list, J0]

def draw_result_train(trace_eps, trace_current_step, train_result_list,
                      figure_folder, case_name='diabetic',patient_type=None, patient_id=None):
    #mkdir(figure_folder)

    train_round_timestep_list_train, lowest_distance_list_train, best_result_for_lowest_distance_list_train, \
        best_outcome_list_train,lowest_dist_for_best_outcome_list_train, J0 = train_result_list
    x_train = range(1, len(train_round_timestep_list_train)+1)
    # if case_name=='diabetic':
    #     start_id = 2
    # else:
    #     start_id = 1
    x_tick = range(1, len(train_round_timestep_list_train) + 1)
    # print('best_result_for_lowest_distance_list_train: ', best_result_for_lowest_distance_list_train)
    # print('best_result_for_lowest_distance_list_test: ', best_result_for_lowest_distance_list_test)
    fig = plt.figure(figsize=(10, 12))
    plt.subplots_adjust(left=None, bottom=None, top=None, right=None, hspace=0.3, wspace=None)
    # plot 1:
    plt.subplot(2, 1, 1)
    plt.ticklabel_format(style='plain')
    plt.plot(x_train, lowest_distance_list_train, color='g', linewidth=1.0, marker='o',
             label='lowest_distance_DDPG_train_{trace_eps}_{trace_current_step}'.format(
                 trace_eps=trace_eps,
                 trace_current_step=trace_current_step), alpha=.8)
    if case_name=='diabetic':
        plt.title('Lowest_action_trace_distance_each_round_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps,
                                                    trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id))
    else:
        plt.title('Lowest_action_trace_distance_each_round_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps,trace_current_step=trace_current_step))

    #plt.title('Lowest_action_trace_distance_each_round_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps,trace_current_step=trace_current_step))
    plt.ylabel('Value')
    plt.xticks(x_tick)
    plt.xlabel('Train round (J0 = {j0})'.format(j0=round(J0,4)))
    plt.legend()

    # plot 2:
    plt.subplot(2, 1, 2)
    plt.ticklabel_format(style='plain')
    plt.plot(x_train, best_result_for_lowest_distance_list_train, color='g', linewidth=1.0, marker='o',
             label='best_accumulated_reward_difference_DDPG_train_{trace_eps}_{trace_current_step}'.format(
                 trace_eps=trace_eps,
                 trace_current_step=trace_current_step), alpha=.6)
    if case_name == 'diabetic':
        plt.title('Best_accumulated_reward_difference_for_lowest_distance_each_round_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps,
                                                                      trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id))
    else:
        plt.title('Best_accumulated_reward_difference_for_lowest_distance_each_round_{trace_eps}_{trace_current_step}'.format(
                trace_eps=trace_eps,
                trace_current_step=trace_current_step))
    # plt.title('Best_accumulated_reward_difference_for_lowest_distance_each_round_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps,
    #                                                                   trace_current_step=trace_current_step))
    plt.ylabel('Value')
    plt.xticks(x_tick)
    plt.xlabel('Train round (J0 = {j0})'.format(j0=round(J0,4)))
    plt.legend()
    if case_name == 'diabetic':
        file_path = '{folder}/train_result_for_each_round_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}.jpg'.format(folder=figure_folder,
                                                                                                                                  trace_eps=trace_eps,
            trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id)
    else:
        file_path = '{folder}/train_result_for_each_round_{trace_eps}_{trace_current_step}.jpg'.format(
            folder=figure_folder, trace_eps=trace_eps,
            trace_current_step=trace_current_step)
    # file_path = '{folder}/train_result_for_each_round_{trace_eps}_{trace_current_step}.jpg'.format(folder=figure_folder, trace_eps=trace_eps,
    #     trace_current_step=trace_current_step)
    plt.savefig(file_path)
    plt.close()
    return

def draw_result_test(trace_eps, trace_current_step, test_result_list, baseline_results_list,
                     figure_folder, case_name='diabetic',patient_type=None, patient_id=None, gravity=None):
    #mkdir(figure_folder)

    train_round_timestep_list_test, lowest_distance_list_test, best_result_for_lowest_distance_list_test, \
        best_outcome_list_test,lowest_dist_for_best_outcome_list_test, J0 = test_result_list
    train_round_timestep_list_baseline, lowest_distance_list_baseline, best_result_for_lowest_distance_list_baseline, \
        best_outcome_list_baseline, lowest_dist_for_best_outcome_list_baseline, J0 = baseline_results_list
    # train_round_timestep_list_baseline = train_round_timestep_list_test
    # lowest_distance_list_baseline = lowest_distance_list_baseline*len(train_round_timestep_list_test)
    # best_result_for_lowest_distance_list_baseline = best_result_for_lowest_distance_list_baseline * len(train_round_timestep_list_test)
    # best_outcome_list_baseline = best_outcome_list_baseline * len(train_round_timestep_list_test)
    # lowest_dist_for_best_outcome_list_baseline = lowest_dist_for_best_outcome_list_baseline * len(train_round_timestep_list_test)
    # if case_name=='diabetic':
    #     start_id = 2
    # else:
    #     start_id = 1
    start_id = 1
    x_test = range(start_id, len(train_round_timestep_list_test)+start_id)
    x_baseline = range(start_id, len(train_round_timestep_list_baseline) + start_id)
    x_tick = range(1, len(train_round_timestep_list_baseline) + 1)
    # print('best_result_for_lowest_distance_list_train: ', best_result_for_lowest_distance_list_train)
    # print('best_result_for_lowest_distance_list_test: ', best_result_for_lowest_distance_list_test)
    fig = plt.figure(figsize=(10, 12))
    plt.subplots_adjust(left=None, bottom=None, top=None, right=None, hspace=0.3, wspace=None)
    # plot 1:
    plt.subplot(2, 1, 1)
    plt.ticklabel_format(style='plain')
    plt.plot(x_test, lowest_distance_list_test, color='b', linewidth=1.0,
             marker='+',
             label='lowest_distance_DDPG_test_{trace_eps}_{trace_current_step}'.format(
                 trace_eps=trace_eps,
                 trace_current_step=trace_current_step), alpha=.8)
    plt.plot(x_baseline, lowest_distance_list_baseline, color='r', linewidth=1.0,
             marker='^',
             label='lowest_distance_Baseline_{trace_eps}_{trace_current_step}'.format(
                 trace_eps=trace_eps,
                 trace_current_step=trace_current_step), alpha=.8)
    if case_name=='diabetic':
        plt.title('Lowest_action_trace_distance_each_round_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps,
                                                    trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id))
    else:
        plt.title('Lowest_action_trace_distance_each_round_g_{gravity}_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps,
                                                                                            trace_current_step=trace_current_step,gravity=gravity))

    plt.ylabel('Value')
    plt.xticks(x_tick)
    plt.xlabel('Train round (J0 = {j0})'.format(j0=round(J0,4)))
    plt.legend()

    # plot 2:
    plt.subplot(2, 1, 2)
    plt.ticklabel_format(style='plain')
    plt.plot(x_test, best_result_for_lowest_distance_list_test, color='b', linewidth=1.0,
             marker='+',
             label='best_accumulated_reward_difference_DDPG_test_{trace_eps}_{trace_current_step}'.format(
                 trace_eps=trace_eps,
                 trace_current_step=trace_current_step), alpha=.6)
    plt.plot(x_baseline, best_result_for_lowest_distance_list_baseline, color='r', linewidth=1.0,
             marker='^',
             label='best_accumulated_reward_difference_Baseline_{trace_eps}_{trace_current_step}'.format(
                 trace_eps=trace_eps,
                 trace_current_step=trace_current_step), alpha=.6)
    if case_name == 'diabetic':
        plt.title('Best_accumulated_reward_difference_for_lowest_distance_each_round_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps,
                                                                      trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id))
    else:
        plt.title('Best_accumulated_reward_difference_for_lowest_distance_each_round_{trace_eps}_{trace_current_step}'.format(
                trace_eps=trace_eps,
                trace_current_step=trace_current_step))

    plt.ylabel('Value')
    plt.xticks(x_tick)
    plt.xlabel('Train round (J0 = {j0})'.format(j0=round(J0,4)))
    plt.legend()
    if case_name == 'diabetic':
        file_path = '{folder}/test_result_for_each_round_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}.jpg'.format(folder=figure_folder,
                                                                                                                                  trace_eps=trace_eps,
            trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id)
    else:
        file_path = '{folder}/test_result_for_each_round_{trace_eps}_{trace_current_step}.jpg'.format(
            folder=figure_folder, trace_eps=trace_eps,
            trace_current_step=trace_current_step)
    plt.savefig(file_path)
    plt.close()
    return

def draw_all_test_trace_result_normalized(max_train_round_index, all_test_trace_lowest_distance_each_round_dict,
                                          all_test_trace_best_outcome_for_lowest_distance_each_round_dict,figure_folder_path,round_to_step_convert):
    # get the lowest distance and outcome for lowest distance for the trace that can be found with a better CF, normalize them and draw the mean, std
    train_round_list = np.arange(0.5, max_train_round_index+1.5, 0.5)#np.arange(1, max_train_round_index+1)
    train_step_list = [x*round_to_step_convert for x in train_round_list]
    normalized_lowest_distance_mean_each_round_list = []
    normalized_lowest_distance_std_each_round_list = []
    normalized_outcome_mean_each_round_list = []
    normalized_outcome_std_each_round_list = []
    #for r in range(1, max_train_round_index+1):
    for r in np.arange(0.5, max_train_round_index + 1.5, 0.5):
        normalized_lowest_dist_list = []
        normalized_outcome_list = []
        all_test_trace_lowest_distance_this_round_list = all_test_trace_lowest_distance_each_round_dict[r]
        all_test_trace_best_outcome_for_lowest_distance_this_round_list = all_test_trace_best_outcome_for_lowest_distance_each_round_dict[r]
        max_low_dist = max(all_test_trace_lowest_distance_this_round_list)
        min_low_dist = min(all_test_trace_lowest_distance_this_round_list)
        max_outcome = max(all_test_trace_best_outcome_for_lowest_distance_this_round_list)
        min_outcome = min(all_test_trace_best_outcome_for_lowest_distance_this_round_list)
        for i in all_test_trace_lowest_distance_this_round_list:
            normalized_lowest_dist_list.append((i-min_low_dist)/(max_low_dist-min_low_dist))
        for j in all_test_trace_best_outcome_for_lowest_distance_this_round_list:
            normalized_outcome_list.append((j-min_outcome)/(max_outcome-min_outcome))
        normalized_lowest_distance_mean_each_round_list.append(np.mean(normalized_lowest_dist_list))
        normalized_lowest_distance_std_each_round_list.append(np.std(normalized_lowest_dist_list, ddof=1))
        normalized_outcome_mean_each_round_list.append(np.mean(normalized_outcome_list))
        normalized_outcome_std_each_round_list.append(np.std(normalized_outcome_list, ddof=1))
        # print('r: ', r)
        # print('all_test_trace_lowest_distance_this_round_list: ', all_test_trace_lowest_distance_this_round_list)
        # print('all_test_trace_best_outcome_for_lowest_distance_this_round_list: ',all_test_trace_best_outcome_for_lowest_distance_this_round_list)
        # print('normalized_lowest_distance_mean_each_round_list: ', normalized_lowest_distance_mean_each_round_list)
        # print('normalized_lowest_distance_std_each_round_list: ', normalized_lowest_distance_std_each_round_list)
        # print('normalized_outcome_mean_each_round_list: ', normalized_outcome_mean_each_round_list)
        # print('normalized_outcome_std_each_round_list: ', normalized_outcome_std_each_round_list)
    fig = plt.figure(figsize=(10, 12))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(train_step_list, normalized_lowest_distance_mean_each_round_list, color='b', linewidth=1.0,
             label='normalized_lowest_distance_DDPG_test', alpha=.8)
    r1 = list(map(lambda x: x[0]-x[1], zip(normalized_lowest_distance_mean_each_round_list, normalized_lowest_distance_std_each_round_list)))
    r2 = list(map(lambda x: x[0] + x[1],zip(normalized_lowest_distance_mean_each_round_list, normalized_lowest_distance_std_each_round_list)))
    ax1.fill_between(train_step_list, r1, r2, color='b', alpha=0.2)
    ax1.legend()
    ax1.set_title('Normalized_lowest_distance_DDPG_test_set')
    ax1.set_xticks(train_step_list)
    ax1.set_xlabel('Train step')
    ax1.set_ylabel('Value')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(train_step_list, normalized_outcome_mean_each_round_list, color='g', linewidth=1.0,
             label='normalized_best_total_reward_difference_for_lowest_distance_DDPG_test', alpha=.8)
    r1 = list(map(lambda x: x[0] - x[1],
                  zip(normalized_outcome_mean_each_round_list, normalized_outcome_std_each_round_list)))
    r2 = list(map(lambda x: x[0] + x[1],
                  zip(normalized_outcome_mean_each_round_list, normalized_outcome_std_each_round_list)))
    ax2.fill_between(train_step_list, r1, r2, color='g', alpha=0.2)
    ax2.legend()
    ax2.set_title('Normalized_best_total_reward_difference_for_lowest_distance_DDPG_test_set')
    ax2.set_xticks(train_step_list)
    ax2.set_xlabel('Train step')
    ax2.set_ylabel('Value')
    plt.savefig('{figure_folder_path}/test_performance_each_round_normalize.png'.format(figure_folder_path=figure_folder_path))
    plt.close()

    return
def result_summary_each_segment(result_df, distance_func, result_type='ddpg'):
    # get the lowest distance value and corresponding accumulated reward among all test rounds
    # get the best outcome value and corresponding distance among all test rounds
    #print(len(result_df['orig_accumulated_reward']))
    #print(result_df)
    if len(result_df) !=0:
        #result_df = result_df[result_df['train_round']<=max_train_round]
        orig_accumulated_reward = result_df['orig_accumulated_reward'].tolist()[0]
        all_traces_with_better_result_df = result_df[result_df['difference']>0]
        if len(all_traces_with_better_result_df)>0:
            lowest_distance_all = min(all_traces_with_better_result_df[distance_func].tolist())
            best_outcome_for_ld = max(all_traces_with_better_result_df[all_traces_with_better_result_df[distance_func]==lowest_distance_all]['difference'].tolist())
            improvement_for_lowest_dist = best_outcome_for_ld/abs(orig_accumulated_reward)
            if result_type=='ddpg':
                earlist_train_round_for_good_cf = min(all_traces_with_better_result_df[(all_traces_with_better_result_df[distance_func]==lowest_distance_all) &
                                                        (all_traces_with_better_result_df['difference']==best_outcome_for_ld)]['train_round'].tolist())
                #earlist_train_round_for_good_cf += 1 # orig value start from 0
            else:
                earlist_train_round_for_good_cf = 9999
            best_outcome_all = max(all_traces_with_better_result_df['difference'].tolist())
            lowest_distance_for_bo = min(all_traces_with_better_result_df[all_traces_with_better_result_df['difference'] == best_outcome_all][distance_func].tolist())
            best_improvement_percentage = best_outcome_all/abs(orig_accumulated_reward)
        else:
            lowest_distance_all, best_outcome_for_ld, best_outcome_all, lowest_distance_for_bo, \
                best_improvement_percentage, improvement_for_lowest_dist, earlist_train_round_for_good_cf = 9999, -9999, -9999, 9999, -9999, -9999, 9999
        #result_list = [lowest_distance_all, best_outcome_for_ld, best_outcome_all, lowest_distance_for_bo]
        return [lowest_distance_all, best_outcome_for_ld, best_outcome_all, lowest_distance_for_bo,
                best_improvement_percentage, improvement_for_lowest_dist, earlist_train_round_for_good_cf]
    else:
        return [None, None, None, None, None, None, None]

def result_summary_all_segment(trace_segment_index_list, save_folder, result_folder, figure_folder,
                               distance_func, case_name, trace_type='test', patient_info_list=None, max_train_round=0,
                               gravity_info_list=None,all_train_round_list=None, exp_type=None, exp_id=None,action_threshold=None,test_index_df=None,trial_index=None):
    summary_df = pd.DataFrame(columns=['exp_type', 'exp_id','trial_index','patient_type', 'patient_id','gravity','max_train_round',
                                       'trace_eps','trace_current_step','orig_accumulated_reward','start_state',
                                       'ddpg_lowest_distance','ddpg_best_outcome_for_lowest_distance',
                                       'baseline_lowest_distance', 'baseline_best_outcome_for_lowest_distance',
                                       'ddpg_best_outcome', 'ddpg_lowest_distance_for_best_outcome',
                                       'baseline_best_outcome', 'baseline_lowest_distance_for_best_outcome',
                                       'ddpg_BIP', 'baseline_BIP',
                                       'ddpg_improvement_for_lowest_dist', 'baseline_improvement_for_lowest_dist',
                                       'distance_ratio', 'ddpg_earlist_train_round_with_good_cf'])
    metric_df = pd.DataFrame()
    earlist_training_round_good_cf_summary_df = pd.DataFrame()
    ddpg_better_cf_for_orig_count = 0
    baseline_better_cf_for_orig_count = 0
    ddpg_cf_better_than_baseline_count = 0
    total_orig_trace_num_count = 0
    #print(trace_segment_index_list)
    for i in range(len(trace_segment_index_list)):
        (trace_eps, trace_current_step) = trace_segment_index_list[i]
        if case_name == 'diabetic':
            (patient_type, patient_id) = patient_info_list[i]
            gravity = None
            start_state_this_seg = test_index_df[(test_index_df['orig_episode'] == trace_eps) & (test_index_df['orig_end_time_index'] == trace_current_step)]['start_state_this_seg'].tolist()[0]
        else:
            patient_type, patient_id = None, None
            gravity = gravity_info_list[i]
            start_state_this_seg = None

        if os.path.exists('{result_folder}/td3_cf_results/trial_{trial_index}/test/all_{trace_type}_result.csv'.format(result_folder=result_folder,
                                                                                                                        trace_type=trace_type,trial_index=trial_index)):
            ddpg_result_df_this_segment = pd.read_csv('{result_folder}/td3_cf_results/trial_{trial_index}/test/all_{trace_type}_result.csv'.format(result_folder=result_folder,
                                                                                                                                                    trace_type=trace_type,trial_index=trial_index), index_col=0)
            if case_name=='diabetic':
                env_id = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type, patient_id=patient_id)
                ddpg_result_df_this_segment = ddpg_result_df_this_segment[(ddpg_result_df_this_segment['orig_trace_episode']==trace_eps)
                                                                          & (ddpg_result_df_this_segment['orig_end_step']==trace_current_step)
                                                                            & (ddpg_result_df_this_segment['ENV_ID']==env_id)
                                                                            #&(ddpg_result_df_this_segment['train_round']<=max_train_round)
                                                                            & (ddpg_result_df_this_segment.train_round.isin(all_train_round_list))]
                baseline_result_df_this_segment = pd.read_csv(
                    '{result_folder}/baseline_results/accumulated_reward_test_baseline_trained_0_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}.csv'.format(
                        result_folder=result_folder, patient_type=patient_type, patient_id=patient_id,
                        trace_eps=trace_eps, trace_current_step=trace_current_step), index_col=0)
            else:
                ddpg_result_df_this_segment = ddpg_result_df_this_segment[
                            (ddpg_result_df_this_segment['orig_trace_episode'] == trace_eps)
                            & (ddpg_result_df_this_segment['orig_end_step'] == trace_current_step)
                            #&(ddpg_result_df_this_segment['train_round']<=max_train_round)
                            & (ddpg_result_df_this_segment.train_round.isin(all_train_round_list))
                            &(ddpg_result_df_this_segment['gravity']==gravity)]
                baseline_result_df_this_segment = pd.read_csv('{result_folder}/baseline_results/accumulated_reward_test_baseline_trained_0_g_{gravity}_{trace_eps}_{trace_current_step}.csv'.format(result_folder=result_folder,
                                                                        trace_eps=trace_eps,trace_current_step=trace_current_step,
                                                                        gravity=gravity), index_col=0)
            [lowest_distance_all_ddpg, best_outcome_for_ld_ddpg, best_outcome_all_ddpg,
             lowest_distance_for_bo_ddpg,
             best_improvement_percentage_ddpg, improvement_for_lowest_dist_ddpg, earlist_train_round_for_good_cf_ddpg] = \
                result_summary_each_segment(ddpg_result_df_this_segment, distance_func, result_type='ddpg')
            [lowest_distance_all_baseline, best_outcome_for_ld_baseline, best_outcome_all_baseline,
             lowest_distance_for_bo_baseline,
             best_improvement_percentage_baseline, improvement_for_lowest_dist_baseline, earlist_train_round_for_good_cf_baseline] = \
                result_summary_each_segment(baseline_result_df_this_segment, distance_func,result_type='baseline')
            baseline_result_orig_accumulate_reward = baseline_result_df_this_segment['orig_accumulated_reward'].tolist()[0]
            if (lowest_distance_all_ddpg is not None) and (baseline_result_orig_accumulate_reward>-10000):
                orig_accumulated_reward = ddpg_result_df_this_segment['orig_accumulated_reward'].tolist()[0]
                if lowest_distance_all_ddpg!=9999 and lowest_distance_all_baseline!=9999:
                    if lowest_distance_all_baseline!=0:
                        #distance_decrease_perc = (lowest_distance_all_baseline-lowest_distance_all_ddpg)/lowest_distance_all_baseline
                        distance_ratio = lowest_distance_all_ddpg / lowest_distance_all_baseline
                    else:
                        print('lowest_distance_all_baseline == 0')
                        #distance_decrease_perc = (lowest_distance_all_baseline - lowest_distance_all_ddpg)
                        distance_ratio = lowest_distance_all_ddpg / (lowest_distance_all_baseline + 0.01)
                    #distance_decrease_perc = lowest_distance_all_ddpg / (lowest_distance_all_baseline + 0.01)
                    #distance_decrease_perc = lowest_distance_all_ddpg / lowest_distance_all_baseline
                elif lowest_distance_all_ddpg==9999 and lowest_distance_all_baseline!=9999:
                    distance_ratio = -9999
                elif lowest_distance_all_baseline==9999 and lowest_distance_all_ddpg!=9999:
                    distance_ratio = 9999
                else:
                    distance_ratio = -9999
                result_dict = {'exp_type':exp_type, 'exp_id':exp_id,'trial_index':trial_index,'action_threshold':action_threshold,
                    'patient_type':patient_type, 'patient_id':patient_id,'gravity':gravity,'max_train_round':max_train_round,
                                'trace_eps':trace_eps,
                               'trace_current_step':trace_current_step,
                               'orig_accumulated_reward':orig_accumulated_reward,
                               'start_state':start_state_this_seg,
                               'ddpg_lowest_distance':lowest_distance_all_ddpg,
                               'ddpg_best_outcome_for_lowest_distance':best_outcome_for_ld_ddpg,
                                'baseline_lowest_distance':lowest_distance_all_baseline,
                               'baseline_best_outcome_for_lowest_distance':best_outcome_for_ld_baseline,
                               'ddpg_best_outcome':best_outcome_all_ddpg,
                               'ddpg_lowest_distance_for_best_outcome':lowest_distance_for_bo_ddpg,
                               'ddpg_BIP':best_improvement_percentage_ddpg,
                               'baseline_best_outcome': best_outcome_all_baseline,
                               'baseline_lowest_distance_for_best_outcome': lowest_distance_for_bo_baseline,
                               'baseline_BIP':best_improvement_percentage_baseline,
                               'ddpg_improvement_for_lowest_dist':improvement_for_lowest_dist_ddpg,
                               'baseline_improvement_for_lowest_dist':improvement_for_lowest_dist_baseline,
                               'distance_ratio':distance_ratio,
                               'ddpg_earlist_train_round_with_good_cf':earlist_train_round_for_good_cf_ddpg,
                               }

                summary_df = pd.concat([summary_df, pd.DataFrame([result_dict])], ignore_index=True)
                total_orig_trace_num_count += 1

    ddpg_better_cf_for_orig_list = []
    baseline_better_cf_for_orig_list = []
    ddpg_cf_better_than_baseline_list = []
    metric_name_list = ['lowest_distance', 'best_outcome_for_lowest_distance', 'best_outcome',
                        'lowest_distance_for_best_outcome',
                        'BIP', 'improvement_for_lowest_dist', 'distance_ratio', 'earlist_train_round_with_good_cf']


    for item, row in summary_df.iterrows():
        trace_eps = row['trace_eps']
        trace_current_step = row['trace_current_step']
        if case_name=='diabetic':
            patient_type, patient_id = row['patient_type'], row['patient_id']
            trace_info = '{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps, trace_current_step=trace_current_step,
                                                                                     patient_type=patient_type, patient_id=patient_id)
        else:
            gravity = row['gravity']
            trace_info = 'g_{gravity}_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps,
                                                                                               trace_current_step=trace_current_step,
                                                                                               gravity=gravity)
        ddpg_lowest_distance = row['ddpg_lowest_distance']
        ddpg_best_outcome_for_lowest_distance = row['ddpg_best_outcome_for_lowest_distance']
        baseline_lowest_distance = row['baseline_lowest_distance']
        baseline_best_outcome_for_lowest_distance = row['baseline_best_outcome_for_lowest_distance']
        if ddpg_best_outcome_for_lowest_distance > 0:
            ddpg_better_cf_for_orig_count += 1
            ddpg_better_cf_for_orig_list.append(trace_info)
            #print('ddpg_better_cf_for_orig_count: ', ddpg_better_cf_for_orig_count)
        if baseline_best_outcome_for_lowest_distance > 0:
            baseline_better_cf_for_orig_count += 1
            baseline_better_cf_for_orig_list.append(trace_info)
            #print('baseline_better_cf_for_orig_count: ', baseline_better_cf_for_orig_count)
        if (ddpg_best_outcome_for_lowest_distance>0) and (round(ddpg_best_outcome_for_lowest_distance,4) >= round(baseline_best_outcome_for_lowest_distance,4)) and (
                    round(ddpg_lowest_distance,4) < round(baseline_lowest_distance,4)):
            ddpg_cf_better_than_baseline_count += 1
            ddpg_cf_better_than_baseline_list.append(trace_info)
            #print('ddpg_cf_better_than_baseline_count: ', ddpg_cf_better_than_baseline_count)
    metric_dict = {'exp_type':exp_type, 'exp_id':exp_id,'trial_index':trial_index,'action_threshold':action_threshold,'max_train_round':max_train_round,
                   'total_orig_trace_number':total_orig_trace_num_count,
                        'ddpg_better_cf_for_orig_count':ddpg_better_cf_for_orig_count,
                        'baseline_better_cf_for_orig_count':baseline_better_cf_for_orig_count,
                        'ddpg_cf_better_than_baseline_count':ddpg_cf_better_than_baseline_count,
                       'ddpg_better_cf_for_orig_perc': ddpg_better_cf_for_orig_count/total_orig_trace_num_count,
                       'baseline_better_cf_for_orig_perc': baseline_better_cf_for_orig_count/total_orig_trace_num_count,
                       'ddpg_cf_better_than_baseline_perc': ddpg_cf_better_than_baseline_count/total_orig_trace_num_count,
                       'ddpg_better_cf_for_orig_list':ddpg_better_cf_for_orig_list,
                       'baseline_better_cf_for_orig_list':baseline_better_cf_for_orig_list,
                       'ddpg_cf_better_than_baseline_list':ddpg_cf_better_than_baseline_list}
    print('total_orig_trace_number: ', total_orig_trace_num_count,
                        'ddpg_better_cf_for_orig_count: ', ddpg_better_cf_for_orig_count,
                        'baseline_better_cf_for_orig_count: ', baseline_better_cf_for_orig_count,
                        'ddpg_cf_better_than_baseline_count: ', ddpg_cf_better_than_baseline_count,
                       'ddpg_better_cf_for_orig_perc: ', round(ddpg_better_cf_for_orig_count/total_orig_trace_num_count, 3),
                       'baseline_better_cf_for_orig_perc: ', round(baseline_better_cf_for_orig_count/total_orig_trace_num_count, 3),
                       'ddpg_cf_better_than_baseline_perc: ', round(ddpg_cf_better_than_baseline_count/total_orig_trace_num_count,3))
    all_test_result_df = pd.read_csv('{result_folder}/td3_cf_results/trial_{trial_index}/test/all_{trace_type}_result.csv'.format(result_folder=result_folder,
                                                                                                                                   trace_type=trace_type,trial_index=trial_index))
    for name in metric_name_list:
        ddpg_aveg_value_this_metric, ddpg_std_value_this_metric, baseline_aveg_value_this_metric,baseline_std_value_this_metric,\
            accumulated_trace_perc_dict = \
            draw_result_metric_distribution_all_segment(summary_df, figure_folder, metric_name=name,
                                                        max_train_round=max_train_round,all_train_round_list=all_train_round_list,
                                                        all_test_result_df=all_test_result_df)
        metric_dict['aveg_{name}_ddpg'.format(name=name)] = ddpg_aveg_value_this_metric
        metric_dict['std_{name}_ddpg'.format(name=name)] = ddpg_std_value_this_metric
        metric_dict['aveg_{name}_baseline'.format(name=name)] = baseline_aveg_value_this_metric
        metric_dict['std_{name}_baseline'.format(name=name)] = baseline_std_value_this_metric
        if name=='earlist_train_round_with_good_cf':
            accumulated_trace_perc_dict['exp_type'] = exp_type
            accumulated_trace_perc_dict['exp_id'] = exp_id
            accumulated_trace_perc_dict['action_threshold'] = action_threshold
            earlist_training_round_good_cf_summary_df = pd.concat([earlist_training_round_good_cf_summary_df,
                                                                   pd.DataFrame([accumulated_trace_perc_dict])],
                                                                  ignore_index=True)
            #print('accumulated_trace_perc_dict: ', accumulated_trace_perc_dict)
        #draw_result_metric_boxplot_normalized_all_segment(summary_df, figure_folder, metric_name=name)
    #metric_df = metric_df.append(metric_dict, ignore_index=True)

    metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)
    summary_df.to_csv('{save_folder}/all_{trace_type}_result_summary_{case_name}_trial_{trial_index}.csv'.format(save_folder=save_folder,trace_type=trace_type, case_name=case_name, trial_index=trial_index))
    metric_df.to_csv('{save_folder}/all_{trace_type}_result_metric_{case_name}_trial_{trial_index}.csv'.format(save_folder=save_folder, trace_type=trace_type, case_name=case_name, trial_index=trial_index))
    earlist_training_round_good_cf_summary_df.to_csv('{save_folder}/all_{trace_type}_result_earlist_training_round_good_cf_{case_name}_trial_{trial_index}.csv'.format(save_folder=save_folder,
                                                                                           trace_type=trace_type,
                                                                                           case_name=case_name, trial_index=trial_index))

    return metric_df, earlist_training_round_good_cf_summary_df, summary_df

def draw_result_metric_distribution_all_segment(result_summary_df, figure_folder, max_train_round, all_train_round_list,
                                                metric_name=None, exp_type=None, exp_id=None, all_test_result_df=None):
    # draw the distribution of metrics like distance_lowest, distance_decrease_percentage, total_reward_improvement_percentage
    fig = plt.figure(figsize=(30, 15))
    if metric_name!='distance_ratio' and metric_name!='earlist_train_round_with_good_cf':
        ddpg_metric = 'ddpg_{metric_name}'.format(metric_name=metric_name)
        baseline_metric = 'baseline_{metric_name}'.format(metric_name=metric_name)
        ddpg_data = result_summary_df[(result_summary_df[ddpg_metric]!=-9999) & (result_summary_df[ddpg_metric]!=9999)][ddpg_metric].tolist()
        baseline_data = result_summary_df[(result_summary_df[baseline_metric]!=-9999) & (result_summary_df[baseline_metric]!=9999)][baseline_metric].tolist()
        plt.hist(ddpg_data, bins=30, color='green', alpha=0.7, label='DDPG')
        plt.hist(baseline_data, bins=30, color='red', alpha=0.5, label='Baseline')
        plt.legend()
        plt.title('{metric_name}_distribution'.format(metric_name=metric_name))
        plt.xlabel('Value')
        plt.ylabel('Percentage')
        file_path = '{folder}/test_result_{metric_name}.jpg'.format(folder=figure_folder, metric_name=metric_name)
        plt.savefig(file_path)
        plt.close()
        # #plt.hist(ddpg_data, bins=30, color='green', alpha=0.7, label='DDPG')
        # plt.hist(baseline_data, bins=30, color='red', alpha=0.5, label='Baseline')
        # plt.legend()
        # plt.title('{metric_name}_distribution_baseline'.format(metric_name=metric_name))
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # file_path = '{folder}/test_result_{metric_name}_baseline.jpg'.format(folder=figure_folder, metric_name=metric_name)
        # plt.savefig(file_path)
        # plt.close()
        ddpg_aveg_value_this_metric = np.mean(ddpg_data)
        ddpg_std_value_this_metric = np.std(ddpg_data, ddof=1)
        baseline_aveg_value_this_metric = np.mean(baseline_data)
        baseline_std_value_this_metric = np.std(baseline_data, ddof=1)
        accumulated_trace_perc_dict = None
        #aveg_metric_value = None
    elif metric_name=='distance_ratio':
        data = result_summary_df[(result_summary_df[metric_name]!=-9999) & (result_summary_df[metric_name]!=9999)][metric_name].tolist()
        plt.hist(data, bins=30, color='green', alpha=0.7)
        plt.title('{metric_name}_distribution'.format(metric_name=metric_name))
        plt.xlabel('Value')
        plt.ylabel('Percentage')
        # plt.legend()
        file_path = '{folder}/test_result_{metric_name}.jpg'.format(folder=figure_folder, metric_name=metric_name)
        plt.savefig(file_path)
        plt.close()
        #aveg_metric_value = np.mean(data)
        ddpg_aveg_value_this_metric = np.mean(data)
        ddpg_std_value_this_metric = np.std(data, ddof=1)
        baseline_aveg_value_this_metric = None
        baseline_std_value_this_metric = None
        accumulated_trace_perc_dict = None
    elif metric_name=='earlist_train_round_with_good_cf':
        ddpg_metric = 'ddpg_{metric_name}'.format(metric_name=metric_name)
        total_trace_num = len(result_summary_df)
        ddpg_data = result_summary_df[(result_summary_df[ddpg_metric]!=-9999) & (result_summary_df[ddpg_metric]!=9999)][ddpg_metric].tolist()
        earlist_train_round_cal = []
        accumulated_trace_perc = []
        accumulated_trace_perc_dict = {}
        accumulated_trace_perc_df = pd.DataFrame()
        not_accumulate_perc = []
        for i in all_train_round_list:
            better_cf_df_this_round = all_test_result_df[(all_test_result_df['train_round']==i)&(all_test_result_df['difference']>0)]
            better_cf_df_this_round = better_cf_df_this_round.drop_duplicates(subset=['orig_trace_episode', 'orig_end_step'], keep='first')
            #if len(better_cf_df_this_round)>0:
            not_accumulate_perc.append(len(better_cf_df_this_round)/total_trace_num)
        for i in all_train_round_list:
            count = ddpg_data.count(i)
            #print(i, count, len(ddpg_data))
            earlist_train_round_cal.append(count/total_trace_num)
        #print('earlist_train_round_cal: ', earlist_train_round_cal)
        for j in range(0, len(earlist_train_round_cal)):
            # print(j, j+1, earlist_train_round_cal[:j+1])
            accumulated_trace_perc.append(sum(earlist_train_round_cal[:j+1]))
            accumulated_trace_perc_dict[j] = sum(earlist_train_round_cal[:j+1])

        # accumulated_trace_perc_dict['exp_type'] = exp_type
        # accumulated_trace_perc_dict['exp_id'] = exp_id
        # print('accumulated_trace_perc_dict: ', accumulated_trace_perc_dict)
        #accumulated_trace_perc_df = pd.concat([accumulated_trace_perc_df, pd.DataFrame([accumulated_trace_perc_dict])], ignore_index=True)
        # plt.bar(np.arange(0.5, max_train_round+1.5, 0.5), earlist_train_round_cal, color='green', alpha=0.7,width=0.5,
        #         tick_label=np.arange(0.5, max_train_round+1.5, 0.5))
        # for k in range(len(all_train_round_list)):
        #     all_train_round_list[k] = all_train_round_list[k]+1
        plt.plot(all_train_round_list, accumulated_trace_perc, color='green', alpha=0.7, marker='+', label='accumulated')
        plt.plot(all_train_round_list, not_accumulate_perc, color='red', alpha=0.7, marker='o', label='not accumulated')
        for a, b in zip(all_train_round_list, not_accumulate_perc):
            plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='bottom', fontsize=9)
        #plt.hist(ddpg_data, bins=max_train_round+1, color='green', alpha=0.7)
        plt.title('{metric_name}_perc'.format(metric_name=metric_name))
        plt.xticks(all_train_round_list)
        plt.xlabel('Train round')
        plt.ylabel('Percentage')
        plt.legend()
        file_path = '{folder}/test_result_{metric_name}.jpg'.format(folder=figure_folder, metric_name=metric_name)
        plt.savefig(file_path)
        plt.close()
        #aveg_metric_value = np.mean(ddpg_data)
        ddpg_aveg_value_this_metric = np.mean(ddpg_data)
        ddpg_std_value_this_metric = np.std(ddpg_data, ddof=1)
        baseline_aveg_value_this_metric = None
        baseline_std_value_this_metric = None
    # plt.title('{metric_name}_distribution'.format(metric_name=metric_name))
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # #plt.legend()
    # file_path = '{folder}/test_result_{metric_name}.jpg'.format(folder=figure_folder, metric_name=metric_name)
    # plt.savefig(file_path)
    # plt.close()
    return ddpg_aveg_value_this_metric, ddpg_std_value_this_metric, baseline_aveg_value_this_metric,baseline_std_value_this_metric, accumulated_trace_perc_dict

def draw_result_metric_boxplot_normalized_all_segment(result_summary_df, figure_folder,metric_name=None):
    # draw the distribution of metrics like distance_lowest, distance_decrease_percentage, total_reward_improvement_percentage
    # fig = plt.figure(figsize=(10, 12))
    if metric_name != 'distance_ratio' and metric_name != 'earlist_train_round_with_good_cf':
        ddpg_metric = 'ddpg_{metric_name}'.format(metric_name=metric_name)
        baseline_metric = 'baseline_{metric_name}'.format(metric_name=metric_name)
        ddpg_df = result_summary_df[(result_summary_df[ddpg_metric] != -9999) & (result_summary_df[ddpg_metric] != 9999)][ddpg_metric]
        ddpg_min, ddpg_max = min(ddpg_df.tolist()), max(ddpg_df.tolist())
        #ddpg_df_norm = (ddpg_df-ddpg_df.min())/(ddpg_df.max()-ddpg_df.min())

        baseline_df = result_summary_df[(result_summary_df[baseline_metric] != -9999) & (result_summary_df[baseline_metric] != 9999)][baseline_metric]
        baseline_min, baseline_max = min(baseline_df.tolist()), max(baseline_df.tolist())
        #baseline_df_norm = (baseline_df - baseline_df.min()) / (baseline_df.max() - baseline_df.min())

        all_min, all_max = min(ddpg_min, baseline_min), max(ddpg_max, baseline_max)
        #ddpg_df_norm = ddpg_df.apply(lambda x: (x - all_min) / (all_max - all_min))
        #baseline_df_norm = baseline_df.apply(lambda x: (x - all_min) / (all_max - all_min))
        #ddpg_data = ddpg_df_norm.tolist()
        #baseline_data = baseline_df_norm.tolist()
        ddpg_data = ddpg_df.tolist()
        baseline_data = baseline_df.tolist()

        #print('ddpg_data: ', ddpg_data)
        #print('ddpg_df_norm: ', ddpg_df_norm)
        x1 = [1.5]
        x2 = [2.5]
        ###
        box1 = plt.boxplot(ddpg_data, positions=x1, patch_artist=True, showmeans=True,
                           boxprops={"facecolor": "C0",
                                     "edgecolor": "grey",
                                     "linewidth": 0.5},
                           medianprops={"color": "k", "linewidth": 0.5},
                           meanprops={'marker': '+',
                                      'markerfacecolor': 'k',
                                      'markeredgecolor': 'k',
                                      'markersize': 5})
        box2 = plt.boxplot(baseline_data, positions=x2, patch_artist=True, showmeans=True,
                           boxprops={"facecolor": "C1",
                                     "edgecolor": "grey",
                                     "linewidth": 0.5},
                           medianprops={"color": "k", "linewidth": 0.5},
                           meanprops={'marker': '+',
                                      'markerfacecolor': 'k',
                                      'markeredgecolor': 'k',
                                      'markersize': 5})

        plt.xticks([1.5, 2.5], ['DDPG', 'Baseline'], fontsize=11)
        #plt.ylim(10, 45)
        plt.title('{metric_name}'.format(metric_name=metric_name))
        #plt.xlabel('Value')
        plt.ylabel('Value')
        plt.grid(axis='y', ls='--', alpha=0.8)
        # 给箱体添加图例，每类箱线图中取第一个颜色块用于代表图例
        plt.legend(handles=[box1['boxes'][0], box2['boxes'][0]], labels=['DDPG', 'Baseline'])
        file_path = '{folder}/test_result_{metric_name}_boxplot.jpg'.format(folder=figure_folder, metric_name=metric_name)
        plt.savefig(file_path)
        #plt.show()
        plt.close()
        # #plt.hist(ddpg_data, bins=30, color='green', alpha=0.7, label='DDPG')
        # plt.hist(baseline_data, bins=30, color='red', alpha=0.5, label='Baseline')
        # plt.legend()
        # plt.title('{metric_name}_distribution_baseline'.format(metric_name=metric_name))
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # file_path = '{folder}/test_result_{metric_name}_baseline.jpg'.format(folder=figure_folder, metric_name=metric_name)
        # plt.savefig(file_path)
        # plt.close()
        ddpg_aveg_value_this_metric = np.mean(ddpg_data)
        baseline_aveg_value_this_metric = np.mean(baseline_data)
        # aveg_metric_value = None
    elif metric_name == 'distance_ratio':

        data = result_summary_df[(result_summary_df[metric_name] != -9999) & (result_summary_df[metric_name] != 9999)][
            metric_name].tolist()
        metric_name = 'distance_ratio'
        x1 = [1.5]
        x2 = [2.5]
        ###
        box1 = plt.boxplot(data, positions=x1, patch_artist=True, showmeans=True,
                           boxprops={"facecolor": "C0",
                                     "edgecolor": "grey",
                                     "linewidth": 0.5},
                           medianprops={"color": "k", "linewidth": 0.5},
                           meanprops={'marker': '+',
                                      'markerfacecolor': 'k',
                                      'markeredgecolor': 'k',
                                      'markersize': 5})
        plt.xticks([1.5], ['DDPG distance decrease percentage'], fontsize=11)
        plt.xticks([1.5], ['DDPG distance ratio'], fontsize=11)
        # plt.ylim(10, 45)
        plt.title('{metric_name}'.format(metric_name=metric_name))
        # plt.xlabel('Value')
        plt.ylabel('Value')
        plt.grid(axis='y', ls='--', alpha=0.8)
        # 给箱体添加图例，每类箱线图中取第一个颜色块用于代表图例
        #plt.legend(handles=[box1['boxes'][0]], labels=['DDPG'])
        file_path = '{folder}/test_result_{metric_name}_boxplot.jpg'.format(folder=figure_folder, metric_name=metric_name)
        plt.savefig(file_path)
        plt.close()
        # aveg_metric_value = np.mean(data)
        ddpg_aveg_value_this_metric = np.mean(data)
        baseline_aveg_value_this_metric = None
    elif metric_name == 'earlist_train_round_with_good_cf':
        pass
        # ddpg_metric = 'ddpg_{metric_name}'.format(metric_name=metric_name)
        # ddpg_data = result_summary_df[(result_summary_df[ddpg_metric] != -9999) & (result_summary_df[ddpg_metric] != 9999)][ddpg_metric].tolist()
        # earlist_train_round_cal = []
        # for i in range(1, max_train_round+1):
        #     count = ddpg_data.count(i)
        #     print(i, count, len(ddpg_data))
        #     earlist_train_round_cal.append(round(count/len(ddpg_data),2))
        # print('earlist_train_round_cal: ', earlist_train_round_cal)
        # plt.bar(range(1, max_train_round+1), earlist_train_round_cal, color='green', alpha=0.7, tick_label=range(1, max_train_round+1))
        # #plt.hist(ddpg_data, bins=max_train_round+1, color='green', alpha=0.7)
        # plt.title('{metric_name}_distribution'.format(metric_name=metric_name))
        # #plt.xticks(np.arange(1.5, max_train_round+2.5), np.arange(1, max_train_round+2))
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # file_path = '{folder}/test_result_{metric_name}.jpg'.format(folder=figure_folder, metric_name=metric_name)
        # plt.savefig(file_path)
        # plt.close()
        # ddpg_aveg_value_this_metric = np.mean(ddpg_data)
        # baseline_aveg_value_this_metric = None

    return

def get_data(file_path, max_train_round, all_train_round_list):
    result_summary_df = pd.read_csv(file_path, index_col=0)
    metric_name = 'earlist_train_round_with_good_cf'
    ddpg_metric = 'ddpg_{metric_name}'.format(metric_name=metric_name)
    ##
    total_trace_num = len(result_summary_df)
    ddpg_data = result_summary_df[(result_summary_df[ddpg_metric] != -9999) & (result_summary_df[ddpg_metric] != 9999)][
        ddpg_metric].tolist()
    earlist_train_round_cal = []
    accumulated_trace_perc = []
    for i in all_train_round_list:
        count = ddpg_data.count(i)
        print(i, count, len(ddpg_data))
        earlist_train_round_cal.append(round(count / total_trace_num, 2))
    for j in range(len(earlist_train_round_cal)):
        accumulated_trace_perc[j] = sum(earlist_train_round_cal[:j + 1])
    for k in range(len(all_train_round_list)):
        all_train_round_list[k] = all_train_round_list[k] + 1
    plt.plot(all_train_round_list, accumulated_trace_perc, color='green', alpha=0.7, marker='+')
    # ##
    # ddpg_data = result_summary_df[(result_summary_df[ddpg_metric] != -9999) & (result_summary_df[ddpg_metric] != 9999)][ddpg_metric].tolist()
    # earlist_train_round_cal = []
    # #all_round_index_list = np.arange(0.5, max_train_round + 1.5, 0.5)
    # for i in all_train_round_list:
    #     count = ddpg_data.count(i)
    #     #print(i, count, len(ddpg_data))
    #     earlist_train_round_cal.append(round(count / len(ddpg_data), 2))

    return all_train_round_list, accumulated_trace_perc

def draw_earlist_training_round_best_cf_all_experiment(max_train_round, all_train_round_list):
    result_file_folder = r'Project\Counterfactual_Explanation\code\diabetic_example\server\result_summary_files'
    multi_patient_multi_trace_with_a_thre_path_1 = '{result_file_folder}/all_test_result_summary_diabetic_mp_mt_wa.csv'.format(result_file_folder=result_file_folder)
    multi_patient_multi_trace_no_a_thre_path_1 = '{result_file_folder}/all_test_result_summary_diabetic_mp_mt_na.csv'.format(result_file_folder=result_file_folder)
    one_patient_multi_trace_with_a_thre_path_p_5 = '{result_file_folder}/all_test_result_summary_diabetic_1p_mt_wa_p5.csv'.format(result_file_folder=result_file_folder)
    one_patient_multi_trace_no_a_thre_path_p_5 = '{result_file_folder}/all_test_result_summary_diabetic_1p_mt_na_p5.csv'.format(result_file_folder=result_file_folder)
    one_patient_multi_trace_with_a_thre_path_p_7 = '{result_file_folder}/all_test_result_summary_diabetic_1p_mt_wa_p7.csv'.format(result_file_folder=result_file_folder)
    one_patient_multi_trace_no_a_thre_path_p_7 = '{result_file_folder}/all_test_result_summary_diabetic_1p_mt_na_p7.csv'.format(result_file_folder=result_file_folder)
    # one_patient_multi_trace_with_a_thre_path_p_9 = ''
    # one_patient_multi_trace_no_a_thre_path_p_9 = ''
    # one_patient_1_trace_with_a_thre_path_p_5 = ''
    # one_patient_1_trace_no_a_thre_path_p_5 = ''
    # one_patient_1_trace_with_a_thre_path_p_7 = ''
    # one_patient_1_trace_no_a_thre_path_p_7 = ''
    # one_patient_1_trace_with_a_thre_path_p_9 = ''
    # one_patient_1_trace_no_a_thre_path_p_9 = ''
    column = ['experiment', 'train_round_index', 'earlist_training_round_freq']
    exp_type_list = ['mp_mt_na', 'mp_mt_wa', '1p_mt_na_p5','1p_mt_wa_p5', '1p_mt_na_p7','1p_mt_wa_p7','1p_mt_na_p9','1p_mt_wa_p9',
              '1p_1t_na_p5','1p_1t_wa_p5', '1p_1t_na_p7','1p_1t_wa_p7','1p_1t_na_p9','1p_1t_wa_p9']
    all_df = pd.DataFrame(columns=column)
    result_summary_dict = {
                            'mp_mt_wa':multi_patient_multi_trace_with_a_thre_path_1,
                           'mp_mt_na':multi_patient_multi_trace_no_a_thre_path_1,
                           # '1p_mt_na_p5':one_patient_multi_trace_no_a_thre_path_p_5,
                           # '1p_mt_wa_p5':one_patient_multi_trace_with_a_thre_path_p_5,
                           # '1p_mt_na_p7':one_patient_multi_trace_no_a_thre_path_p_7,
                           # '1p_mt_wa_p7':one_patient_multi_trace_with_a_thre_path_p_7,
                           # '1p_mt_na_p9':one_patient_multi_trace_no_a_thre_path_p_9,
                           # '1p_mt_wa_p9':one_patient_multi_trace_with_a_thre_path_p_9,
                           #  '1p_1t_na_p5':one_patient_1_trace_no_a_thre_path_p_5,
                           # '1p_1t_wa_p5':one_patient_1_trace_with_a_thre_path_p_5,
                           # '1p_1t_na_p7':one_patient_1_trace_no_a_thre_path_p_7,
                           # '1p_1t_wa_p7':one_patient_1_trace_with_a_thre_path_p_7,
                           # '1p_1t_na_p9':one_patient_1_trace_no_a_thre_path_p_9,
                           # '1p_1t_wa_p9':one_patient_1_trace_with_a_thre_path_p_9
                           }
    for k in result_summary_dict.keys():
        file_path = result_summary_dict[k]
        all_round_index_list, accumulated_trace_perc = get_data(file_path, max_train_round, all_train_round_list)
        dict = {'experiment':k, 'train_round_index':all_round_index_list, 'accumulate_earlist_training_round_freq':accumulated_trace_perc}
        all_df = all_df._append(dict, ignore_index=True)
        plt.plot(all_round_index_list, accumulated_trace_perc, alpha=0.7, label=k)
        # plt.hist(ddpg_data, bins=max_train_round+1, color='green', alpha=0.7)
    plt.title('earlist_train_round_for_good_cf_distribution')
    plt.xticks(np.arange(1.0, max_train_round+1.0, 1.0).tolist())
    plt.xlabel('Train round')
    plt.ylabel('Percentage')
    plt.legend()
    file_path = '{folder}/test_result_earlist_train_round_for_good_cf_distribution_all_exp_mp_mt.jpg'.format(folder=result_file_folder)
    #file_path = '{folder}/test_result_earlist_train_round_for_good_cf_distribution_all_exp_1p_mt.jpg'.format(folder=result_file_folder)
    plt.savefig(file_path)
    plt.show()
    plt.close()


    return

def get_final_result(whole_trace_len, id, encoder, dist_func, dist_func_1, case_name, lr, grad, max_train_round_index,
                     exp_type, action_threshold, trial_index, start_index, interval,xaxis_interval, all_result_folder, baseline_result_folder, baseline_result_exp_id):
    if case_name=='diabetic':
        # result_folder = '{all_result_folder}/all_trace_len_{whole_trace_len}_encoder_{encoder}_{dist_func}_{id}'.format(all_result_folder=all_result_folder,
        #     whole_trace_len=whole_trace_len, dist_func=dist_func, id=id, encoder=encoder)
        result_folder = '{all_result_folder}/all_trace_len_{whole_trace_len}_{dist_func}_lr_{lr}_grad_{grad}_{id}'.format(
            all_result_folder=all_result_folder, encoder=encoder,
            whole_trace_len=whole_trace_len, lr=lr, grad=grad,
            dist_func=dist_func, id=id)
        baseline_result_path = '{baseline_result_folder}/all_trace_len_{whole_trace_len}_{dist_func}_lr_{lr}_grad_{grad}_{id}'.format(
            baseline_result_folder=baseline_result_folder, encoder=encoder,
            whole_trace_len=whole_trace_len, lr=lr, grad=grad,
            dist_func=dist_func, id=baseline_result_exp_id)
    else:
        result_folder = '{all_result_folder}/all_trace_len_{whole_trace_len}_{dist_func}_lr_{lr}_grad_{grad}_{id}'.format(whole_trace_len=whole_trace_len,
                                                                                                                          all_result_folder=all_result_folder,
                                                                                                                            dist_func=dist_func, id=id,lr=lr,grad=grad)
        baseline_result_path = '{baseline_result_folder}/all_trace_len_{whole_trace_len}_{dist_func}_lr_{lr}_grad_{grad}_{id}'.format(whole_trace_len=whole_trace_len,
                                                                                                                          baseline_result_folder=baseline_result_folder,
                                                                                                                            dist_func=dist_func, id=baseline_result_exp_id,lr=lr,grad=grad)

    ddpg_result_folder = '{result_folder}/td3_cf_results/trial_{trial_index}'.format(result_folder=result_folder,trial_index=trial_index)
    # baseline_result_folder = '{result_folder}/baseline_results'.format(result_folder=result_folder,
    #                                                                    whole_trace_len=whole_trace_len,
    #                                                                    dist_func=dist_func, id=id)
    baseline_result_folder = '{baseline_result_path}/td3_cf_results/trial_{trial_index}'.format(baseline_result_path=baseline_result_path,trial_index=trial_index)
    log_file_folder = '{ddpg_result_folder}/logs/train_log'.format(ddpg_result_folder=ddpg_result_folder)
    all_train_index_file_path = '{result_folder}/train_index_file.csv'.format(result_folder=result_folder)
    all_train_index_df = pd.read_csv(all_train_index_file_path, index_col=0).drop_duplicates()
    all_test_index_file_path = '{result_folder}/test_index_file.csv'.format(result_folder=result_folder)
    all_test_index_df = pd.read_csv(all_test_index_file_path, index_col=0).drop_duplicates()
    # all_test_index_df = pd.read_csv(all_train_index_file_path, index_col=0).drop_duplicates()
    figure_folder = '{result_folder}/figures'.format(result_folder=result_folder)
    mkdir(figure_folder)
    get_all_test_results_in_one_file(test_result_folder=ddpg_result_folder, max_train_round=max_train_round_index,
                                     save_folder=ddpg_result_folder, model_type='ddpg', start_index=start_index, interval=interval)
    get_all_test_results_in_one_file(test_result_folder=baseline_result_folder, max_train_round=max_train_round_index,
                                     save_folder=ddpg_result_folder, model_type='baseline', start_index=start_index,
                                     interval=interval)
    print('get_all_test_results_in_one_file')
    draw_test_curve_single_trace(data_folder=ddpg_result_folder, exp_id=id, test_index_df=all_test_index_df,
                                 save_folder=figure_folder, trial_num=trial_index, xaxis_interval=xaxis_interval,
                                 test_result_type='test',case_name=case_name)
    draw_test_curve_single_trace(data_folder=ddpg_result_folder, exp_id=id, test_index_df=all_train_index_df,
                                 save_folder=figure_folder, trial_num=trial_index, xaxis_interval=xaxis_interval,
                                 test_result_type='train',case_name=case_name)
    print('draw_test_curve_single_trace')

    # # PREV CODE
    # # end_index = 33
    # # max_train_round_index = 7  # start from 0.5
    # # round_to_step_convert = 500 * len(all_train_index_df)
    # all_test_trace_lowest_distance_each_round_dict = {}
    # all_test_trace_best_outcome_for_lowest_distance_each_round_dict = {}
    # #all_train_round_list = np.arange(0.5, max_train_round_index+0.5, 0.5).tolist()
    # all_train_round_list = np.arange(start_index, max_train_round_index + 1.0, interval).tolist()
    # print('all_train_round_list: ', all_train_round_list)
    # # for r in np.arange(0, max_train_round_index+1):
    # for r in all_train_round_list:
    #     all_test_trace_lowest_distance_each_round_dict[r] = []
    #     all_test_trace_best_outcome_for_lowest_distance_each_round_dict[r] = []
    # # for getting results when training with multiple traces
    # # draw testing performance
    # for item, row in all_test_index_df[:].iterrows():
    #     if case_name == 'diabetic':
    #         trace_eps, trace_current_step, patient_type, patient_id = row['orig_episode'], row['orig_end_time_index'], \
    #                                                                   row['patient_type'], row['patient_id']
    #         # ddpg_train_traces = '{result_folder}/all_cf_traces_train.csv'.format(result_folder=ddpg_result_folder,
    #         #                                                                      trace_eps=trace_eps,
    #         #                                                                      trace_current_step=trace_current_step)
    #         # ddpg_test_traces = '{result_folder}/all_cf_traces_test.csv'.format(result_folder=ddpg_result_folder,
    #         #                                                                    trace_eps=trace_eps,
    #         #                                                                    trace_current_step=trace_current_step)
    #         # ddpg_train_result = '{result_folder}/all_train_result.csv'.format(result_folder=ddpg_result_folder,
    #         #                                                                   trace_eps=trace_eps,
    #         #                                                                   trace_current_step=trace_current_step)
    #         ddpg_test_result = '{result_folder}/test/all_test_result.csv'.format(result_folder=ddpg_result_folder)
    #         # Diabetic
    #         baseline_traces = '{result_folder}/cf_trace_file_test_baseline_trained_0_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}.csv'.format(
    #             result_folder=baseline_result_folder,
    #             patient_type=patient_type, patient_id=patient_id,
    #             trace_eps=trace_eps, trace_current_step=trace_current_step)
    #         baseline_results = '{result_folder}/accumulated_reward_test_baseline_trained_0_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}.csv'.format(
    #             result_folder=baseline_result_folder,
    #             patient_type=patient_type, patient_id=patient_id,
    #             trace_eps=trace_eps, trace_current_step=trace_current_step)
    #         ddpg_test_result_df = pd.read_csv(ddpg_test_result, index_col=0)
    #         baseline_result_df = pd.read_csv(baseline_results, index_col=0)
    #         # Diabetic
    #         env_id = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type, patient_id=patient_id)
    #         ddpg_test_result_df = ddpg_test_result_df[(ddpg_test_result_df['orig_trace_episode'] == trace_eps)
    #                                                   & (ddpg_test_result_df['orig_end_step'] == trace_current_step)
    #                                                   & (ddpg_test_result_df['ENV_ID'] == env_id)]
    #     else:
    #         trace_eps, trace_current_step, gravity, patient_type, patient_id = row['orig_episode'], row['orig_end_time_index'], row['gravity'], None, None
    #         # ddpg_train_traces = '{result_folder}/all_cf_traces_train.csv'.format(result_folder=ddpg_result_folder,
    #         #                                                                      trace_eps=trace_eps,
    #         #                                                                      trace_current_step=trace_current_step)
    #         # ddpg_test_traces = '{result_folder}/all_cf_traces_test.csv'.format(result_folder=ddpg_result_folder,
    #         #                                                                    trace_eps=trace_eps,
    #         #                                                                    trace_current_step=trace_current_step)
    #         # ddpg_train_result = '{result_folder}/all_train_result.csv'.format(result_folder=ddpg_result_folder,
    #         #                                                                   trace_eps=trace_eps,
    #         #                                                                   trace_current_step=trace_current_step)
    #         ddpg_test_result = '{result_folder}/test/all_test_result_ddpg.csv'.format(result_folder=ddpg_result_folder)
    #         baseline_results = '{result_folder}/test/all_test_result_baseline.csv'.format(result_folder=ddpg_result_folder)
    #         # # Lunar Lander
    #         # baseline_traces = '{result_folder}/cf_trace_file_test_baseline_trained_0_g_{gravity}_{trace_eps}_{trace_current_step}.csv'.format(
    #         #     result_folder=baseline_result_folder, trace_eps=trace_eps, trace_current_step=trace_current_step,
    #         #     gravity=gravity)
    #         # baseline_results = '{result_folder}/accumulated_reward_test_baseline_trained_0_g_{gravity}_{trace_eps}_{trace_current_step}.csv'.format(
    #         #     result_folder=baseline_result_folder, trace_eps=trace_eps, trace_current_step=trace_current_step,
    #         #     gravity=gravity)
    #
    #         ddpg_test_result_df = pd.read_csv(ddpg_test_result, index_col=0)
    #         baseline_test_result_df = pd.read_csv(baseline_results, index_col=0)
    #         # Lunar Lander
    #         ddpg_test_result_df = ddpg_test_result_df[(ddpg_test_result_df['orig_trace_episode'] == trace_eps)
    #                                                   & (ddpg_test_result_df['orig_end_step'] == trace_current_step)
    #                                                   & (ddpg_test_result_df['gravity'] == gravity)]
    #         baseline_test_result_df = baseline_test_result_df[(baseline_test_result_df['orig_trace_episode'] == trace_eps)
    #                                                   & (baseline_test_result_df['orig_end_step'] == trace_current_step)
    #                                                   & (baseline_test_result_df['gravity'] == gravity)]
    #
    #
    #     # # draw figure from train log
    #     # draw_log_data(log_file_folder, baseline_results, id=id, trace_eps=trace_eps, trace_time_index=trace_current_step,
    #     #               all_test_result_df=ddpg_test_result_df, all_train_id_list=all_train_round_list)
    #     ddpg_test_results_list = get_result_each_round(ddpg_test_result_df, dist_func_1, total_train_round_list=all_train_round_list, type='test')
    #     baseline_results_list = get_result_each_round(baseline_test_result_df, dist_func_1, total_train_round_list=all_train_round_list, type='baseline')
    #     if ddpg_test_results_list[0] is not None: # draw for each trace
    #         #print(trace_eps, trace_current_step, len(ddpg_test_results_list[1]))
    #         draw_result_test(trace_eps, trace_current_step, ddpg_test_results_list, baseline_results_list,
    #                          figure_folder, case_name=case_name, patient_type=patient_type, patient_id=patient_id)
    #     if ddpg_test_results_list[0] is not None:  # draw for all trace normalized
    #         #print('ddpg_test_results_list: ', ddpg_test_results_list, ddpg_test_results_list[1], len(ddpg_test_results_list[1]))
    #         # print('all_round_list: ', all_round_list)
    #         for r in all_train_round_list:
    #             # for r in np.arange(0, max_train_round_index + 1, 1):
    #             r_index = all_train_round_list.index(r)
    #             #print(trace_eps, trace_current_step, r, r_index, len(ddpg_test_results_list[1]))
    #             if ddpg_test_results_list[1][r_index] != 99:
    #                 all_test_trace_lowest_distance_each_round_dict[r].append(ddpg_test_results_list[1][r_index])
    #             if ddpg_test_results_list[2][r_index] != -99:
    #                 all_test_trace_best_outcome_for_lowest_distance_each_round_dict[r].append(ddpg_test_results_list[2][r_index])
    #
    # # draw_all_test_trace_result_normalized(max_train_round_index, all_test_trace_lowest_distance_each_round_dict,
    # #                                           all_test_trace_best_outcome_for_lowest_distance_each_round_dict,figure_folder_path=figure_folder,
    # #                                       round_to_step_convert=round_to_step_convert)
    #
    # # get results summary of test set
    # # get result summary
    # if case_name == 'diabetic':
    #     # diabetic case
    #     test_trace_eps_list = all_test_index_df['orig_episode'].tolist()[:]
    #     test_trace_current_step_list = all_test_index_df['orig_end_time_index'].tolist()[:]
    #     test_patient_type_list = all_test_index_df['patient_type'].tolist()[:]
    #     test_patient_id_list = all_test_index_df['patient_id'].tolist()[:]
    #     test_trace_segment_index_list = list(zip(test_trace_eps_list, test_trace_current_step_list))
    #     test_trace_patient_info_list = list(zip(test_patient_type_list, test_patient_id_list))
    #     metric_df, earlist_training_round_good_cf_summary_df, summary_df = result_summary_all_segment(trace_segment_index_list=test_trace_segment_index_list, save_folder=result_folder,
    #                                result_folder=result_folder, figure_folder=figure_folder, distance_func=dist_func_1,
    #                                case_name=case_name,
    #                                trace_type='test', patient_info_list=test_trace_patient_info_list,
    #                                max_train_round=max_train_round_index,
    #                                all_train_round_list=all_train_round_list,
    #                                 exp_type=exp_type, exp_id=id,action_threshold=action_threshold, test_index_df=all_test_index_df, trial_index=trial_index)
    # else:
    #     # lunar lander case
    #     test_trace_eps_list = all_test_index_df['orig_episode'].tolist()[:]
    #     test_trace_current_step_list = all_test_index_df['orig_end_time_index'].tolist()[:]
    #     test_gravity_list = all_test_index_df['gravity'].tolist()[:]
    #     test_trace_segment_index_list = list(zip(test_trace_eps_list, test_trace_current_step_list))
    #     metric_df, earlist_training_round_good_cf_summary_df, summary_df = result_summary_all_segment(trace_segment_index_list=test_trace_segment_index_list, save_folder=result_folder,
    #                                result_folder=result_folder, figure_folder=figure_folder, distance_func=dist_func_1, case_name=case_name,
    #                                trace_type='test', patient_info_list=None,max_train_round=max_train_round_index,gravity_info_list=test_gravity_list,
    #                                all_train_round_list=all_train_round_list,
    #                                 exp_type=exp_type, exp_id=id,action_threshold=action_threshold, test_index_df=all_test_index_df, trial_index=trial_index)
    #
    # #draw_earlist_training_round_best_cf_all_experiment(max_train_round_index, all_train_round_list)
    return #metric_df, earlist_training_round_good_cf_summary_df, summary_df

def get_earlist_training_round_aveg_one_exp_type(earlist_training_round_all_df, exp_type):
    earlist_training_round_all_df = earlist_training_round_all_df[earlist_training_round_all_df['exp_type']==exp_type]
    df = earlist_training_round_all_df[earlist_training_round_all_df.columns.difference(['exp_type', 'exp_id', 'action_threshold'])]
    new_col_name = {}
    for i in df.columns:
        new_col_name[i]=int(i)
    df.rename(columns=new_col_name, inplace=True)
    df_col_mean = df.mean(axis=0)
    df_col_std = df.std(axis=0)
    df_col_mean.sort_index(inplace=True)
    df_col_std.sort_index(inplace=True)
    #print('df_col_mean: ', df_col_mean, ' df_col_std: ', df_col_std)
    new_df = pd.concat([df_col_mean, df_col_std], axis=1)
    new_df.rename(columns={0:'mean_{exp_type}'.format(exp_type=exp_type), 1:'std_{exp_type}'.format(exp_type=exp_type)}, inplace=True)
    #print('new_df: ', new_df)
    return new_df

def draw_earlist_training_round_aveg_all_exp_type(earlist_training_round_all_df, exp_type_list, figure_folder):
    # color_dict = {'multi_env_multi_trace_with_action_thre':'r',
    #            'multi_env_multi_trace_no_action_thre':'g',
    #            # 'one_env_multi_trace_with_action_thre_p5':[1,2,17,18, 19, 20, 21, 22, 23, 24],
    #            # 'one_env_multi_trace_no_action_thre_p5':[1,2,17,18, 19, 20, 21, 22, 23, 24],
    #            # 'one_env_multi_trace_with_action_thre_p7':[1,2,17,18, 19, 20, 21, 22, 23, 24],
    #            # 'one_env_multi_trace_no_action_thre_p7':[1,2,17,18, 19, 20, 21, 22, 23, 24]
    #             }

    # lunar  lander
    color_dict = {
        'multi_env_multi_trace_no_action_thre': 'r',
        'multi_env_multi_trace_with_action_thre': 'g',
        'one_env_multi_trace_with_action_thre_g10': 'b',
        'one_env_multi_trace_no_action_thre_g10': 'y',
        'one_env_multi_trace_with_action_thre_g8': 'purple',
        # 'one_env_multi_trace_no_action_thre_g8':[],
        # 'one_env_multi_trace_with_action_thre_g6':[45, 46, 47],
        # 'one_env_multi_trace_no_action_thre_g6':[1,2,17,18, 19, 20, 21, 22, 23, 24]
    }
    fig = plt.figure(figsize=(20, 10))
    for e_type in exp_type_list:
        color = color_dict[e_type]
        mean_value = earlist_training_round_all_df['mean_{e_type}'.format(e_type=e_type)].tolist()
        std_value = earlist_training_round_all_df['std_{e_type}'.format(e_type=e_type)].tolist()
        r1 = list(map(lambda x: x[0] - x[1], zip(mean_value, std_value)))
        r2 = list(map(lambda x: x[0] + x[1], zip(mean_value, std_value)))
        x_value = range(1, len(mean_value)+1)
        #x_value = np.arange(0.5, max_train_round_index+0.5, 0.5).tolist()
        plt.plot(x_value, mean_value, color=color, label=e_type, alpha=0.8)
        plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
    plt.xlabel('Train round')
    plt.ylabel('Percentage')
    plt.legend()
    plt.title('Aveg_earlist_train_round_all_exp')
    #plt.show()
    file_path = '{folder}/test_result_Aveg_earlist_train_round_all_exp.jpg'.format(folder=figure_folder)
    plt.savefig(file_path)
    plt.close()
    return

def get_all_test_results_in_one_file(test_result_folder, max_train_round, save_folder, model_type='ddpg', start_index=0, interval=1):
    all_test_result_df_list = []
    all_test_statistic_df_list = []
    test_result_type_list = ['test', 'train'] #include the test result on the training set
    for test_result_type in test_result_type_list:
        all_test_result_path = '{save_folder}/{test_result_type}/all_test_result_{model_type}.csv'.format(save_folder=save_folder,test_result_type=test_result_type, model_type=model_type)
        all_test_statistic_path = '{save_folder}/{test_result_type}/all_test_statistic_{model_type}.csv'.format(save_folder=save_folder,test_result_type=test_result_type,
                                                                                            model_type=model_type)
        for r in range(start_index, max_train_round+1, interval):
            ddpg_test_result_path = '{test_result_folder}/{test_result_type}/test_result_{model_type}_r_{r}.pkl'.format(test_result_folder=test_result_folder,
                                                                                                                        test_result_type=test_result_type,
                                                                                                                        r=r,model_type=model_type)
            ddpg_test_statistic_path = '{test_result_folder}/{test_result_type}/test_statistic_{model_type}_r_{r}.pkl'.format(
                test_result_folder=test_result_folder, r=r, test_result_type=test_result_type, model_type=model_type)
            # ddpg_test_csv_result_path = '{test_result_folder}/test/test_result_r_{r}.csv'.format(test_result_folder=test_result_folder, r=r)
            # ddpg_trace_path = '{test_result_folder}/test/cf_traces_test_r_{r}.pkl'.format(test_result_folder=test_result_folder, r=r)
            # ddpg_trace_csv_path = '{test_result_folder}/test/cf_traces_test_r_{r}.csv'.format(test_result_folder=test_result_folder, r=r)
            df_result = pd.read_pickle(ddpg_test_result_path)
            df_statistic = pd.read_pickle(ddpg_test_statistic_path)
            # ddpg_test_result_path = '{test_result_folder}/test/test_result_r_{r}.csv'.format(test_result_folder=test_result_folder, r=r)
            #df.to_csv(ddpg_test_csv_result_path)
            #trace_df = pd.read_pickle(ddpg_trace_path)
            #trace_df.to_csv(ddpg_trace_csv_path)
            all_test_result_df_list.append(df_result)
            all_test_statistic_df_list.append(df_statistic)
        all_test_result_df = pd.concat(all_test_result_df_list)
        all_test_result_df.to_csv(all_test_result_path)
        all_test_statistic_df = pd.concat(all_test_statistic_df_list)
        all_test_statistic_df.to_csv(all_test_statistic_path)
    return

def draw_each_trace_compare(max_train_round_index, whole_trace_len, encoder, dist_func, id, case_name):

    if case_name=='diabetic':
        result_folder = 'Project/Counterfactual_Explanation/code/diabetic_example/server/DDPG_results_slurm/all_trace_len_{whole_trace_len}_encoder_{encoder}_{dist_func}_{id}'.format(
            whole_trace_len=whole_trace_len, dist_func=dist_func, id=id, encoder=encoder)

    else:
        result_folder = 'Project/Counterfactual_Explanation/code/OpenAI_example/lunar_lander/results/all_trace_len_{whole_trace_len}_{dist_func}_lr_{lr}_grad_{grad}_{id}'.format(whole_trace_len=whole_trace_len,
                                                                               lr=lr, grad=grad,
                                                                                dist_func=dist_func, id=id)

    ddpg_trace_folder = '{result_folder}/td3_cf_results'.format(result_folder=result_folder,
                                                                  whole_trace_len=whole_trace_len, dist_func=dist_func,
                                                                  id=id)
    baseline_trace_folder = '{result_folder}/baseline_results'.format(result_folder=result_folder,
                                                                       whole_trace_len=whole_trace_len,
                                                                       dist_func=dist_func, id=id)
    all_train_index_file_path = '{result_folder}/train_index_file.csv'.format(result_folder=result_folder)
    all_train_index_df = pd.read_csv(all_train_index_file_path, index_col=0).drop_duplicates()
    all_test_index_file_path = '{result_folder}/test_index_file.csv'.format(result_folder=result_folder)
    all_test_index_df = pd.read_csv(all_test_index_file_path, index_col=0).drop_duplicates()
    # all_test_index_df['orig_episode'].astype('int8')
    # all_test_index_df.to_csv('{result_folder}/test_index_file_int8.csv'.format(result_folder=result_folder))
    # all_test_index_df = pd.read_csv(all_train_index_file_path, index_col=0).drop_duplicates()
    figure_folder_0 = '{result_folder}/figures'.format(result_folder=result_folder)
    #mkdir(figure_folder)

    for item, row in all_test_index_df.iterrows():
        for r in range(1, max_train_round_index):
            if case_name == 'diabetic':

                trace_eps, trace_current_step, patient_type, patient_id = row['orig_episode'], row['orig_end_time_index'], \
                                                                          row['patient_type'], row['patient_id']
                #print(trace_eps, trace_current_step, patient_type, patient_id)
                figure_folder = '{figure_folder_0}/{trace_eps}_{trace_current_step}'.format(figure_folder_0=figure_folder_0,
                                                                                            trace_eps=trace_eps, trace_current_step=trace_current_step)
                mkdir(figure_folder)
                # TODO: to be removed
                # trace_eps_int_8 = np.int8(trace_eps)
                # print(trace_eps, trace_current_step, r)
                ddpg_test_traces = '{result_folder}/test/cf_traces_test_r_{round}.pkl'.format(result_folder=ddpg_trace_folder,
                                                                                   trace_eps=trace_eps,
                                                                                   trace_current_step=trace_current_step,round=r)
                ddpg_test_result = '{result_folder}/test/all_test_result.csv'.format(result_folder=ddpg_trace_folder)

                # Diabetic
                ##########
                # Diabetic
                baseline_traces = '{result_folder}/cf_trace_file_test_baseline_trained_0_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}.csv'.format(
                    result_folder=baseline_trace_folder,
                    patient_type=patient_type, patient_id=patient_id,
                    trace_eps=trace_eps, trace_current_step=trace_current_step)
                baseline_results = '{result_folder}/accumulated_reward_test_baseline_trained_0_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}.csv'.format(
                    result_folder=baseline_trace_folder,
                    patient_type=patient_type, patient_id=patient_id,
                    trace_eps=trace_eps, trace_current_step=trace_current_step)
                orig_trace_file = '{result_folder}/orig_trace_test_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}.csv'.format(
                    result_folder=baseline_trace_folder, patient_type=patient_type,
                    patient_id=patient_id, trace_eps=trace_eps, trace_current_step=trace_current_step)
                ddpg_test_result_df = pd.read_csv(ddpg_test_result, index_col=0)
                baseline_result_df = pd.read_csv(baseline_results, index_col=0)
                ddpg_test_trace_df = pd.read_pickle(ddpg_test_traces)
                # ddpg_test_trace_df['orig_trace_episode'].astype('int64')
                # ddpg_test_trace_df['orig_end_step'].astype('int64')
                # ddpg_test_trace_df['step'].astype('int64')
                # ddpg_test_trace_df['episode'].astype('int64')
                # ddpg_test_trace_df['episode_step'].astype('int64')
                # print(ddpg_test_trace_df.info(),ddpg_test_trace_df['orig_trace_episode'].tolist())
                ddpg_test_trace_df.to_csv('{result_folder}/test/cf_traces_test_r_{round}.csv'.format(result_folder=ddpg_trace_folder,
                                                                               trace_eps=trace_eps,
                                                                               trace_current_step=trace_current_step,
                                                                               round=r))
                baseline_trace_df = pd.read_csv(baseline_traces, index_col=0)
                orig_trace_df = pd.read_csv(orig_trace_file, index_col=0)
                # Diabetic
                env_id = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type,
                                                                           patient_id=patient_id)
                ddpg_test_result_df = ddpg_test_result_df[(ddpg_test_result_df['orig_trace_episode'] == trace_eps)
                                                          & (ddpg_test_result_df['orig_end_step'] == trace_current_step)
                                                          & (ddpg_test_result_df['ENV_ID'] == env_id)
                                                          & (ddpg_test_result_df['train_round'] == r)]
                ddpg_test_trace_df = ddpg_test_trace_df[(ddpg_test_trace_df['orig_trace_episode'] == trace_eps)
                                                        & (ddpg_test_trace_df['orig_end_step'] == trace_current_step)
                                                        & (ddpg_test_trace_df['ENV_ID'] == env_id)]
                # print('df info: ',ddpg_test_trace_df.info())

                orig_obs_trace_this_seg = orig_trace_df['observation_CGM'].tolist()
                orig_action_trace_this_seg = orig_trace_df['action'].tolist()
                orig_reward_trace_this_seg = orig_trace_df['reward'].tolist()
                J0 = sum(orig_trace_df['reward'].tolist())
                x_tick = range(1, len(orig_obs_trace_this_seg) + 1)
                total_eps_num = 1
                # print('best_result_for_lowest_distance_list_train: ', best_result_for_lowest_distance_list_train)
                # print('best_result_for_lowest_distance_list_test: ', best_result_for_lowest_distance_list_test)
                fig = plt.figure(figsize=(10, 18))
                plt.subplots_adjust(left=None, bottom=None, top=None, right=None, hspace=0.3, wspace=None)
                plt.subplot(3, 1, 1)
                plt.ticklabel_format(style='plain')
                plt.plot(x_tick, orig_obs_trace_this_seg, color='b', linewidth=1.0,
                         label='orig_obs_{trace_eps}_{trace_current_step}_r_{round}'.format(
                             trace_eps=trace_eps, trace_current_step=trace_current_step, round=r))
                for eps in range(0, total_eps_num):
                    # print(trace_eps, trace_current_step, eps)
                    ddpg_obs_trace_this_seg = ddpg_test_trace_df[ddpg_test_trace_df['episode'] == eps]['observation_CGM'].tolist()
                    baseline_obs_trace_this_seg = baseline_trace_df[baseline_trace_df['episode'] == eps]['observation_CGM'].tolist()
                    if len(ddpg_obs_trace_this_seg) > 20:
                        ddpg_obs_trace_this_seg = ddpg_obs_trace_this_seg[:20]
                    ddpg_tick = range(1, len(ddpg_obs_trace_this_seg) + 1)
                    plt.plot(ddpg_tick, ddpg_obs_trace_this_seg, linewidth=0.5,
                             marker='.',
                             label='ddpg_obs_test_{trace_eps}_{trace_current_step}_r_{round}'.format(
                                 trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.8)
                    plt.plot(x_tick, baseline_obs_trace_this_seg, linewidth=0.5,
                             marker='+',
                             label='baseline_obs_test_{trace_eps}_{trace_current_step}_r_{round}'.format(trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.8)
                plt.title(
                    'obs_trace_compare_round_{round}_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}'.format(
                        round=r, trace_eps=trace_eps,
                        trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id))
                plt.ylabel('Value')
                plt.xticks(x_tick)
                plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
                plt.legend()
                plt.subplot(3, 1, 2)
                plt.ticklabel_format(style='plain')
                plt.plot(x_tick, orig_action_trace_this_seg, color='b', linewidth=1.0,
                         label='orig_action_{trace_eps}_{trace_current_step}_r_{round}'.format(
                             trace_eps=trace_eps, trace_current_step=trace_current_step, round=r))
                for eps in range(0, total_eps_num):
                    ddpg_action_trace_this_seg = ddpg_test_trace_df[ddpg_test_trace_df['episode'] == eps]['action'].tolist()
                    baseline_action_trace_this_seg = baseline_trace_df[baseline_trace_df['episode'] == eps]['action'].tolist()
                    if len(ddpg_action_trace_this_seg) > 20:
                        ddpg_action_trace_this_seg = ddpg_action_trace_this_seg[:20]
                    plt.plot(ddpg_tick, ddpg_action_trace_this_seg, linewidth=0.5,
                             marker='.',
                             label='ddpg_action_test_{trace_eps}_{trace_current_step}_r_{round}'.format(
                                 trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.8)
                    plt.plot(x_tick, baseline_action_trace_this_seg, linewidth=0.5,
                             marker='+',
                             label='baseline_action_test_{trace_eps}_{trace_current_step}_r_{round}'.format(
                                 trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.8)
                plt.title(
                    'action_trace_compare_round_{round}_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}'.format(
                        round=r, trace_eps=trace_eps,
                        trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id))
                plt.ylabel('Value')
                plt.xticks(x_tick)
                plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
                plt.legend()
                plt.subplot(3, 1, 3)
                plt.ticklabel_format(style='plain')
                plt.plot(x_tick, orig_reward_trace_this_seg, color='b', linewidth=1.0,
                         label='orig_reward_{trace_eps}_{trace_current_step}_r_{round}'.format(
                             trace_eps=trace_eps, trace_current_step=trace_current_step, round=r))
                for eps in range(0, total_eps_num):
                    ddpg_reward_trace_this_seg = ddpg_test_trace_df[ddpg_test_trace_df['episode'] == eps]['reward'].tolist()
                    baseline_reward_trace_this_seg = baseline_trace_df[baseline_trace_df['episode'] == eps]['reward'].tolist()
                    if len(ddpg_reward_trace_this_seg) > 20:
                        ddpg_reward_trace_this_seg = ddpg_reward_trace_this_seg[:20]
                    plt.plot(ddpg_tick, ddpg_reward_trace_this_seg, linewidth=0.5,
                             marker='.',
                             label='ddpg_action_test_{trace_eps}_{trace_current_step}_r_{round}'.format(
                                 trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.8)
                    plt.plot(x_tick, baseline_reward_trace_this_seg, linewidth=0.5,
                             marker='+',
                             label='ddpg_action_test_{trace_eps}_{trace_current_step}_r_{round}'.format(
                                 trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.8)
                plt.title(
                    'reward_trace_compare_round_{round}_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}'.format(
                        round=r, trace_eps=trace_eps,
                        trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id))
                plt.ylabel('Value')
                plt.xticks(x_tick)
                plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
                plt.legend()
                file_path = '{folder}/trace_compare_round_{round}_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}.png'.format(
                    folder=figure_folder, round=r, trace_eps=trace_eps,
                    trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id)
                plt.savefig(file_path)
                plt.close()
            else:
                trace_eps, trace_current_step, gravity, patient_type, patient_id = row['orig_episode'], row['orig_end_time_index'], row['gravity'], None, None
                figure_folder = '{figure_folder_0}/{trace_eps}_{trace_current_step}'.format(
                    figure_folder_0=figure_folder_0,
                    trace_eps=trace_eps, trace_current_step=trace_current_step)
                mkdir(figure_folder)
                # TODO: to be removed
                # trace_eps_int_8 = np.int8(trace_eps)
                # print(trace_eps, trace_current_step, r)
                ddpg_test_traces = '{result_folder}/test/cf_traces_test_r_{round}.pkl'.format(
                    result_folder=ddpg_trace_folder,
                    trace_eps=trace_eps,
                    trace_current_step=trace_current_step, round=r)
                ddpg_test_result = '{result_folder}/test/all_test_result.csv'.format(result_folder=ddpg_trace_folder)
                # Lunar lander
                baseline_traces = '{result_folder}/cf_trace_file_test_baseline_trained_0_g_{gravity}_{trace_eps}_{trace_current_step}.csv'.format(
                    result_folder=baseline_trace_folder, trace_eps=trace_eps, trace_current_step=trace_current_step,
                    gravity=gravity)
                baseline_results = '{result_folder}/accumulated_reward_test_baseline_trained_0_g_{gravity}_{trace_eps}_{trace_current_step}.csv'.format(
                    result_folder=baseline_trace_folder, trace_eps=trace_eps, trace_current_step=trace_current_step,
                    gravity=gravity)
                orig_trace_file = '{result_folder}/orig_trace_test_g_{gravity}_{trace_eps}_{trace_current_step}.csv'.format(
                    result_folder=baseline_trace_folder, trace_eps=trace_eps, trace_current_step=trace_current_step,
                    gravity=gravity)
                ddpg_test_result_df = pd.read_csv(ddpg_test_result, index_col=0)
                baseline_result_df = pd.read_csv(baseline_results, index_col=0)
                ddpg_test_trace_df = pd.read_pickle(ddpg_test_traces)
                # ddpg_test_trace_df['orig_trace_episode'].astype('int64')
                # ddpg_test_trace_df['orig_end_step'].astype('int64')
                # ddpg_test_trace_df['step'].astype('int64')
                # ddpg_test_trace_df['episode'].astype('int64')
                # ddpg_test_trace_df['episode_step'].astype('int64')
                # print(ddpg_test_trace_df.info(),ddpg_test_trace_df['orig_trace_episode'].tolist())
                ddpg_test_trace_df.to_csv('{result_folder}/test/cf_traces_test_r_{round}.csv'.format(result_folder=ddpg_trace_folder,
                                                                               trace_eps=trace_eps,
                                                                               trace_current_step=trace_current_step,
                                                                               round=r))
                baseline_trace_df = pd.read_csv(baseline_traces, index_col=0)
                orig_trace_df = pd.read_csv(orig_trace_file, index_col=0)
                # Lunar lander
                ddpg_test_result_df = ddpg_test_result_df[(ddpg_test_result_df['orig_trace_episode'] == trace_eps)
                                                          & (ddpg_test_result_df['orig_end_step'] == trace_current_step)
                                                          & (ddpg_test_result_df['gravity'] == gravity)
                                                            & (ddpg_test_result_df['train_round'] == r)]
                ddpg_test_trace_df = ddpg_test_trace_df[(ddpg_test_trace_df['orig_trace_episode'] == trace_eps)
                                                        & (ddpg_test_trace_df['orig_end_step'] == trace_current_step)
                                                        & (ddpg_test_trace_df['gravity'] == gravity)]
                # print('df info: ',ddpg_test_trace_df.info())

                orig_obs_trace_this_seg = orig_trace_df['observation'].tolist()
                orig_obs_trace_dict_this_seg = get_lunar_lander_trace_list(orig_obs_trace_this_seg, trace_type='obs', exp_type='orig_trace')
                orig_action_trace_this_seg = orig_trace_df['action'].tolist()
                orig_action_trace_dict_this_seg = get_lunar_lander_trace_list(orig_action_trace_this_seg, trace_type='action', exp_type='orig_trace')
                orig_reward_trace_this_seg = orig_trace_df['reward'].tolist()
                J0 = sum(orig_trace_df['reward'].tolist())
                x_tick = range(1, len(orig_obs_trace_this_seg) + 1)
                total_eps_num = 10
                # print('best_result_for_lowest_distance_list_train: ', best_result_for_lowest_distance_list_train)
                # print('best_result_for_lowest_distance_list_test: ', best_result_for_lowest_distance_list_test)
                fig = plt.figure(figsize=(10, 80))
                plt.subplots_adjust(left=None, bottom=None, top=None, right=None, hspace=0.3, wspace=None)
                num_sub_polt = 10
                for obs_index in range(1, 7):
                    plt.subplot(num_sub_polt, 1, obs_index)
                    plt.ticklabel_format(style='plain')
                    plt.plot(x_tick, orig_obs_trace_dict_this_seg[obs_index-1], color='b', linewidth=1.0,
                             label='orig_obs_{obs_index}_{trace_eps}_{trace_current_step}_r_{round}'.format(obs_index=obs_index,
                                 trace_eps=trace_eps, trace_current_step=trace_current_step, round=r))
                    for eps in range(0, total_eps_num):
                        # print(trace_eps, trace_current_step, eps)
                        #print('Begin DDPG trace.')
                        ddpg_obs_trace_this_seg = ddpg_test_trace_df[ddpg_test_trace_df['episode'] == eps]['observation'].tolist()
                        if len(ddpg_obs_trace_this_seg) > 20:
                            ddpg_obs_trace_this_seg = ddpg_obs_trace_this_seg[:20]
                        #print('ddpg_obs_trace_this_seg: ', ddpg_obs_trace_this_seg)
                        for a in range(len(ddpg_obs_trace_this_seg)):
                            ddpg_obs_trace_this_seg[a] = ddpg_obs_trace_this_seg[a].tolist()
                        ddpg_obs_trace_dict_this_seg = get_lunar_lander_trace_list(ddpg_obs_trace_this_seg,
                                                                                   trace_type='obs', exp_type='ddpg_trace')
                        plt.plot(x_tick, ddpg_obs_trace_dict_this_seg[obs_index-1], linewidth=0.2,
                                 marker='.',
                                 label='ddpg_obs_{obs_index}_test_{trace_eps}_{trace_current_step}_r_{round}'.format(obs_index=obs_index,
                                     trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.5)
                    plt.title(
                        'obs_{obs_index}_trace_compare_round_{round}_{gravity}_{trace_eps}_{trace_current_step}'.format(
                            obs_index=obs_index, round=r, trace_eps=trace_eps,
                            trace_current_step=trace_current_step, gravity=gravity))
                    plt.ylabel('Value')
                    plt.xticks(x_tick)
                    plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
                    # plt.legend()
                ## Action
                for action_index in range(7, 9):
                    plt.subplot(num_sub_polt, 1, action_index)
                    plt.ticklabel_format(style='plain')
                    plt.plot(x_tick, orig_action_trace_dict_this_seg[action_index-7], color='b', linewidth=1.0,
                             label='orig_action_{action_index}_{trace_eps}_{trace_current_step}_r_{round}'.format(action_index=action_index-6,
                                 trace_eps=trace_eps, trace_current_step=trace_current_step, round=r))
                    for eps in range(0, total_eps_num):
                        ddpg_action_trace_this_seg = ddpg_test_trace_df[ddpg_test_trace_df['episode'] == eps]['action'].tolist()
                        if len(ddpg_action_trace_this_seg) > 20:
                            ddpg_action_trace_this_seg = ddpg_action_trace_this_seg[:20]
                        for a in range(len(ddpg_action_trace_this_seg)):
                            ddpg_action_trace_this_seg[a] = ddpg_action_trace_this_seg[a].tolist()
                        ddpg_action_trace_dict_this_seg = get_lunar_lander_trace_list(ddpg_action_trace_this_seg,
                                                                                   trace_type='action', exp_type='ddpg_trace')
                        plt.plot(x_tick, ddpg_action_trace_dict_this_seg[action_index-7], linewidth=0.2,
                                 marker='.',
                                 label='ddpg_action_{action_index}_test_{trace_eps}_{trace_current_step}_r_{round}'.format(action_index=action_index-7,
                                     trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.5)
                    plt.title(
                        'action_{action_index}_trace_compare_round_{round}_{gravity}_{trace_eps}_{trace_current_step}'.format(
                            action_index=action_index-7, round=r, trace_eps=trace_eps,
                            trace_current_step=trace_current_step, gravity=gravity))
                    plt.ylabel('Value')
                    plt.xticks(x_tick)
                    plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
                # plt.legend()
                ## Reward
                plt.subplot(num_sub_polt, 1, 10)
                plt.ticklabel_format(style='plain')
                plt.plot(x_tick, orig_reward_trace_this_seg, color='b', linewidth=1.0,
                         label='orig_reward_{trace_eps}_{trace_current_step}_r_{round}'.format(
                             trace_eps=trace_eps, trace_current_step=trace_current_step, round=r))
                for eps in range(0, total_eps_num):
                    ddpg_reward_trace_this_seg = ddpg_test_trace_df[ddpg_test_trace_df['episode'] == eps]['reward'].tolist()
                    if len(ddpg_reward_trace_this_seg) > 20:
                        ddpg_reward_trace_this_seg = ddpg_reward_trace_this_seg[:20]
                    plt.plot(x_tick, ddpg_reward_trace_this_seg, linewidth=0.2,
                             marker='.',
                             label='ddpg_reward_test_{trace_eps}_{trace_current_step}_r_{round}'.format(
                                 trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.5)
                plt.title(
                    'reward_trace_compare_round_{round}_{gravity}_{trace_eps}_{trace_current_step}'.format(
                        round=r, trace_eps=trace_eps,
                        trace_current_step=trace_current_step, gravity=gravity))
                plt.ylabel('Value')
                plt.xticks(x_tick)
                plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
                # plt.legend()
                file_path = '{folder}/trace_compare_round_{round}_{gravity}_{trace_eps}_{trace_current_step}.png'.format(
                    folder=figure_folder, round=r, trace_eps=trace_eps,
                    trace_current_step=trace_current_step, gravity=gravity)
                plt.savefig(file_path)
                plt.close()

    return

def draw_distribution_no_better_cf_trace_first_state(all_summary_df,figure_folder):
    exp_id_list = all_summary_df['exp_id'].drop_duplicates().tolist()
    #bins = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260]
    bins = [60, 80, 100, 130, 150, 180, 260]
    #labels = range(1, len(bins))
    labels = ['60-80', '80-100', '100-130', '130-150', '150-180', '180-260']
    print(len(bins), len(labels))
    for exp_id in exp_id_list:
        this_exp_df = all_summary_df[all_summary_df['exp_id']==exp_id]
        patient_type = this_exp_df['patient_type'].tolist()[0]
        patient_id = this_exp_df['patient_id'].tolist()[0]
        # trace_eps = this_exp_df['trace_eps'].tolist()[0]
        # trace_current_step = this_exp_df['trace_current_step'].tolist()[0]
        no_better_cf_df = this_exp_df[this_exp_df['ddpg_lowest_distance']==9999]
        first_state_df = no_better_cf_df['start_state']
        groups = pd.cut(first_state_df.tolist(), bins=bins, labels=labels)
        freq = groups.value_counts().sort_index()
        #plt.hist(first_state_df, bins=bins)
        plt.bar(freq.index.astype(str), freq.values)
        plt.title('first state distribution ({patient_type}_{patient_id}_Exp_{exp_id})'.format(patient_type=patient_type,patient_id=patient_id, exp_id=exp_id))
        # plt.legend()
        file_path = '{folder}/first_state_distribution_({patient_type}_{patient_id}_Exp_{exp_id})'.format(folder=figure_folder,
                                                                                                          patient_type=patient_type,patient_id=patient_id, exp_id=exp_id)
        plt.savefig(file_path)
        plt.close()
    return

def get_each_patient_result_separate(all_test_result_summary_df, save_folder):
    # check on each patient in test when train and test with multiple patient envs
    result_df = pd.DataFrame(columns=['exp_id','patient_type', 'patient_id', 'total_trace_num','P_hr_ddpg','P_hr_baseline', 'P_better', 'r_d',
                                      'ddpg_better_cf_for_orig_count', 'baseline_better_cf_for_orig_count','ddpg_cf_better_than_baseline_count'])
    all_patient_id_list = all_test_result_summary_df['patient_id'].drop_duplicates().tolist()
    all_exp_list = all_test_result_summary_df['exp_id'].drop_duplicates().tolist()
    for exp_id in all_exp_list:
        for p_id in all_patient_id_list:
            this_p_df = all_test_result_summary_df[(all_test_result_summary_df['patient_id']==p_id)&(all_test_result_summary_df['exp_id']==exp_id)]
            total_trace_num = len(this_p_df)
            baseline_count = len(this_p_df[this_p_df['baseline_lowest_distance']!=9999])
            ddpg_count = len(this_p_df[this_p_df['ddpg_lowest_distance'] != 9999])
            r_d = this_p_df[(this_p_df['baseline_lowest_distance']!=9999)&(this_p_df['ddpg_lowest_distance'] != 9999)]['distance_ratio'].mean()
            ddpg_better_count = 0
            for item, row in this_p_df.iterrows():
                ddpg_distance_this_seg = row['ddpg_lowest_distance']
                ddpg_outcome_this_seg = row['ddpg_best_outcome_for_lowest_distance']
                baseline_distance_this_seg = row['baseline_lowest_distance']
                baseline_outcome_this_seg = row['baseline_best_outcome_for_lowest_distance']
                # if (ddpg_outcome_this_seg>=baseline_outcome_this_seg) and (ddpg_distance_this_seg<baseline_distance_this_seg):
                #     ddpg_better_count += 1
                if (ddpg_outcome_this_seg > 0) and (round(ddpg_outcome_this_seg, 4) >= round(baseline_outcome_this_seg,4)) \
                        and (round(ddpg_distance_this_seg, 4) < round(baseline_distance_this_seg, 4)):
                    ddpg_better_count += 1
            result_dict = {'exp_id':exp_id,'patient_type':'adult', 'patient_id':p_id, 'total_trace_num':total_trace_num,
                           'P_hr_ddpg':ddpg_count/total_trace_num,
                           'P_hr_baseline':baseline_count/total_trace_num,
                           'P_better':ddpg_better_count/total_trace_num,
                           'r_d':r_d,
                            'ddpg_better_cf_for_orig_count': ddpg_count,
                           'baseline_better_cf_for_orig_count':baseline_count,
                           'ddpg_cf_better_than_baseline_count':ddpg_better_count}
            result_df = pd.concat([result_df,pd.DataFrame([result_dict])],ignore_index=True)
    save_path = '{save_folder}/test_result_summary_each_patient.csv'.format(save_folder=save_folder)
    result_df.to_csv(save_path)
    return

def get_lunar_lander_trace_list(trace, trace_type='action', exp_type='orig_trace'):
    trace_dict = {}
    if trace_type=='action':
        for i in range(0, 2):
            trace_dict[i] = []
    elif trace_type=='obs':
        for i in range(0, 8):
            trace_dict[i] = []
    for item in trace:
        if exp_type=='orig_trace':
            #print('item: ', item, type(item))
            item = item[1:-1].split(' ')
            #print('item before: ', item)
            for i in item:
                #print('i: ', i, type(i))
                index = item.index(i)
                i = i.rstrip('\n').strip(',')
                #print('i after: ', i)
                item[index] = i
            while '' in item:
                item.remove('')
            # for i in item:
            #     if i=='':
            #         item.remove(i)
            #print('item after: ', item)
            item_list = [float(x) for x in item]
            #print(item, item_list)
            for k in range(len(item_list)):
                value = item_list[k]
                trace_dict[k].append(value)
                #print('k: ', k, 'trace_dict[index]: ', trace_dict[k])
        else:
            item_list = [float(x) for x in item]
            # print(item, item_list)
            for k in range(len(item_list)):
                value = item_list[k]
                trace_dict[k].append(value)
                # print('k: ', k, 'trace_dict[index]: ', trace_dict[k])
    #print(trace_dict)
    return trace_dict

def draw_log_data(folder, baseline_path, id, trace_eps, trace_time_index, all_test_result_df, all_train_id_list):
    baseline_df = pd.read_csv(baseline_path, index_col=0)
    cf_accumulated_reward_baseline = baseline_df['cf_accumulated_reward'].tolist()
    difference_baseline = baseline_df['difference'].tolist()
    cf_pairwise_distance_baseline = baseline_df['cf_pairwise_distance'].tolist()
    cf_accumulated_reward_baseline_mean = np.mean(cf_accumulated_reward_baseline)
    cf_accumulated_reward_baseline_SE = np.std(cf_accumulated_reward_baseline, ddof=1)/np.sqrt(len(cf_accumulated_reward_baseline))
    difference_baseline_mean = np.mean(difference_baseline)
    difference_baseline_SE = np.std(difference_baseline, ddof=1)/np.sqrt(len(difference_baseline))
    cf_pairwise_distance_baseline_mean = np.mean(cf_pairwise_distance_baseline)
    cf_pairwise_distance_baseline_SE = np.std(cf_pairwise_distance_baseline, ddof=1)/np.sqrt(len(cf_pairwise_distance_baseline))
    log_csv_path = '{folder}/progress.csv'.format(folder=folder)
    log_df = pd.read_csv(log_csv_path, index_col=0)
    test_result_col_name = ['test/test_cf_pairwise_distance', 'test/test_total_reward_difference', 'test/test_accumulated_reward_compare',
                            'test/test_cf_pairwise_distance_SE', 'test/test_accumulated_reward_SE', 'test/test_accumulated_reward',
                            'test/test_total_reward_difference_compare', 'test/test_cf_pairwise_distance_compare',
                            'test/test_total_reward_difference_SE', 'test/test_cf_pairwise_distance_ratio']
    col_name = ['cf_pairwise_distance', 'difference', 'cf_accumulated_reward']
    up_dist_list, low_dist_list, aveg_dist_list, std_dist_list = get_up_low_bound_from_test_result(all_test_result_df, all_train_id_list, col_name='cf_pairwise_distance')
    up_difference_list, low_difference_list, aveg_difference_list, std_difference_list = get_up_low_bound_from_test_result(all_test_result_df,
                                                                                                   all_train_id_list,
                                                                                                   col_name='difference')
    up_reward_list, low_reward_list, aveg_reward_list, std_reward_list = get_up_low_bound_from_test_result(all_test_result_df,
                                                                                                   all_train_id_list,
                                                                                                   col_name='cf_accumulated_reward')
    test_df = log_df[test_result_col_name].dropna(axis=0, how='any')
    #print(test_df)
    color = 'green'
    color_1 = 'black'
    color_2 = 'blue'
    start_idx = 0
    fig = plt.figure(figsize=(30, 39))
    plt.subplots_adjust(left=None, bottom=None, top=None, right=None, hspace=0.3, wspace=None)
    # plot 1:test_cf_pairwise_distance
    plt.subplot(3, 1, 1)
    plt.ticklabel_format(style='plain')
    test_cf_pairwise_distance = test_df['test/test_cf_pairwise_distance'].tolist()[start_idx:]
    test_cf_pairwise_distance_SE = test_df['test/test_cf_pairwise_distance_SE'].tolist()[start_idx:]
    r1 = list(map(lambda x: x[0] - x[1], zip(test_cf_pairwise_distance, test_cf_pairwise_distance_SE)))
    r2 = list(map(lambda x: x[0] + x[1], zip(test_cf_pairwise_distance, test_cf_pairwise_distance_SE)))
    x_value = range(1, len(test_cf_pairwise_distance) + 1)
    plt.plot(x_value, test_cf_pairwise_distance,color=color, label='cf_pairwise_distance_ddpg_mean', alpha=0.8, linewidth=0.5)
    plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
    #plt.plot(x_value, aveg_dist_list, color=color_2, label='cf_pairwise_distance_ddpg_aveg', alpha=0.8,linewidth=0.5, linestyle='-.')
    x_value_1 = range(3, len(aveg_dist_list) + 3)
    plt.plot(x_value_1, up_dist_list, color=color_2, label='cf_pairwise_distance_ddpg_upper', alpha=0.8,linewidth=0.5, linestyle='-.')
    plt.plot(x_value_1, low_dist_list, color=color_2, label='cf_pairwise_distance_ddpg_lower', alpha=0.8,linewidth=0.5, linestyle='-.')
    r1_std = list(map(lambda x: x[0] - x[1], zip(aveg_dist_list, std_dist_list)))
    r2_std = list(map(lambda x: x[0] + x[1], zip(aveg_dist_list, std_dist_list)))
    plt.fill_between(x_value_1, r1_std, r2_std, color=color_2, alpha=0.1)
    # draw baseline
    cf_pairwise_distance_mean_baseline = [cf_pairwise_distance_baseline_mean]*len(x_value)
    cf_pairwise_distance_SE_baseline = [cf_pairwise_distance_baseline_SE]*len(x_value)
    r1_b = list(map(lambda x: x[0] - x[1], zip(cf_pairwise_distance_mean_baseline, cf_pairwise_distance_SE_baseline)))
    r2_b = list(map(lambda x: x[0] + x[1], zip(cf_pairwise_distance_mean_baseline, cf_pairwise_distance_SE_baseline)))
    plt.plot(x_value, cf_pairwise_distance_mean_baseline, color=color_1, label='cf_pairwise_distance_baseline', alpha=0.8, linewidth=0.5)
    plt.fill_between(x_value, r1_b, r2_b, color=color_1, alpha=0.2)
    plt.xlabel('Train step (*1000)')
    plt.ylabel('Value')
    plt.legend()
    plt.title('test_cf_pairwise_distance')
    # plot 2:test_total_reward_difference
    plt.subplot(3, 1, 2)
    plt.ticklabel_format(style='plain')
    test_total_reward_difference = test_df['test/test_total_reward_difference'].tolist()[start_idx:]
    test_total_reward_difference_SE = test_df['test/test_total_reward_difference_SE'].tolist()[start_idx:]
    r1 = list(map(lambda x: x[0] - x[1], zip(test_total_reward_difference, test_total_reward_difference_SE)))
    r2 = list(map(lambda x: x[0] + x[1], zip(test_total_reward_difference, test_total_reward_difference_SE)))
    x_value = range(1, len(test_total_reward_difference) + 1)
    plt.plot(x_value, test_total_reward_difference, color=color, label='total_reward_difference_ddpg_mean', alpha=0.8, linewidth=0.5)
    plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
    #plt.plot(x_value, aveg_difference_list, color=color_2, label='total_reward_difference_ddpg_aveg', alpha=0.8, linewidth=0.5,linestyle='-.')
    x_value_1 = range(3, len(aveg_difference_list) + 3)
    plt.plot(x_value_1, up_difference_list, color=color_2, label='total_reward_difference_ddpg_upper', alpha=0.8, linewidth=0.5,linestyle='-.')
    plt.plot(x_value_1, low_difference_list, color=color_2, label='total_reward_difference_ddpg_lower', alpha=0.8, linewidth=0.5,linestyle='-.')
    r1_std = list(map(lambda x: x[0] - x[1], zip(aveg_difference_list, std_difference_list)))
    r2_std = list(map(lambda x: x[0] + x[1], zip(aveg_difference_list, std_difference_list)))
    plt.fill_between(x_value_1, r1_std, r2_std, color=color_2, alpha=0.1)
    # draw baseline
    difference_mean_baseline = [difference_baseline_mean] * len(x_value)
    difference_SE_baseline = [difference_baseline_SE] * len(x_value)
    r1_b = list(map(lambda x: x[0] - x[1], zip(difference_mean_baseline, difference_SE_baseline)))
    r2_b = list(map(lambda x: x[0] + x[1], zip(difference_mean_baseline, difference_SE_baseline)))
    plt.plot(x_value, difference_mean_baseline, color=color_1, label='total_reward_difference_baseline', alpha=0.8, linewidth=0.5)
    plt.fill_between(x_value, r1_b, r2_b, color=color_1, alpha=0.2)
    plt.xlabel('Train step (*1000)')
    plt.ylabel('Value')
    plt.legend()
    plt.title('test_total_reward_difference')
    # plot 3:test_accumulated_reward
    plt.subplot(3, 1, 3)
    plt.ticklabel_format(style='plain')
    test_accumulated_reward = test_df['test/test_accumulated_reward'].tolist()[start_idx:]
    test_accumulated_reward_SE = test_df['test/test_accumulated_reward_SE'].tolist()[start_idx:]
    r1 = list(map(lambda x: x[0] - x[1], zip(test_accumulated_reward, test_accumulated_reward_SE)))
    r2 = list(map(lambda x: x[0] + x[1], zip(test_accumulated_reward, test_accumulated_reward_SE)))
    x_value = range(1, len(test_accumulated_reward) + 1)
    plt.plot(x_value, test_accumulated_reward, color=color, label='accumulated_reward_ddpg_mean', alpha=0.8, linewidth=0.5)
    plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
    #plt.plot(x_value, aveg_reward_list, color=color_2, label='accumulated_reward_ddpg_aveg', alpha=0.8, linewidth=0.5,linestyle='-.')
    x_value_1 = range(3, len(aveg_reward_list) + 3)
    plt.plot(x_value_1, up_reward_list, color=color_2, label='accumulated_reward_ddpg_upper', alpha=0.8, linewidth=0.5,linestyle='-.')
    plt.plot(x_value_1, low_reward_list, color=color_2, label='accumulated_reward_ddpg_lower', alpha=0.8, linewidth=0.5,linestyle='-.')
    r1_std = list(map(lambda x: x[0] - x[1], zip(aveg_reward_list, std_reward_list)))
    r2_std = list(map(lambda x: x[0] + x[1], zip(aveg_reward_list, std_reward_list)))
    plt.fill_between(x_value_1, r1_std, r2_std, color=color_2, alpha=0.1)
    # draw baseline
    cf_accumulated_reward_mean_baseline = [cf_accumulated_reward_baseline_mean] * len(x_value)
    cf_accumulated_reward_SE_baseline = [cf_accumulated_reward_baseline_SE] * len(x_value)
    r1_b = list(map(lambda x: x[0] - x[1], zip(cf_accumulated_reward_mean_baseline, cf_accumulated_reward_SE_baseline)))
    r2_b = list(map(lambda x: x[0] + x[1], zip(cf_accumulated_reward_mean_baseline, cf_accumulated_reward_SE_baseline)))
    plt.plot(x_value, cf_accumulated_reward_mean_baseline, color=color_1, label='accumulated_reward_baseline', alpha=0.8, linewidth=0.5)
    plt.fill_between(x_value, r1_b, r2_b, color=color_1, alpha=0.2)
    plt.xlabel('Train step (*1000)')
    plt.ylabel('Value')
    plt.legend()
    plt.title('test_accumulated_reward')

    file_path = '{folder}/test_curve_{id}_{trace_eps}_{trace_time_index}.png'.format(folder=folder, id=id, trace_eps=trace_eps, trace_time_index=trace_time_index)
    plt.savefig(file_path)
    plt.close()

    return

def draw_test_curve_single_trace(data_folder, exp_id, test_index_df, save_folder, trial_num, xaxis_interval, test_result_type, case_name='lunar_lander'):
    test_statistic_baseline = '{data_folder}/{test_result_type}/all_test_statistic_baseline.csv'.format(data_folder=data_folder,test_result_type=test_result_type)
    test_statistic_ddpg = '{data_folder}/{test_result_type}/all_test_statistic_ddpg.csv'.format(data_folder=data_folder,test_result_type=test_result_type)
    test_statistic_baseline_df = pd.read_csv(test_statistic_baseline)
    test_statistic_ddpg_df = pd.read_csv(test_statistic_ddpg)
    for item, row in test_index_df.iterrows():
        if case_name=='lunar_lander':
            trace_eps, trace_current_step, ENV_ID = row['orig_episode'], row['orig_end_time_index'], row['gravity']
            ddpg_df_this_seg = test_statistic_ddpg_df[(test_statistic_ddpg_df['orig_trace_episode'] == trace_eps)
                                                      & (test_statistic_ddpg_df['orig_end_step'] == trace_current_step)
                                                      & (test_statistic_ddpg_df['gravity'] == ENV_ID)]
            baseline_df_this_seg = test_statistic_baseline_df[
                (test_statistic_baseline_df['orig_trace_episode'] == trace_eps)
                & (test_statistic_baseline_df['orig_end_step'] == trace_current_step)
                & (test_statistic_baseline_df['gravity'] == ENV_ID)]
        else:
            trace_eps, trace_current_step, ENV_ID = row['orig_episode'], row['orig_end_time_index'], row['ENV_NAME']
            ddpg_df_this_seg = test_statistic_ddpg_df[(test_statistic_ddpg_df['orig_trace_episode']==trace_eps)
                                                      &(test_statistic_ddpg_df['orig_end_step']==trace_current_step)
                                                      &(test_statistic_ddpg_df['ENV_ID']==ENV_ID)]
            baseline_df_this_seg = test_statistic_baseline_df[(test_statistic_baseline_df['orig_trace_episode'] == trace_eps)
                                                              & (test_statistic_baseline_df['orig_end_step'] == trace_current_step)
                                                                &(test_statistic_baseline_df['ENV_ID']==ENV_ID)]
        ddpg_distance_mean = ddpg_df_this_seg['aveg_cf_pairwise_distance'].tolist()
        ddpg_distance_SE = ddpg_df_this_seg['cf_pairwise_distance_SE'].tolist()
        ddpg_distance_SD = ddpg_df_this_seg['cf_pairwise_distance_SD'].tolist()
        ddpg_difference_mean = ddpg_df_this_seg['aveg_total_reward_difference'].tolist()
        ddpg_difference_SE = ddpg_df_this_seg['total_reward_difference_SE'].tolist()
        ddpg_difference_SD = ddpg_df_this_seg['total_reward_difference_SD'].tolist()
        ddpg_accumulate_reward_mean = ddpg_df_this_seg['aveg_accumulated_reward'].tolist()
        ddpg_accumulate_reward_SE = ddpg_df_this_seg['accumulated_reward_SE'].tolist()
        ddpg_accumulate_reward_SD = ddpg_df_this_seg['accumulated_reward_SD'].tolist()
        ddpg_total_final_reward_mean = ddpg_df_this_seg['aveg_total_final_reward'].tolist()
        ddpg_total_final_reward_SE = ddpg_df_this_seg['total_final_reward_difference_SE'].tolist()
        ddpg_total_final_reward_SD = ddpg_df_this_seg['total_final_reward_difference_SD'].tolist()
        ddpg_successful_rate = ddpg_df_this_seg['successful_rate'].tolist()
        ddpg_successful_rate_mean = [np.mean(ddpg_df_this_seg['successful_rate'].tolist())]*len(ddpg_successful_rate)
        ddpg_successful_rate_SD = [np.std(ddpg_df_this_seg['successful_rate'].tolist(), ddof=1)]*len(ddpg_successful_rate)
        ddpg_successful_rate_SE = [ddpg_successful_rate_SD[0]/np.sqrt(len(ddpg_df_this_seg['successful_rate'].tolist()))]*len(ddpg_successful_rate)

        baseline_distance_mean = baseline_df_this_seg['aveg_cf_pairwise_distance'].tolist()
        baseline_distance_SE = baseline_df_this_seg['cf_pairwise_distance_SE'].tolist()
        baseline_distance_SD = baseline_df_this_seg['cf_pairwise_distance_SD'].tolist()
        baseline_difference_mean = baseline_df_this_seg['aveg_total_reward_difference'].tolist()
        baseline_difference_SE = baseline_df_this_seg['total_reward_difference_SE'].tolist()
        baseline_difference_SD = baseline_df_this_seg['total_reward_difference_SD'].tolist()
        baseline_accumulate_reward_mean = baseline_df_this_seg['aveg_accumulated_reward'].tolist()
        baseline_accumulate_reward_SE = baseline_df_this_seg['accumulated_reward_SE'].tolist()
        baseline_accumulate_reward_SD = baseline_df_this_seg['accumulated_reward_SD'].tolist()
        baseline_total_final_reward_mean = baseline_df_this_seg['aveg_total_final_reward'].tolist()
        baseline_total_final_reward_SE = baseline_df_this_seg['total_final_reward_difference_SE'].tolist()
        baseline_total_final_reward_SD = baseline_df_this_seg['total_final_reward_difference_SD'].tolist()
        baseline_successful_rate = baseline_df_this_seg['successful_rate'].tolist()
        baseline_successful_rate_mean = [np.mean(baseline_df_this_seg['successful_rate'].tolist())]*len(baseline_successful_rate)
        baseline_successful_rate_SD = [np.std(baseline_df_this_seg['successful_rate'].tolist(), ddof=1)]*len(baseline_successful_rate)
        baseline_successful_rate_SE = [baseline_successful_rate_SD[0] / np.sqrt(len(baseline_df_this_seg['successful_rate'].tolist()))]*len(baseline_successful_rate)

        color = 'green'
        color_1 = 'black'
        color_2 = 'blue'
        start_idx = 0
        linewidth = 1.0
        fontsize = 30
        fig = plt.figure(figsize=(30, 100))
        plt.subplots_adjust(left=None, bottom=None, top=None, right=None, hspace=0.3, wspace=None)
        # plot 1:test_cf_pairwise_distance
        plt.subplot(5, 1, 1)
        plt.ticklabel_format(style='plain')
        r1 = list(map(lambda x: x[0] - x[1], zip(ddpg_distance_mean, ddpg_distance_SE)))
        r2 = list(map(lambda x: x[0] + x[1], zip(ddpg_distance_mean, ddpg_distance_SE)))
        x_value = range(1, len(ddpg_distance_mean) + 1)
        plt.plot(x_value, ddpg_distance_mean, color=color, label='cf_pairwise_distance_ddpg_mean', alpha=0.8,
                 linewidth=linewidth)
        plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
        # r1_std = list(map(lambda x: x[0] - x[1], zip(ddpg_distance_mean, ddpg_distance_SD)))
        # r2_std = list(map(lambda x: x[0] + x[1], zip(ddpg_distance_mean, ddpg_distance_SD)))
        # plt.fill_between(x_value, r1_std, r2_std, color=color_2, alpha=0.1)
        # draw baseline
        r1_b = list(map(lambda x: x[0] - x[1], zip(baseline_distance_mean, baseline_distance_SE)))
        r2_b = list(map(lambda x: x[0] + x[1], zip(baseline_distance_mean, baseline_distance_SE)))
        plt.plot(x_value, baseline_distance_mean, color=color_1, label='cf_pairwise_distance_baseline_mean', alpha=0.8, linewidth=linewidth)
        plt.fill_between(x_value, r1_b, r2_b, color=color_1, alpha=0.2)
        plt.xlabel('Train step (*{xaxis_interval})'.format(xaxis_interval=xaxis_interval), fontsize=fontsize)
        plt.ylabel('Value', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title('test_cf_pairwise_distance', fontsize=fontsize)

        # plot 2: total reward difference
        plt.subplot(5, 1, 2)
        plt.ticklabel_format(style='plain')
        r1 = list(map(lambda x: x[0] - x[1], zip(ddpg_difference_mean, ddpg_difference_SE)))
        r2 = list(map(lambda x: x[0] + x[1], zip(ddpg_difference_mean, ddpg_difference_SE)))
        x_value = range(1, len(ddpg_difference_mean) + 1)
        plt.plot(x_value, ddpg_difference_mean, color=color, label='total_reward_difference_ddpg_mean', alpha=0.8,
                 linewidth=linewidth)
        plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
        # r1_std = list(map(lambda x: x[0] - x[1], zip(ddpg_difference_mean, ddpg_difference_SD)))
        # r2_std = list(map(lambda x: x[0] + x[1], zip(ddpg_difference_mean, ddpg_difference_SD)))
        # plt.fill_between(x_value, r1_std, r2_std, color=color_2, alpha=0.1)
        # draw baseline
        r1_b = list(map(lambda x: x[0] - x[1], zip(baseline_difference_mean, baseline_difference_SE)))
        r2_b = list(map(lambda x: x[0] + x[1], zip(baseline_difference_mean, baseline_difference_SE)))
        plt.plot(x_value, baseline_difference_mean, color=color_1, label='total_reward_difference_baseline_mean', alpha=0.8,
                 linewidth=linewidth)
        plt.fill_between(x_value, r1_b, r2_b, color=color_1, alpha=0.2)
        plt.xlabel('Train step (*{xaxis_interval})'.format(xaxis_interval=xaxis_interval), fontsize=fontsize)
        plt.ylabel('Value', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title('test_total_reward_difference', fontsize=fontsize)

        # plot 3: total reward
        plt.subplot(5, 1, 3)
        plt.ticklabel_format(style='plain')
        r1 = list(map(lambda x: x[0] - x[1], zip(ddpg_accumulate_reward_mean, ddpg_accumulate_reward_SE)))
        r2 = list(map(lambda x: x[0] + x[1], zip(ddpg_accumulate_reward_mean, ddpg_accumulate_reward_SE)))
        x_value = range(1, len(ddpg_accumulate_reward_mean) + 1)
        plt.plot(x_value, ddpg_accumulate_reward_mean, color=color, label='total_reward_ddpg_mean', alpha=0.8, linewidth=linewidth)
        plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
        # r1_std = list(map(lambda x: x[0] - x[1], zip(ddpg_accumulate_reward_mean, ddpg_accumulate_reward_SD)))
        # r2_std = list(map(lambda x: x[0] + x[1], zip(ddpg_accumulate_reward_mean, ddpg_accumulate_reward_SD)))
        # plt.fill_between(x_value, r1_std, r2_std, color=color_2, alpha=0.1)
        # draw baseline
        r1_b = list(map(lambda x: x[0] - x[1], zip(baseline_accumulate_reward_mean, baseline_accumulate_reward_SE)))
        r2_b = list(map(lambda x: x[0] + x[1], zip(baseline_accumulate_reward_mean, baseline_accumulate_reward_SE)))
        plt.plot(x_value, baseline_accumulate_reward_mean, color=color_1, label='total_reward_baseline_mean', alpha=0.8, linewidth=linewidth)
        plt.fill_between(x_value, r1_b, r2_b, color=color_1, alpha=0.2)
        plt.xlabel('Train step (*{xaxis_interval})'.format(xaxis_interval=xaxis_interval), fontsize=fontsize)
        plt.ylabel('Value', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title('test_total_reward', fontsize=fontsize)

        # plot 4: total final reward
        plt.subplot(5, 1, 4)
        plt.ticklabel_format(style='plain')
        r1 = list(map(lambda x: x[0] - x[1], zip(ddpg_total_final_reward_mean, ddpg_total_final_reward_SE)))
        r2 = list(map(lambda x: x[0] + x[1], zip(ddpg_total_final_reward_mean, ddpg_total_final_reward_SE)))
        x_value = range(1, len(ddpg_total_final_reward_mean) + 1)
        plt.plot(x_value, ddpg_total_final_reward_mean, color=color, label='total_final_reward_ddpg_mean', alpha=0.8,
                 linewidth=linewidth)
        plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
        # r1_std = list(map(lambda x: x[0] - x[1], zip(ddpg_total_final_reward_mean, ddpg_accumulate_reward_SD)))
        # r2_std = list(map(lambda x: x[0] + x[1], zip(ddpg_total_final_reward_mean, ddpg_accumulate_reward_SD)))
        # plt.fill_between(x_value, r1_std, r2_std, color=color_2, alpha=0.1)
        # draw baseline
        r1_b = list(map(lambda x: x[0] - x[1], zip(baseline_total_final_reward_mean, baseline_total_final_reward_SE)))
        r2_b = list(map(lambda x: x[0] + x[1], zip(baseline_total_final_reward_mean, baseline_total_final_reward_SE)))
        plt.plot(x_value, baseline_total_final_reward_mean, color=color_1, label='total_final_reward_baseline_mean', alpha=0.8, linewidth=linewidth)
        plt.fill_between(x_value, r1_b, r2_b, color=color_1, alpha=0.2)
        plt.xlabel('Train step (*{xaxis_interval})'.format(xaxis_interval=xaxis_interval), fontsize=fontsize)
        plt.ylabel('Value', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title('test_total_final_reward',fontsize=fontsize)

        # plot 5: successful rate
        plt.subplot(5, 1, 5)
        plt.ticklabel_format(style='plain')
        # r1 = list(map(lambda x: x[0] - x[1], zip(ddpg_total_final_reward_mean, ddpg_total_final_reward_SE)))
        # r2 = list(map(lambda x: x[0] + x[1], zip(ddpg_total_final_reward_mean, ddpg_total_final_reward_SE)))
        x_value = range(1, len(ddpg_successful_rate) + 1)
        plt.plot(x_value, ddpg_successful_rate, color=color, label='successful_rate_ddpg', alpha=0.8, linewidth=linewidth)
        plt.plot(x_value, ddpg_successful_rate_mean, color=color, label='successful_rate_ddpg_mean', alpha=0.5, linewidth=linewidth*0.5, marker='*')
        r1 = list(map(lambda x: x[0] - x[1], zip(ddpg_successful_rate_mean, ddpg_successful_rate_SE)))
        r2 = list(map(lambda x: x[0] + x[1], zip(ddpg_successful_rate_mean, ddpg_successful_rate_SE)))
        plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
        # plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
        # r1_std = list(map(lambda x: x[0] - x[1], zip(ddpg_total_final_reward_mean, ddpg_accumulate_reward_SD)))
        # r2_std = list(map(lambda x: x[0] + x[1], zip(ddpg_total_final_reward_mean, ddpg_accumulate_reward_SD)))
        # plt.fill_between(x_value, r1_std, r2_std, color=color_2, alpha=0.1)
        # draw baseline
        # r1_b = list(map(lambda x: x[0] - x[1], zip(baseline_total_final_reward_mean, baseline_total_final_reward_SE)))
        # r2_b = list(map(lambda x: x[0] + x[1], zip(baseline_total_final_reward_mean, baseline_total_final_reward_SE)))
        plt.plot(x_value, baseline_successful_rate, color=color_1, label='successful_rate_baseline', alpha=0.8, linewidth=linewidth)
        plt.plot(x_value, baseline_successful_rate_mean, color=color_1, label='successful_rate_baseline_mean', alpha=0.5,
                 linewidth=linewidth*0.5, marker='*')
        r1_b = list(map(lambda x: x[0] - x[1], zip(baseline_successful_rate_mean, baseline_successful_rate_SE)))
        r2_b = list(map(lambda x: x[0] + x[1], zip(baseline_successful_rate_mean, baseline_successful_rate_SE)))
        plt.fill_between(x_value, r1_b, r2_b, color=color_1, alpha=0.2)
        plt.xlabel('Train step (*{xaxis_interval})'.format(xaxis_interval=xaxis_interval), fontsize=fontsize)
        plt.ylabel('Value', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title('test_successful_rate',fontsize=fontsize)

        file_path = '{folder}/test_curve_exp_{id}_trial_{trial_num}_ENV_{ENV_ID}_{trace_eps}_{trace_time_index}_{test_result_type}.png'.format(folder=save_folder, id=exp_id,trial_num=trial_num,
                                                                                         trace_eps=trace_eps,ENV_ID=ENV_ID,
                                                                                         trace_time_index=trace_current_step,test_result_type=test_result_type)
        plt.savefig(file_path)
        plt.close()
    return

def get_up_low_bound_from_test_result(all_test_result_df, all_train_id_list, col_name):
    up_list = []
    low_list = []
    aveg_list = []
    std_list = []
    for i in all_train_id_list:
        df_values = all_test_result_df[(all_test_result_df['train_round']==i)][col_name].tolist()
        up_list.append(max(df_values))
        low_list.append(min(df_values))
        aveg_list.append(np.mean(df_values))
        std_list.append(np.std(df_values, ddof=1))
    return up_list, low_list, aveg_list, std_list

def calculate_successful_rate(total_trial_number, all_train_round_list, result_folder, save_folder, exp_id, baseline_result_df):
    successful_rate_dict = {}
    successful_rate_df = pd.DataFrame(columns=['test_index', 'trial_index', 'successful_rate'])
    # baseline_result_path = '{result_folder}/baseline_results/accumulated_reward_test_baseline_trained_0_g_-10.0_0_30.csv'.format(result_folder=result_folder)
    # baseline_result_df = pd.read_csv(baseline_result_path)
    for train_round in all_train_round_list:
        for t in range(1, total_trial_number + 1):
            all_test_result_path = '{result_folder}/td3_cf_results/trial_{t}/test/all_test_result.csv'.format(result_folder=result_folder, t=t)
            all_test_result_df = pd.read_csv(all_test_result_path)
            test_result_this_round = all_test_result_df[all_test_result_df['train_round']==train_round]
            trace_better_than_orig = test_result_this_round[(test_result_this_round['difference'] > 0)]
            successful_rate_dict['test_index'] = train_round
            successful_rate_dict['trial_index'] = t
            successful_rate_dict['successful_rate'] = len(trace_better_than_orig)/len(test_result_this_round)
            baseline_trace_better_than_orig = baseline_result_df[(baseline_result_df['difference'] > 0)]
            successful_rate_dict['baseline_successful_rate'] = len(baseline_trace_better_than_orig) / len(baseline_result_df)
            successful_rate_df = pd.concat([successful_rate_df, pd.DataFrame([successful_rate_dict])], ignore_index=True)

    successful_rate_df.to_csv('{save_folder}/successful_rate.csv'.format(save_folder=save_folder))

    draw_successful_rate(successful_rate_df, all_train_round_list, save_folder, exp_id, total_trial_number)

    return

def draw_successful_rate_0(successful_rate_df, all_train_round_list, figure_folder, exp_id, total_trial_num=5):
    mean_successful_rate_list = []
    upper_std_list = []
    lower_std_list = []
    upper_se_list = []
    lower_se_list = []
    successful_rate_value_dict = {}
    for test_id in all_train_round_list:
        all_trial_value_this_test = successful_rate_df[successful_rate_df['test_index']==test_id]['successful_rate'].tolist()
        mean_success_rate = np.mean(all_trial_value_this_test)
        std_success_rate = np.std(all_trial_value_this_test, ddof=1)
        SE = std_success_rate/np.sqrt(len(all_trial_value_this_test))
        mean_successful_rate_list.append(mean_success_rate)
        if mean_success_rate+std_success_rate>=1:
            upper_std_list.append(1.0)
        else:
            upper_std_list.append(mean_success_rate+std_success_rate)
        if mean_success_rate - std_success_rate <=0.0:
            lower_std_list.append(0.0)
        else:
            lower_std_list.append(mean_success_rate-std_success_rate)
        if mean_success_rate + SE >= 1:
            upper_se_list.append(1.0)
        else:
            upper_se_list.append(mean_success_rate + SE)
        if mean_success_rate - SE <= 0.0:
            lower_se_list.append(0.0)
        else:
            lower_se_list.append(mean_success_rate - SE)
    for trial_id in range(1, total_trial_num+1):
        successful_rate_this_trial = successful_rate_df[successful_rate_df['trial_index']==trial_id]['successful_rate'].tolist()
        successful_rate_value_dict[trial_id] = successful_rate_this_trial
    # print(test_df)
    color = 'green'
    color_2 = 'blue'
    fig = plt.figure(figsize=(25, 25))
    plt.ticklabel_format(style='plain')
    plt.subplots_adjust(left=None, bottom=None, top=None, right=None, hspace=0.3, wspace=None)
    plt.subplot(2, 1, 1)
    x_value = range(1, len(mean_successful_rate_list) + 1)
    plt.plot(x_value, mean_successful_rate_list, color=color, label='successful_rate_ddpg_mean', alpha=0.8, linewidth=0.5)
    plt.fill_between(x_value, lower_se_list, upper_se_list, color=color, alpha=0.2)
    plt.fill_between(x_value, lower_std_list, upper_std_list, color=color_2, alpha=0.1)
    plt.plot(x_value, [successful_rate_df['baseline_successful_rate'][0]] * len(x_value),
             label='successful_rate_baseline', alpha=1.0, linewidth=1.0, color='black')
    plt.xlabel('Train step (*1000)')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Successful_rate_aveg')

    plt.subplot(2, 1, 2)
    for trial_id in range(1, total_trial_num+1):
        x_value = range(1, len(successful_rate_value_dict[trial_id])+1)
        plt.plot(x_value, successful_rate_value_dict[trial_id], label='successful_rate_ddpg_trial_{trial_id}'.format(trial_id=trial_id), alpha=0.8, linewidth=0.5)
    plt.plot(x_value, [successful_rate_df['baseline_successful_rate'][0]]*len(x_value), label='successful_rate_baseline', alpha=1.0, linewidth=1.0, color='black')
    plt.xlabel('Train step (*1000)')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Successful_rate_each_trial')

    file_path = '{folder}/sucessful_rate_{exp_id}.png'.format(folder=figure_folder,exp_id=exp_id)
    plt.savefig(file_path)
    plt.close()
    return

def draw_successful_rate_all_trial(data_folder, total_trial_number, all_train_round_list, save_folder, exp_id, all_test_index_df,
                                   xaxis_interval, test_result_type='test', case_name='lunar_lander'):
    # successful_rate_dict = {}
    # successful_rate_df = pd.DataFrame(columns=['orig_episode', 'orig_end_time_index','test_index', 'trial_index',
    #                                            'successful_rate_mean', 'successful_rate_SE', 'successful_rate_SD'])
    ddpg_data_list = []
    baseline_data_list = []
    all_trial_col = []
    if case_name == 'lunar_lander':
        env_index_col = 'gravity'
    else:
        env_index_col = 'ENV_ID'
    all_col_name = ['orig_trace_episode', 'orig_end_step', 'model_type', 'train_round', env_index_col]
    for t in range(1, total_trial_number + 1):
        new_col_name = 'successful_rate_trial_{t}'.format(t=t)
        all_trial_col.append(new_col_name)
        all_col_name.append(new_col_name)
        all_test_statistic_ddpg_path = '{result_folder}/td3_cf_results/trial_{t}/{test_result_type}/all_test_statistic_ddpg.csv'.format(
            result_folder=data_folder, t=t,test_result_type=test_result_type)
        all_test_statistic_ddpg_df = pd.read_csv(all_test_statistic_ddpg_path)
        all_test_statistic_baseline_path = '{result_folder}/td3_cf_results/trial_{t}/{test_result_type}/all_test_statistic_baseline.csv'.format(
            result_folder=data_folder, t=t,test_result_type=test_result_type)
        all_test_statistic_baseline_df = pd.read_csv(all_test_statistic_baseline_path)
        ddpg_data_df = all_test_statistic_ddpg_df[all_test_statistic_ddpg_df['train_round'].isin(all_train_round_list)][['orig_trace_episode', 'orig_end_step','model_type','train_round', env_index_col,'successful_rate']].rename(columns={'successful_rate':new_col_name})
        baseline_data_df = all_test_statistic_baseline_df[all_test_statistic_baseline_df['train_round'].isin(all_train_round_list)][['orig_trace_episode', 'orig_end_step','model_type', 'train_round', env_index_col,'successful_rate']].rename(columns={'successful_rate':new_col_name})
        ddpg_data_list.append(ddpg_data_df)
        baseline_data_list.append(baseline_data_df)
    ddpg_data_all_seg = pd.concat(ddpg_data_list, axis=1)
    baseline_data_all_seg = pd.concat(baseline_data_list, axis=1)
    #print('ddpg_data_all_seg before: ', ddpg_data_all_seg)
    # ddpg_data_all_seg = ddpg_data_all_seg.T.drop_duplicates().T
    # baseline_data_all_seg = baseline_data_all_seg.T.drop_duplicates().T
    ddpg_data_all_seg = ddpg_data_all_seg.loc[:,~ddpg_data_all_seg.columns.duplicated()]
    baseline_data_all_seg = baseline_data_all_seg.loc[:,~baseline_data_all_seg.columns.duplicated()]
    #print('ddpg_data_all_seg after: ', ddpg_data_all_seg)

    new_ddpg_df_list = []
    new_baseline_df_list = []
    for item, row in all_test_index_df.iterrows():
        if case_name=='lunar_lander':
            trace_eps, trace_current_step, ENV_ID = row['orig_episode'], row['orig_end_time_index'], row['gravity']
            ddpg_data_this_seg = ddpg_data_all_seg[(ddpg_data_all_seg['orig_trace_episode']==trace_eps)&
                                                    (ddpg_data_all_seg['orig_end_step']==trace_current_step)&
                                                    (ddpg_data_all_seg['gravity']==ENV_ID)]
            baseline_data_this_seg = baseline_data_all_seg[(baseline_data_all_seg['orig_trace_episode'] == trace_eps) &
                                                   (baseline_data_all_seg['orig_end_step'] == trace_current_step)&
                                                    (baseline_data_all_seg['gravity'] == ENV_ID)]
        else:
            trace_eps, trace_current_step, ENV_ID = row['orig_episode'], row['orig_end_time_index'], row['ENV_NAME']
            ddpg_data_this_seg = ddpg_data_all_seg[(ddpg_data_all_seg['orig_trace_episode'] == trace_eps) &
                                                   (ddpg_data_all_seg['orig_end_step'] == trace_current_step) &
                                                   (ddpg_data_all_seg['ENV_ID'] == ENV_ID)]
            baseline_data_this_seg = baseline_data_all_seg[(baseline_data_all_seg['orig_trace_episode'] == trace_eps) &
                                                           (baseline_data_all_seg['orig_end_step'] == trace_current_step) &
                                                           (baseline_data_all_seg['ENV_ID'] == ENV_ID)]
        #print('ddpg_data_this_seg: ', ddpg_data_this_seg)
        ddpg_data_this_seg['ENV_ID'] = ENV_ID
        ddpg_data_this_seg['successful_rate_mean'] = ddpg_data_this_seg[all_trial_col].mean(axis=1)
        ddpg_data_this_seg['successful_rate_SD'] = ddpg_data_this_seg[all_trial_col].std(axis=1)
        ddpg_data_this_seg['successful_rate_SE'] = ddpg_data_this_seg['successful_rate_SD']/total_trial_number
        baseline_data_this_seg['ENV_ID'] = ENV_ID
        baseline_data_this_seg['successful_rate_mean'] = baseline_data_this_seg[all_trial_col].mean(axis=1)
        baseline_data_this_seg['successful_rate_SD'] = baseline_data_this_seg[all_trial_col].std(axis=1)
        baseline_data_this_seg['successful_rate_SE'] = baseline_data_this_seg['successful_rate_SD'] / total_trial_number
        new_ddpg_df_list.append(ddpg_data_this_seg)
        new_baseline_df_list.append(baseline_data_this_seg)

        fig = plt.figure(figsize=(30, 10))
        plt.ticklabel_format(style='plain')
        color = 'green'
        color_1 = 'black'
        x_value = range(1, len(ddpg_data_this_seg)+1)
        plt.plot(x_value, ddpg_data_this_seg['successful_rate_mean'].tolist(), label='successful_rate_mean_ddpg', alpha=1.0, linewidth=1.0, color=color)
        r1 = list(map(lambda x: x[0] - x[1], zip(ddpg_data_this_seg['successful_rate_mean'].tolist(), ddpg_data_this_seg['successful_rate_SE'].tolist())))
        r2 = list(map(lambda x: x[0] + x[1], zip(ddpg_data_this_seg['successful_rate_mean'].tolist(), ddpg_data_this_seg['successful_rate_SE'].tolist())))
        plt.fill_between(x_value, r1, r2, color=color, alpha=0.2)
        plt.plot(x_value, baseline_data_this_seg['successful_rate_mean'].tolist(), label='successful_rate_mean_baseline',
                 alpha=1.0, linewidth=1.0, color=color_1)
        r1 = list(map(lambda x: x[0] - x[1], zip(baseline_data_this_seg['successful_rate_mean'].tolist(),
                                                 baseline_data_this_seg['successful_rate_SE'].tolist())))
        r2 = list(map(lambda x: x[0] + x[1], zip(baseline_data_this_seg['successful_rate_mean'].tolist(),
                                                 baseline_data_this_seg['successful_rate_SE'].tolist())))
        plt.fill_between(x_value, r1, r2, color=color_1, alpha=0.2)
        plt.xlabel('Train step (*{xaxis_interval})'.format(xaxis_interval=xaxis_interval))
        plt.ylabel('Value')
        plt.legend()
        plt.title('Successful_rate_all_trial_{trace_eps}_{trace_current_step}'.format(trace_eps=trace_eps,
                                                                                      trace_current_step=trace_current_step))

        file_path = '{folder}/successful_rate_all_trial_{id}_ENV_{ENV_ID}_{trace_eps}_{trace_time_index}_{test_result_type}.png'.format(folder=save_folder, id=exp_id,
                                                                                         trace_eps=trace_eps,ENV_ID=ENV_ID,
                                                                                         trace_time_index=trace_current_step,test_result_type=test_result_type)
        plt.savefig(file_path)
        plt.close()

    new_ddpg_data = pd.concat(new_ddpg_df_list)
    new_baseline_data = pd.concat(new_baseline_df_list)
    new_ddpg_data.to_csv('{save_folder}/successful_rate_all_trial_ddpg_{test_result_type}.csv'.format(save_folder=save_folder,test_result_type=test_result_type))
    new_baseline_data.to_csv('{save_folder}/successful_rate_all_trial_baseline_{test_result_type}.csv'.format(save_folder=save_folder,test_result_type=test_result_type))
    return

def get_metric_each_trial(data_folder, total_trial_number, all_train_round_list, save_folder, exp_id, all_test_index_df,
                          total_test_trace_num, xaxis_interval, total_trial_num, exp_type, case_name='lunar_lander',
                          start_index=120,interval=20, compare_type='with_pointwise', best_baseline_test_index=None,
                          best_baseline_trial_index=None, test_result_type='test'):
    # metric_summary_df = pd.DataFrame(columns=['test_index', 'trial_index','orig_trace_episode', 'orig_end_step',
    #                                           'P_hr_ddpg', 'P_hr_baseline', 'P_better', 'r_d'])
    metric_dict_each_trace_list = []
    metric_dict_all_trace_list = []
    for t in range(1, total_trial_number + 1):
        all_test_statistic_ddpg_path = '{result_folder}/td3_cf_results/trial_{t}/{test_result_type}/all_test_statistic_ddpg.csv'.format(result_folder=data_folder, test_result_type=test_result_type, t=t)
        all_test_statistic_ddpg_df = pd.read_csv(all_test_statistic_ddpg_path)
        all_test_result_ddpg_path = '{result_folder}/td3_cf_results/trial_{t}/{test_result_type}/all_test_result_ddpg.csv'.format(result_folder=data_folder,test_result_type=test_result_type, t=t)
        all_test_result_ddpg_df = pd.read_csv(all_test_result_ddpg_path, index_col=0)

        if compare_type=='with_pointwise':
            all_test_statistic_baseline_path = '{result_folder}/td3_cf_results/trial_{t}/{test_result_type}/all_test_statistic_baseline.csv'.format(result_folder=data_folder,test_result_type=test_result_type, t=t)
            all_test_statistic_baseline_df = pd.read_csv(all_test_statistic_baseline_path)
            all_test_result_baseline_path = '{result_folder}/td3_cf_results/trial_{t}/{test_result_type}/all_test_result_baseline.csv'.format(result_folder=data_folder,test_result_type=test_result_type, t=t)
            all_test_result_baseline_df = pd.read_csv(all_test_result_baseline_path, index_col=0)
        else:
            all_test_statistic_baseline_path = '{result_folder}/td3_cf_results/trial_{t}/{test_result_type}/all_test_statistic_baseline.csv'.format(
                result_folder=data_folder, t=best_baseline_trial_index, test_result_type=test_result_type)
            all_test_statistic_baseline_df = pd.read_csv(all_test_statistic_baseline_path)
            all_test_result_baseline_path = '{result_folder}/td3_cf_results/trial_{t}/{test_result_type}/all_test_result_baseline.csv'.format(
                result_folder=data_folder, t=best_baseline_trial_index, test_result_type=test_result_type)
            all_test_result_baseline_df = pd.read_csv(all_test_result_baseline_path, index_col=0)

        # all_test_result_ddpg_df['total_reward_improve_perc'] = all_test_result_ddpg_df.apply(lambda x: x['difference'] / abs(x['orig_accumulated_reward']), axis=1)
        # all_test_result_baseline_df['total_reward_improve_perc'] = all_test_result_baseline_df.apply(lambda x: x['difference'] / abs(x['orig_accumulated_reward']), axis=1)
        # all_test_result_ddpg_df.to_csv(all_test_result_ddpg_path)
        # all_test_result_baseline_df.to_csv(all_test_result_baseline_path)

        for test_index in all_train_round_list:
            P_hr_ddpg, P_hr_baseline, P_better, r_d_list = 0, 0, 0, []
            compare_ratio_list, compare_count_list = [], []
            compare_count = 0
            for item, row in all_test_index_df.iterrows():
                ddpg_better_than_orig, baseline_better_than_orig, ddpg_better_than_baseline, r_d = 0, 0, 0, 0
                if case_name == 'lunar_lander':
                    trace_eps, trace_current_step, ENV_ID = row['orig_episode'], row['orig_end_time_index'], row['gravity']
                    ddpg_statistic_this_seg = all_test_statistic_ddpg_df[
                        (all_test_statistic_ddpg_df['orig_trace_episode'] == trace_eps) &
                        (all_test_statistic_ddpg_df['orig_end_step'] == trace_current_step) &
                        (all_test_statistic_ddpg_df['gravity'] == ENV_ID)]
                    baseline_statistic_this_seg = all_test_statistic_baseline_df[
                        (all_test_statistic_baseline_df['orig_trace_episode'] == trace_eps) &
                        (all_test_statistic_baseline_df['orig_end_step'] == trace_current_step) &
                        (all_test_statistic_baseline_df['gravity'] == ENV_ID)]
                    ddpg_result_this_seg = all_test_result_ddpg_df[
                        (all_test_result_ddpg_df['orig_trace_episode'] == trace_eps) &
                        (all_test_result_ddpg_df['orig_end_step'] == trace_current_step) &
                        (all_test_result_ddpg_df['gravity'] == ENV_ID)]
                    baseline_result_this_seg = all_test_result_baseline_df[
                        (all_test_result_baseline_df['orig_trace_episode'] == trace_eps) &
                        (all_test_result_baseline_df['orig_end_step'] == trace_current_step) &
                        (all_test_result_baseline_df['gravity'] == ENV_ID)]
                else:
                    trace_eps, trace_current_step, ENV_ID = row['orig_episode'], row['orig_end_time_index'], row['ENV_NAME']
                    ddpg_statistic_this_seg = all_test_statistic_ddpg_df[
                        (all_test_statistic_ddpg_df['orig_trace_episode'] == trace_eps) &
                        (all_test_statistic_ddpg_df['orig_end_step'] == trace_current_step) &
                        (all_test_statistic_ddpg_df['ENV_ID'] == ENV_ID)]
                    baseline_statistic_this_seg = all_test_statistic_baseline_df[
                        (all_test_statistic_baseline_df['orig_trace_episode'] == trace_eps) &
                        (all_test_statistic_baseline_df['orig_end_step'] == trace_current_step) &
                        (all_test_statistic_baseline_df['ENV_ID'] == ENV_ID)]
                    ddpg_result_this_seg = all_test_result_ddpg_df[
                        (all_test_result_ddpg_df['orig_trace_episode'] == trace_eps) &
                        (all_test_result_ddpg_df['orig_end_step'] == trace_current_step) &
                        (all_test_result_ddpg_df['ENV_ID'] == ENV_ID)]
                    baseline_result_this_seg = all_test_result_baseline_df[
                        (all_test_result_baseline_df['orig_trace_episode'] == trace_eps) &
                        (all_test_result_baseline_df['orig_end_step'] == trace_current_step) &
                        (all_test_result_baseline_df['ENV_ID'] == ENV_ID)]


                #trace_eps, trace_current_step = row['orig_episode'], row['orig_end_time_index']
                # ddpg_statistic_this_seg = all_test_statistic_ddpg_df[(all_test_statistic_ddpg_df['orig_trace_episode'] == trace_eps) &
                #                                        (all_test_statistic_ddpg_df['orig_end_step'] == trace_current_step)&
                #                                         (all_test_statistic_ddpg_df['gravity'] == ENV_ID)]
                # baseline_statistic_this_seg = all_test_statistic_baseline_df[(all_test_statistic_baseline_df['orig_trace_episode'] == trace_eps) &
                #                                                (all_test_statistic_baseline_df['orig_end_step'] == trace_current_step)&
                #                                                 (all_test_statistic_baseline_df['gravity'] == ENV_ID)]
                # ddpg_result_this_seg = all_test_result_ddpg_df[
                #     (all_test_result_ddpg_df['orig_trace_episode'] == trace_eps) &
                #     (all_test_result_ddpg_df['orig_end_step'] == trace_current_step)&
                #     (all_test_result_ddpg_df['gravity'] == ENV_ID)]
                # baseline_result_this_seg = all_test_result_baseline_df[
                #     (all_test_result_baseline_df['orig_trace_episode'] == trace_eps) &
                #     (all_test_result_baseline_df['orig_end_step'] == trace_current_step)&
                # (all_test_result_baseline_df['gravity'] == ENV_ID)]
                if compare_type=='with_pointwise':
                    baseline_stat_this_seg_this_test = baseline_statistic_this_seg[baseline_statistic_this_seg['train_round'] == test_index]
                    baseline_result_this_seg_this_test = baseline_result_this_seg[baseline_result_this_seg['train_round'] == test_index]
                else:
                    baseline_stat_this_seg_this_test = baseline_statistic_this_seg[baseline_statistic_this_seg['train_round'] == best_baseline_test_index]
                    baseline_result_this_seg_this_test = baseline_result_this_seg[baseline_result_this_seg['train_round'] == best_baseline_test_index]

                ddpg_stat_this_seg_this_test = ddpg_statistic_this_seg[ddpg_statistic_this_seg['train_round'] == test_index]
                ddpg_result_this_seg_this_test = ddpg_result_this_seg[ddpg_result_this_seg['train_round'] == test_index]

                if ddpg_stat_this_seg_this_test['successful_rate'].tolist()[0] > 0.0:
                    P_hr_ddpg += 1
                    ddpg_better_than_orig = 1
                if baseline_stat_this_seg_this_test['successful_rate'].tolist()[0] > 0.0:
                    P_hr_baseline += 1
                    baseline_better_than_orig = 1
                if ddpg_stat_this_seg_this_test['successful_rate'].tolist()[0] > 0.0 and \
                        baseline_stat_this_seg_this_test['successful_rate'].tolist()[0] == 0.0:
                    P_better += 1
                    ddpg_better_than_baseline = 1
                    r_d = -99999
                elif ddpg_stat_this_seg_this_test['successful_rate'].tolist()[0] > 0.0 and baseline_stat_this_seg_this_test['successful_rate'].tolist()[0] > 0.0:
                    ddpg_lowest_dist = ddpg_result_this_seg_this_test[ddpg_result_this_seg_this_test['difference'] > 0.0]['cf_pairwise_distance'].min()
                    ddpg_best_outcome_for_lowest_dist = ddpg_result_this_seg_this_test[ddpg_result_this_seg_this_test['cf_pairwise_distance'] == ddpg_lowest_dist]['difference'].max()
                    baseline_lowest_dist = baseline_result_this_seg_this_test[baseline_result_this_seg_this_test['difference'] > 0.0]['cf_pairwise_distance'].min()
                    baseline_best_outcome_for_lowest_dist = baseline_result_this_seg_this_test[baseline_result_this_seg_this_test['cf_pairwise_distance'] == ddpg_lowest_dist]['difference'].max()
                    if (ddpg_lowest_dist < baseline_lowest_dist) and (ddpg_best_outcome_for_lowest_dist >= baseline_best_outcome_for_lowest_dist):
                        P_better += 1
                        ddpg_better_than_baseline = 1
                    r_d = ddpg_lowest_dist / baseline_lowest_dist
                    r_d_list.append(r_d)
                else:
                    r_d = 99999
                    pass
                # calculate total reward improvement percentage-average, sd, se
                ddpg_total_reward_improve_perc_df = ddpg_result_this_seg_this_test[ddpg_result_this_seg_this_test['percentage'] > 0.0]
                if len(ddpg_total_reward_improve_perc_df)>0:
                    lowest_dist_ddpg = ddpg_total_reward_improve_perc_df['cf_pairwise_distance'].min()
                    best_total_r_imp_for_low_d_ddpg = ddpg_total_reward_improve_perc_df[ddpg_total_reward_improve_perc_df['cf_pairwise_distance']==lowest_dist_ddpg]['percentage'].max()
                else:
                    lowest_dist_ddpg = 99999
                    best_total_r_imp_for_low_d_ddpg = -99999
                baseline_total_reward_improve_perc_df = baseline_result_this_seg_this_test[baseline_result_this_seg_this_test['percentage'] > 0.0]
                if len(baseline_total_reward_improve_perc_df) > 0:
                    lowest_dist_baseline = baseline_total_reward_improve_perc_df['cf_pairwise_distance'].min()
                    best_total_r_imp_for_low_d_baseline = baseline_total_reward_improve_perc_df[baseline_total_reward_improve_perc_df['cf_pairwise_distance'] == lowest_dist_baseline]['percentage'].max()
                else:
                    lowest_dist_baseline = 99999
                    best_total_r_imp_for_low_d_baseline = -99999
                if (best_total_r_imp_for_low_d_ddpg != -99999) and (best_total_r_imp_for_low_d_baseline != -99999) and (lowest_dist_ddpg!=0) and (lowest_dist_baseline!=0):
                    compare_ratio = (best_total_r_imp_for_low_d_ddpg/best_total_r_imp_for_low_d_baseline) / (lowest_dist_ddpg/lowest_dist_baseline)
                    compare_ratio_list.append(compare_ratio)
                    if compare_ratio > 1.0:
                        compare_count += 1
                else:
                    compare_ratio = -99999
                dict = {'test_index': test_index, 'trial_index': t,'ENV_ID':ENV_ID,
                                'orig_trace_episode': trace_eps,'orig_end_step': trace_current_step,
                                'ddpg_better_than_orig': ddpg_better_than_orig,
                                'baseline_better_than_orig': baseline_better_than_orig,
                                'ddpg_better_than_baseline': ddpg_better_than_baseline,
                        'aveg_TD3_action_count': ddpg_stat_this_seg_this_test['aveg_TD3_action_count'].tolist()[0],
                        'r_d': r_d,
                        'ddpg_total_reward_improve_perc_aveg_this_trace':ddpg_total_reward_improve_perc_df['percentage'].mean() if len(ddpg_total_reward_improve_perc_df)>0 else -99999,
                        'baseline_total_reward_improve_perc_aveg_this_trace': baseline_total_reward_improve_perc_df['percentage'].mean() if len(baseline_total_reward_improve_perc_df)>0 else -99999,
                        'ddpg_cf_distance_aveg_this_trace': ddpg_total_reward_improve_perc_df['cf_pairwise_distance'].mean() if len(ddpg_total_reward_improve_perc_df)>0 else 99999,
                        'baseline_cf_distance_aveg_this_trace': baseline_total_reward_improve_perc_df['cf_pairwise_distance'].mean() if len(baseline_total_reward_improve_perc_df)>0 else 99999,
                        'ddpg_total_reward_improve_perc_Best_CF_this_trace':best_total_r_imp_for_low_d_ddpg,
                        'ddpg_cf_distance_Best_CF_this_trace': lowest_dist_ddpg,
                        'baseline_total_reward_improve_perc_Best_CF_this_trace': best_total_r_imp_for_low_d_baseline,
                        'baseline_cf_distance_Best_CF_this_trace': lowest_dist_baseline,
                        'compare_ratio':compare_ratio,'best_baseline_test_index':best_baseline_test_index, 'best_baseline_trial_index':best_baseline_trial_index,
                        }
                metric_dict_each_trace_list.append(dict)
            dict_summary = {'test_index': test_index, 'trial_index': t,
                        'P_hr_ddpg': P_hr_ddpg / total_test_trace_num,
                        'P_hr_baseline': P_hr_baseline / total_test_trace_num,
                        'P_better': P_better / total_test_trace_num, 'r_d_aveg': np.mean(r_d_list),
                        'r_d_SD': np.std(r_d_list, ddof=1), 'r_d_SE': np.std(r_d_list, ddof=1)/np.sqrt(len(r_d_list)),
                            'compare_ratio_aveg':np.mean(compare_ratio_list) if len(compare_ratio_list) else -1.0,
                            'compare_count': compare_count,
                            'compare_count_perc': compare_count/len(compare_ratio_list) if len(compare_ratio_list) else -1.0,
                             'best_baseline_test_index':best_baseline_test_index, 'best_baseline_trial_index':best_baseline_trial_index
                        }
            metric_dict_all_trace_list.append(dict_summary)

    metric_summary_df = pd.DataFrame(metric_dict_each_trace_list)
    metric_summary_df['imp_ratio'] = metric_summary_df['ddpg_total_reward_improve_perc_Best_CF_this_trace']/metric_summary_df['baseline_total_reward_improve_perc_Best_CF_this_trace']
    metric_summary_df['distance_ratio'] = metric_summary_df['ddpg_cf_distance_Best_CF_this_trace'] / metric_summary_df['baseline_cf_distance_Best_CF_this_trace']
    metric_summary_df.to_csv('{save_folder}/metric_summary_each_trace_{compare_type}_exp_{exp_id}_{test_result_type}.csv'.format(save_folder=save_folder,
                                                                                                                                 compare_type=compare_type,exp_id=exp_id,
                                                                                                                                 test_result_type=test_result_type))
    metric_summary_df_1 = pd.DataFrame(metric_dict_all_trace_list)
    metric_summary_df_1.to_csv('{save_folder}/metric_summary_all_trace_{compare_type}_exp_{exp_id}_{test_result_type}.csv'.format(save_folder=save_folder,
                                                                                                                                  compare_type=compare_type,exp_id=exp_id,
                                                                                                                                  test_result_type=test_result_type))
    chosen_test_result = draw_metric(metric_summary_df_1, metric_summary_df, save_folder, total_trial_num,
                                     exp_id, xaxis_interval, all_train_round_list, case_name, exp_type,start_index,interval, compare_type, test_result_type)
    return chosen_test_result

def draw_metric(metric_summary_df_1, metric_summary_df, save_folder, total_trial_num, exp_id, xaxis_interval,all_train_round_list,
                case_name, exp_type,start_index,interval, compare_type,test_result_type):
    metric_name_list = ['P_hr_ddpg', 'P_hr_baseline', 'P_better', 'compare_ratio_aveg', 'compare_count', 'compare_count_perc']
    aveg_P_hr_ddpg_list, aveg_P_hr_baseline_list, aveg_P_better_list = [], [], []
    P_hr_ddpg_SD_list, P_hr_baseline_SD_list, P_better_SD_list = [], [], []
    P_hr_ddpg_SE_list, P_hr_baseline_SE_list, P_better_SE_list = [], [], []
    compare_ratio_aveg_list,  compare_ratio_SE_list, compare_ratio_SD_list = [], [], []
    compare_count_aveg_list,  compare_count_SE_list, compare_count_SD_list = [], [], []
    for t in range(1, total_trial_num+1):
        fig = plt.figure(figsize=(20, 45))
        plt.subplot(3, 1, 1)
        for metric in metric_name_list[:2]:
            data = metric_summary_df_1[(metric_summary_df_1['trial_index']==t)][metric].tolist()
            x_value = range(1, len(data)+1)
            plt.plot(x_value, data, label='{metric}_trial_{t}'.format(metric=metric,t=t), alpha=1.0, linewidth=1.0)
        plt.xlabel('Train step (*{xaxis_interval})'.format(xaxis_interval=xaxis_interval))
        plt.ylabel('Percentage')
        plt.legend()

        plt.subplot(3, 1, 2)
        data = metric_summary_df_1[(metric_summary_df_1['trial_index'] == t)]['compare_count'].tolist()
        x_value = range(1, len(data) + 1)
        plt.plot(x_value, data, label='{metric}_trial_{t}'.format(metric='compare_count', t=t), alpha=1.0,
                 linewidth=1.0)
        plt.legend()

        plt.subplot(3, 1, 3)
        data = metric_summary_df_1[(metric_summary_df_1['trial_index'] == t)]['compare_count_perc'].tolist()
        x_value = range(1, len(data) + 1)
        plt.plot(x_value, data, label='{metric}_trial_{t}'.format(metric='compare_count_perc', t=t), alpha=1.0, linewidth=1.0)
        plt.legend()

        plt.title('metric_trial_{t}'.format(t=t))
        file_path = '{folder}/metric_exp_{id}_trial_{t}_{compare_type}_{test_result_type}.png'.format(folder=save_folder, id=exp_id, t=t,
                                                                                                      compare_type=compare_type,test_result_type=test_result_type)
        plt.savefig(file_path)
        plt.close()

    for test_id in all_train_round_list:
        data_df = metric_summary_df_1[metric_summary_df_1['test_index']==test_id]
        aveg_P_hr_ddpg_list.append(data_df['P_hr_ddpg'].mean())
        aveg_P_hr_baseline_list.append(data_df['P_hr_baseline'].mean())
        aveg_P_better_list.append(data_df['P_better'].mean())

        compare_ratio_aveg_list.append(data_df[data_df['compare_count_perc']!=-1.0]['compare_count_perc'].mean())
        compare_ratio_SD = data_df[data_df['compare_count_perc']!=-1.0]['compare_count_perc'].std(ddof=1)
        compare_ratio_SE = compare_ratio_SD/np.sqrt(total_trial_num)
        compare_ratio_SE_list.append(compare_ratio_SE)
        compare_ratio_SD_list.append(compare_ratio_SD)

        compare_count_aveg_list.append(data_df['compare_count'].mean())
        compare_count_SD = data_df['compare_count'].std(ddof=1)
        compare_count_SE = compare_count_SD / np.sqrt(total_trial_num)
        compare_count_SE_list.append(compare_count_SE)
        compare_count_SD_list.append(compare_count_SD)

        P_hr_ddpg_SD = data_df['P_hr_ddpg'].std(ddof=1)
        P_hr_baseline_SD = data_df['P_hr_baseline'].std(ddof=1)
        P_better_SD = data_df['P_better'].std(ddof=1)
        P_hr_ddpg_SE = P_hr_ddpg_SD/np.sqrt(total_trial_num)
        P_hr_baseline_SE = P_hr_baseline_SD/np.sqrt(total_trial_num)
        P_better_SE = P_better_SD/np.sqrt(total_trial_num)
        P_hr_ddpg_SD_list.append(P_hr_ddpg_SD)
        P_hr_baseline_SD_list.append(P_hr_baseline_SD)
        P_better_SD_list.append(P_better_SD)
        P_hr_ddpg_SE_list.append(P_hr_ddpg_SE)
        P_hr_baseline_SE_list.append(P_hr_baseline_SE)
        P_better_SE_list.append(P_better_SE)

    all_metric_aveg_df = pd.DataFrame()
    all_metric_aveg_df['test_index'] = all_train_round_list
    all_metric_aveg_df['P_hr_ddpg'] = aveg_P_hr_ddpg_list
    all_metric_aveg_df['P_hr_ddpg_SD'] = P_hr_ddpg_SD_list
    all_metric_aveg_df['P_hr_ddpg_SE'] = P_hr_ddpg_SE_list
    all_metric_aveg_df['P_hr_baseline'] = aveg_P_hr_baseline_list
    all_metric_aveg_df['P_hr_baseline_SD'] = P_hr_baseline_SD_list
    all_metric_aveg_df['P_hr_baseline_SE'] = P_hr_baseline_SE_list
    all_metric_aveg_df['P_better'] = aveg_P_better_list
    all_metric_aveg_df['P_better_SD'] = P_better_SD_list
    all_metric_aveg_df['P_better_SE'] = P_better_SE_list
    all_metric_aveg_df['compare_count_perc'] = compare_ratio_aveg_list
    all_metric_aveg_df['compare_count_perc_SD'] = compare_ratio_SD_list
    all_metric_aveg_df['compare_count_perc_SE'] = compare_ratio_SE_list
    all_metric_aveg_df['compare_count'] = compare_count_aveg_list
    all_metric_aveg_df['compare_count_SD'] = compare_count_SD_list
    all_metric_aveg_df['compare_count_SE'] = compare_count_SE_list
    all_metric_aveg_df.to_csv('{save_folder}/all_metric_aveg_{compare_type}_{test_result_type}.csv'.format(save_folder=save_folder,
                                                                                                           compare_type=compare_type,test_result_type=test_result_type))

    fig = plt.figure(figsize=(20, 45))
    plt.ticklabel_format(style='plain')
    color = 'green'
    color_1 = 'black'
    color_2 = 'blue'
    fontsize = 25
    #x_value = range(1, len(all_train_round_list) + 1)
    x_up = len(all_train_round_list) #30
    x_value_list = list(range(1, x_up+1))
    x_value = [x*(xaxis_interval) for x in x_value_list]
    # print('x_up: ', x_up)
    # print('x_value_list: ', x_value_list)
    # print('x_value: ', x_value)
    plt.rcParams['font.size'] = fontsize
    plt.subplot(3, 1, 1)
    #plt.tick_params(labelsize=fontsize)
    plt.plot(x_value, aveg_P_hr_ddpg_list[:x_up], label='Proposed', alpha=1.0,linewidth=1.0, color=color)
    r1 = list(map(lambda x: x[0] - x[1], zip(aveg_P_hr_ddpg_list[:x_up],P_hr_ddpg_SE_list[:x_up])))
    r2 = list(map(lambda x: x[0] + x[1], zip(aveg_P_hr_ddpg_list[:x_up],P_hr_ddpg_SE_list[:x_up])))
    plt.fill_between(x_value, r1, r2, color=color, alpha=0.3)
    # r1_1 = list(map(lambda x: x[0] - x[1], zip(aveg_P_hr_ddpg_list[:x_up], P_hr_ddpg_SD_list[:x_up])))
    # r2_1 = list(map(lambda x: x[0] + x[1], zip(aveg_P_hr_ddpg_list[:x_up], P_hr_ddpg_SD_list[:x_up])))
    # plt.fill_between(x_value, r1_1, r2_1, color=color, alpha=0.1)
    plt.plot(x_value, aveg_P_hr_baseline_list[:x_up], label='Baseline', alpha=1.0, linewidth=1.0, color=color_1)
    r1 = list(map(lambda x: x[0] - x[1], zip(aveg_P_hr_baseline_list[:x_up],P_hr_baseline_SE_list[:x_up])))
    r2 = list(map(lambda x: x[0] + x[1], zip(aveg_P_hr_baseline_list[:x_up],P_hr_baseline_SE_list[:x_up])))
    plt.fill_between(x_value, r1, r2, color=color_1, alpha=0.3)
    # r1_1 = list(map(lambda x: x[0] - x[1], zip(aveg_P_hr_baseline_list[:x_up], P_hr_baseline_SD_list[:x_up])))
    # r2_1 = list(map(lambda x: x[0] + x[1], zip(aveg_P_hr_baseline_list[:x_up], P_hr_baseline_SD_list[:x_up])))
    # plt.fill_between(x_value, r1_1, r2_1, color=color_1, alpha=0.1)
    plt.xlabel('Train Step')
    plt.ylabel('Percentage')
    plt.ylim((0.0,1.0))
    plt.legend()

    # plt.plot(x_value, aveg_P_better_list, label='aveg_P_better', alpha=1.0, linewidth=1.0, color=color_2)
    # r1 = list(map(lambda x: x[0] - x[1], zip(aveg_P_better_list, P_better_SE_list)))
    # r2 = list(map(lambda x: x[0] + x[1], zip(aveg_P_better_list, P_better_SE_list)))
    # plt.fill_between(x_value, r1, r2, color=color_2, alpha=0.2)
    plt.subplot(3, 1, 2)
    plt.plot(x_value, compare_count_aveg_list, label='aveg_compare_count', alpha=1.0, linewidth=1.0, color=color_2)
    r1 = list(map(lambda x: x[0] - x[1], zip(compare_count_aveg_list, compare_count_SE_list)))
    r2 = list(map(lambda x: x[0] + x[1], zip(compare_count_aveg_list, compare_count_SE_list)))
    plt.fill_between(x_value, r1, r2, color=color_2, alpha=0.3)
    # r1_1 = list(map(lambda x: x[0] - x[1], zip(compare_count_aveg_list, compare_count_SD_list)))
    # r2_1 = list(map(lambda x: x[0] + x[1], zip(compare_count_aveg_list, compare_count_SD_list)))
    # plt.fill_between(x_value, r1_1, r2_1, color=color_2, alpha=0.1)
    plt.xlabel('Train Step')
    plt.ylabel('Value')
    #plt.ylim((0.0, 1.0))
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(x_value, compare_ratio_aveg_list, label='aveg_compare_count_perc', alpha=1.0, linewidth=1.0, color=color_2)
    r1 = list(map(lambda x: x[0] - x[1], zip(compare_ratio_aveg_list, compare_ratio_SE_list)))
    r2 = list(map(lambda x: x[0] + x[1], zip(compare_ratio_aveg_list, compare_ratio_SE_list)))
    plt.fill_between(x_value, r1, r2, color=color_2, alpha=0.3)
    # r1_1 = list(map(lambda x: x[0] - x[1], zip(compare_ratio_aveg_list, compare_ratio_SD_list)))
    # r2_1 = list(map(lambda x: x[0] + x[1], zip(compare_ratio_aveg_list, compare_ratio_SD_list)))
    # plt.fill_between(x_value, r1_1, r2_1, color=color_2, alpha=0.1)
    plt.xlabel('Train Step')
    plt.ylabel('Percentage')
    plt.ylim((0.0, 1.1))
    plt.legend()
    #plt.title('Metric_value_aveg')

    file_path = '{folder}/Learning_curve_{case_name}_{compare_type}_{exp_type}_{test_result_type}.png'.format(folder=save_folder, case_name=case_name,
                                                                                           compare_type=compare_type, exp_type=exp_type,test_result_type=test_result_type)
    plt.savefig(file_path)
    plt.close()

    chosen_test_result = draw_bar_chart_compare(all_metric_aveg_df=all_metric_aveg_df, metric_all_trace_df=metric_summary_df_1,
                           metric_each_trace_df=metric_summary_df, save_folder=save_folder, case_name=case_name,
                           exp_type=exp_type, exp_id=exp_id, xaxis_interval=xaxis_interval,start_index=start_index,
                                                interval=interval,compare_type=compare_type, test_result_type=test_result_type)

    return chosen_test_result

def draw_bar_chart_compare(all_metric_aveg_df, metric_all_trace_df, metric_each_trace_df, save_folder,case_name,
                           exp_type, exp_id, xaxis_interval,start_index,interval,compare_type,test_result_type):

    # choose the snapshot test result of the best DDPG test
    max_P_hr_ddpg = all_metric_aveg_df['P_hr_ddpg'].max()
    #max_compare_count = all_metric_aveg_df[all_metric_aveg_df['P_hr_ddpg']==max_P_hr_ddpg]['compare_count'].max()
    #chosen_test_index = all_metric_aveg_df[(all_metric_aveg_df['P_hr_ddpg']==max_P_hr_ddpg)&(all_metric_aveg_df['compare_count']==max_compare_count)]['test_index'].min()
    chosen_test_index_all = all_metric_aveg_df[(all_metric_aveg_df['P_hr_ddpg'] == max_P_hr_ddpg)]['test_index'].tolist()
    #print(chosen_test_index_all)
    all_chosen_test_result_df = metric_all_trace_df[metric_all_trace_df['test_index'].isin(chosen_test_index_all)]
    #print(all_chosen_test_result_df)

    best_trial_P_hr_ddpg = all_chosen_test_result_df['P_hr_ddpg'].max()
    best_trial_compare_count = all_chosen_test_result_df[(metric_all_trace_df['P_hr_ddpg'] == best_trial_P_hr_ddpg)]['compare_count'].max()
    chosen_test_index = all_chosen_test_result_df[(metric_all_trace_df['P_hr_ddpg'] == best_trial_P_hr_ddpg)&
                                                  (metric_all_trace_df['compare_count'] == best_trial_compare_count)]['test_index'].tolist()[0]
    #print('best_trial_compare_count: ', best_trial_compare_count, ' chosen_test_index: ', chosen_test_index)
    chosen_compare_count_perc = metric_all_trace_df[(metric_all_trace_df['P_hr_ddpg'] == best_trial_P_hr_ddpg)&
                                           (metric_all_trace_df['test_index'] == chosen_test_index) &
                                            (metric_all_trace_df['compare_count'] == best_trial_compare_count)]['compare_count_perc'].max()
    best_trial_index = metric_all_trace_df[(metric_all_trace_df['P_hr_ddpg'] == best_trial_P_hr_ddpg)&
                                           (metric_all_trace_df['test_index'] == chosen_test_index) &
                                            (metric_all_trace_df['compare_count'] == best_trial_compare_count)&
                                           ((metric_all_trace_df['compare_count_perc'] == chosen_compare_count_perc))]['trial_index'].tolist()[0]
    #print(metric_all_trace_df[(metric_all_trace_df['P_hr_ddpg'] == best_trial_P_hr_ddpg)&
    #                          (metric_all_trace_df['test_index'] == chosen_test_index)&
    #                           (metric_all_trace_df['compare_count'] == best_trial_compare_count)])
    #print('best_trial_index: ', best_trial_index)
    # chosen_test_index = metric_all_trace_df[(metric_all_trace_df['P_hr_ddpg'] == best_trial_P_hr_ddpg) &
    #                                        (metric_all_trace_df['compare_count'] == best_trial_compare_count)&
    #                                         (metric_all_trace_df['trial_index'] == best_trial_index)]['test_index'].tolist()[0]
    # print(metric_all_trace_df[(metric_all_trace_df['P_hr_ddpg'] == best_trial_P_hr_ddpg) &
    #                                        (metric_all_trace_df['compare_count'] == best_trial_compare_count)&
    #                                         (metric_all_trace_df['trial_index'] == best_trial_index)])
    # print(chosen_test_index)
    test_x_value = int(((chosen_test_index-start_index)/interval + 1) * (xaxis_interval))
    print('best_trial_index: ', best_trial_index, ' chosen_test_index: ', chosen_test_index,
          ' max_P_hr_ddpg: ',max_P_hr_ddpg, ' best_trial_P_hr_ddpg: ',best_trial_P_hr_ddpg, ' test_x_value: ', test_x_value)
    value_col_name = ['ddpg_total_reward_improve_perc_Best_CF_this_trace', 'ddpg_cf_distance_Best_CF_this_trace',
                      'baseline_total_reward_improve_perc_Best_CF_this_trace', 'baseline_cf_distance_Best_CF_this_trace']
    label_list =  ['ddpg_imp_Best_CF', 'ddpg_cf_distance_Best_CF',
                      'baseline_imp_Best_CF', 'baseline_cf_distance_Best_CF']
    color_list = ['green', 'orange','green', 'orange']
    #style_list = ['-', '-', 'X', 'X']
    value_col_name_1 = ['imp_ratio', 'distance_ratio']
    label_list_1 = ['imp_ratio', 'distance_ratio']
    color_list_1 = ['green', 'orange']

    chosen_test_result = metric_each_trace_df[(metric_each_trace_df['test_index']==chosen_test_index)&
                                              (metric_each_trace_df['trial_index']==best_trial_index)]
    chosen_test_result['Exp_type'] = [exp_type]*len(chosen_test_result)
    chosen_test_result['Exp_ID'] = [exp_id] * len(chosen_test_result)
    # color_list_1 = ['green', 'orange'] * len(chosen_test_result)
    # for k in range(0, len(chosen_test_result['compare_ratio'].tolist())):
    #     if chosen_test_result['compare_ratio'].tolist()[k] == -99999:
    #         color_list_1[k] = 'gray'
    #chosen_test_result['imp_ratio'] = chosen_test_result['ddpg_total_reward_improve_perc_Best_CF_this_trace']/chosen_test_result['baseline_total_reward_improve_perc_Best_CF_this_trace']
    #chosen_test_result['distance_ratio'] = chosen_test_result['ddpg_cf_distance_Best_CF_this_trace'] / chosen_test_result['baseline_cf_distance_Best_CF_this_trace']
    #print('chosen_test_result before: ', chosen_test_result)
    chosen_test_result.loc[chosen_test_result['compare_ratio'] == -99999, 'ddpg_total_reward_improve_perc_Best_CF_this_trace'] = -1
    chosen_test_result.loc[chosen_test_result['compare_ratio'] == -99999,'ddpg_cf_distance_Best_CF_this_trace'] = -1
    chosen_test_result.loc[chosen_test_result['compare_ratio'] == -99999, 'baseline_total_reward_improve_perc_Best_CF_this_trace'] = -1
    chosen_test_result.loc[chosen_test_result['compare_ratio'] == -99999, 'baseline_cf_distance_Best_CF_this_trace'] = -1
    chosen_test_result.loc[chosen_test_result['compare_ratio'] == -99999, 'imp_ratio'] = -1
    chosen_test_result.loc[chosen_test_result['compare_ratio'] == -99999, 'distance_ratio'] = -1
    chosen_test_result = chosen_test_result.reset_index(drop=True)
    #print('chosen_test_result after: ', chosen_test_result)
    x_list = list(range(1, len(chosen_test_result) + 1))
    chosen_test_result['X'] = x_list
    ax = chosen_test_result.plot(x='X', y=value_col_name, label=label_list, kind="bar",
                                 color=color_list, rot=0, figsize=(20, 10), use_index=True, legend=True, #fontsize=20,
                                 xlabel='Original_trace_id', ylabel='Value')
    bars = ax.patches
    patterns = ['-', '-', '/', '/'] #('-', '+', 'x', '/', '//', 'O', 'o', '\\', '\\\\')
    hatches = [p for p in patterns for i in range(len(chosen_test_result))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend()
    line_length = len(chosen_test_result)
    plt.plot(list(range(0, line_length)), [0] * line_length, color='red', alpha=1.0, linewidth=0.5)

    # bars = ax.patches
    # hatches = ''.join(h * len(chosen_test_result) for h in 'x/O.')
    # for bar, hatch in zip(bars, hatches):
    #     bar.set_hatch(hatch)
    # ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15))

    file_path = '{folder}/Best_CF_Compare_Bar_trial_{best_trial_index}_test_{chosen_test_index}_{test_x_value}_{case_name}_{exp_type}_{compare_type}_ABS_{test_result_type}.png'.format(folder=save_folder,
                                                                            case_name=case_name,exp_type=exp_type,compare_type=compare_type,
                                                                            best_trial_index=best_trial_index,chosen_test_index=chosen_test_index,
                                                                            test_x_value=test_x_value,test_result_type=test_result_type)
    plt.savefig(file_path)
    plt.close()

    ax = chosen_test_result.plot(x='X', y=value_col_name_1, label=label_list_1, kind="bar",
                                 color=color_list_1, rot=0, figsize=(20, 10), use_index=True, legend=True, #fontsize=20,
                                 xlabel='Trace ID', ylabel='Value')
    # bars = ax.patches
    # patterns = ['-', '-', '/', '/']  # ('-', '+', 'x', '/', '//', 'O', 'o', '\\', '\\\\')
    # hatches = [p for p in patterns for i in range(len(chosen_test_result))]
    # for bar, hatch in zip(bars, hatches):
    #     bar.set_hatch(hatch)
    ax.legend()
    plt.plot(list(range(0, line_length)), [0]*line_length, color='red', alpha=1.0, linewidth=0.5)

    # bars = ax.patches
    # hatches = ''.join(h * len(chosen_test_result) for h in 'x/O.')
    # for bar, hatch in zip(bars, hatches):
    #     bar.set_hatch(hatch)
    # ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15))

    file_path = '{folder}/Best_CF_Compare_Bar_trial_{best_trial_index}_test_{chosen_test_index}_{test_x_value}_{case_name}_{exp_type}_{compare_type}_Relative_{test_result_type}.png'.format(
        folder=save_folder,
        case_name=case_name, exp_type=exp_type,compare_type=compare_type,
        best_trial_index=best_trial_index, chosen_test_index=chosen_test_index,test_x_value=test_x_value,test_result_type=test_result_type)
    plt.savefig(file_path)
    plt.close()
    # for item, row in chosen_test_result.iterrows():
    #
    #     ddpg_imp = chosen_test_result['ddpg_total_reward_improve_perc_Best_CF_this_trace']
    #     ddpg_dist =  chosen_test_result['ddpg_cf_distance_Best_CF_this_trace']
    #     baseline_imp = chosen_test_result['baseline_total_reward_improve_perc_Best_CF_this_trace']
    #     baseline_dist = chosen_test_result['baseline_cf_distance_Best_CF_this_trace']

    return chosen_test_result

def get_metric_with_best_baseline(all_metric_aveg_df, metric_summary_df_all):
    # get the best baseline snapshot result
    # best_aveg_P_hr_ddpg = all_metric_aveg_df['P_hr_ddpg'].max()
    max_test_index_best_aveg_P_hr_ddpg = all_metric_aveg_df['test_index'].max()#all_metric_aveg_df[all_metric_aveg_df['P_hr_ddpg'] == best_aveg_P_hr_ddpg]['test_index'].max()
    # print('best_aveg_P_hr_ddpg: ', best_aveg_P_hr_ddpg, ' max_test_index_best_aveg_P_hr_ddpg: ', max_test_index_best_aveg_P_hr_ddpg)
    print('max_test_index_best_aveg_P_hr_ddpg: ', max_test_index_best_aveg_P_hr_ddpg)
    best_aveg_P_hr_baseline = all_metric_aveg_df[all_metric_aveg_df['test_index'] <= max_test_index_best_aveg_P_hr_ddpg]['P_hr_baseline'].max()
    test_index_best_aveg_P_hr_baseline_list = all_metric_aveg_df[(all_metric_aveg_df['P_hr_baseline'] == best_aveg_P_hr_baseline)&
                                                            (all_metric_aveg_df['test_index'] <= max_test_index_best_aveg_P_hr_ddpg)]['test_index'].tolist()
    best_P_hr_baseline = metric_summary_df_all[metric_summary_df_all['test_index'].isin(test_index_best_aveg_P_hr_baseline_list)]['P_hr_baseline'].max()
    trial_id_best_P_hr_baseline_df = metric_summary_df_all[(metric_summary_df_all['test_index'].isin(test_index_best_aveg_P_hr_baseline_list)) &
                                                        (metric_summary_df_all['P_hr_baseline'] == best_P_hr_baseline)]
    test_index_best_aveg_P_hr_baseline = trial_id_best_P_hr_baseline_df['test_index'].min()
    trial_id_best_P_hr_baseline = trial_id_best_P_hr_baseline_df[(trial_id_best_P_hr_baseline_df['test_index']==test_index_best_aveg_P_hr_baseline)]['trial_index'].min()
    print('best_aveg_P_hr_baseline: ', best_aveg_P_hr_baseline, ' best_P_hr_baseline: ', best_P_hr_baseline)
    print('test_index_best_aveg_P_hr_baseline_list: ', test_index_best_aveg_P_hr_baseline_list)
    # best_baseline_df = metric_summary_df_each[(metric_summary_df_each['test_index']==test_index_best_aveg_P_hr_baseline) &
    #                                           (metric_summary_df_each['trial_index']==trial_id_best_P_hr_baseline)]
    return test_index_best_aveg_P_hr_baseline, trial_id_best_P_hr_baseline

def draw_multiple_test_result_RP2(data_folder, s_thre_list, all_train_round_list, exp_type, exp_id, xaxis_interval,compare_type,test_result_type):
    #color_list = ['red', 'orange','blue', 'purple','green']
    if exp_type == 'Across_RP':
        #color_list = ['orangered','orange', 'cadetblue','blue', 'deeppink','purple', 'green']
        #color_list = ['orangered', 'deeppink', 'orange', 'purple', 'green']
        #color_list = ['deeppink', 'purple', 'green'] #0.13
        #color_list = ['orangered', 'orange', 'green']#0.18
        #color_list = ['cadetblue', 'blue', 'green']#0.08
        color_list = ['orangered', 'purple', 'green']#final figure
    else:
        color_list = ['orange', 'blue', 'purple', 'green']
    aveg_test_result_df_dict = {}
    for s_thre in s_thre_list:
        df = pd.read_csv('{data_folder}/all_metric_aveg_{compare_type}_{test_result_type}_{s_thre}.csv'.format(s_thre=s_thre, compare_type=compare_type,
                                                                                                               test_result_type=test_result_type,
                                                                                                      data_folder=data_folder), index_col=0)
        aveg_test_result_df_dict[s_thre] = df

    fontsize = 25
    fontweight = 'bold'  # fontsize=fontsize, fontweight=fontweight
    alpha = 0.15
    ncol = 1
    linewidth = 2.0
    fig = plt.figure(figsize=(20, 40))
    plt.ticklabel_format(style='plain')
    plt.rcParams['font.size'] = fontsize
    marker_list = ['+', 'x', '^']
    for k in range(len(s_thre_list)):
        if exp_type=='Across_RP' and (k in [0, 2]):#[0, 1, 4]
            line_style = '-'#'solid'
            #marker =
        elif exp_type=='Across_RP' and (k in [1]):#[2, 3]
            line_style = '-'#'dashdot': -.
        else:
            line_style = '-'

        s_thre = s_thre_list[k]
        aveg_P_hr_ddpg_list = aveg_test_result_df_dict[s_thre]['P_hr_ddpg'].tolist()
        P_hr_ddpg_SE_list = aveg_test_result_df_dict[s_thre]['P_hr_ddpg_SE'].tolist()
        compare_count_aveg_list = aveg_test_result_df_dict[s_thre]['compare_count'].tolist()
        compare_count_SE_list = aveg_test_result_df_dict[s_thre]['compare_count_SE'].tolist()
        compare_ratio_aveg_list = aveg_test_result_df_dict[s_thre]['compare_count_perc'].tolist()
        compare_ratio_SE_list = aveg_test_result_df_dict[s_thre]['compare_count_perc_SE'].tolist()
        color = color_list[k]
        aveg_P_hr_baseline_list = aveg_test_result_df_dict[s_thre]['P_hr_baseline'].tolist()
        P_hr_baseline_SE_list = aveg_test_result_df_dict[s_thre]['P_hr_baseline_SE'].tolist()
        color_baseline = 'black'
        x_up = len(all_train_round_list)  # 30
        x_value_list = list(range(1, x_up + 1))
        x_value = [x * (xaxis_interval) for x in x_value_list]

        plt.subplot(3, 1, 1)
        # fig = plt.figure(figsize=(15, 12))
        # plt.ticklabel_format(style='plain')
        # plt.rcParams['font.size'] = fontsize
        plt.plot(x_value, aveg_P_hr_ddpg_list[:x_up], label='Proposed_{s_thre}'.format(s_thre=s_thre), linestyle=line_style,
                                        alpha=1.0, linewidth=linewidth, color=color)
        r1 = list(map(lambda x: x[0] - x[1], zip(aveg_P_hr_ddpg_list[:x_up], P_hr_ddpg_SE_list[:x_up])))
        r2 = list(map(lambda x: x[0] + x[1], zip(aveg_P_hr_ddpg_list[:x_up], P_hr_ddpg_SE_list[:x_up])))
        plt.fill_between(x_value, r1, r2, color=color, alpha=alpha)
        if k==0:
            plt.plot(x_value, aveg_P_hr_baseline_list[:x_up], label='Baseline', alpha=1.0, linewidth=linewidth, color=color_baseline)
            r1 = list(map(lambda x: x[0] - x[1], zip(aveg_P_hr_baseline_list[:x_up], P_hr_baseline_SE_list[:x_up])))
            r2 = list(map(lambda x: x[0] + x[1], zip(aveg_P_hr_baseline_list[:x_up], P_hr_baseline_SE_list[:x_up])))
            plt.fill_between(x_value, r1, r2, color=color_baseline, alpha=alpha)
        plt.xlabel('Train Step',fontsize=fontsize, fontweight=fontweight)
        plt.ylabel('Percentage',fontsize=fontsize, fontweight=fontweight)
        plt.ylim((0.0, 1.0))
        plt.xticks(size=fontsize, weight=fontweight)
        plt.yticks(size=fontsize, weight=fontweight)
        plt.legend(prop={'size':fontsize,'weight':fontweight}, ncol=ncol)
        # file_path = '{folder}/Learning_curve_{case_name}_{compare_type}_{exp_type}_{test_result_type}_exp_{exp_id}.png'.format(
        #     folder=data_folder, case_name=case_name, compare_type=compare_type, exp_type=exp_type,
        #     test_result_type=test_result_type, exp_id=exp_id)
        # plt.savefig(file_path)
        # plt.close()

        plt.subplot(3, 1, 2) #aveg_compare_count
        # fig = plt.figure(figsize=(15, 12))
        # plt.ticklabel_format(style='plain')
        # plt.rcParams['font.size'] = fontsize
        plt.plot(x_value, compare_count_aveg_list[:x_up], label='Proposed_{s_thre}'.format(s_thre=s_thre), linestyle=line_style,
                 alpha=1.0, linewidth=linewidth, color=color)
        r1 = list(map(lambda x: x[0] - x[1], zip(compare_count_aveg_list[:x_up], compare_count_SE_list[:x_up])))
        r2 = list(map(lambda x: x[0] + x[1], zip(compare_count_aveg_list[:x_up], compare_count_SE_list[:x_up])))
        plt.fill_between(x_value, r1, r2, color=color, alpha=alpha)
        plt.xlabel('Train Step',fontsize=fontsize, fontweight=fontweight)
        plt.ylabel('Value',fontsize=fontsize, fontweight=fontweight)
        plt.xticks(size=fontsize, weight=fontweight)
        plt.yticks(size=fontsize, weight=fontweight)
        plt.legend(prop={'size':fontsize,'weight':fontweight}, ncol=ncol)
        # file_path = '{folder}/aveg_compare_count_{case_name}_{compare_type}_{exp_type}_{test_result_type}_exp_{exp_id}.png'.format(
        #     folder=data_folder, case_name=case_name, compare_type=compare_type, exp_type=exp_type,
        #     test_result_type=test_result_type, exp_id=exp_id)
        # plt.savefig(file_path)
        # plt.close()

        plt.subplot(3, 1, 3)#aveg_compare_count_perc
        # fig = plt.figure(figsize=(15, 12))
        # plt.ticklabel_format(style='plain')
        # plt.rcParams['font.size'] = fontsize
        plt.plot(x_value, compare_ratio_aveg_list[:x_up], label='Proposed_{s_thre}'.format(s_thre=s_thre), linestyle=line_style,
                 alpha=1.0, linewidth=linewidth,color=color)
        r1 = list(map(lambda x: x[0] - x[1], zip(compare_ratio_aveg_list[:x_up], compare_ratio_SE_list[:x_up])))
        r2 = list(map(lambda x: x[0] + x[1], zip(compare_ratio_aveg_list[:x_up], compare_ratio_SE_list[:x_up])))
        plt.fill_between(x_value, r1, r2, color=color, alpha=alpha)
        plt.xlabel('Train Step',fontsize=fontsize, fontweight=fontweight)
        plt.ylabel('Percentage',fontsize=fontsize, fontweight=fontweight)
        plt.ylim((0.0, 1.1))
        plt.xticks(size=fontsize, weight=fontweight)
        plt.yticks(size=fontsize, weight=fontweight)
        plt.legend(prop={'size':fontsize,'weight':fontweight}, ncol=ncol)
        # file_path = '{folder}/aveg_compare_count_perc_{case_name}_{compare_type}_{exp_type}_{test_result_type}_exp_{exp_id}.png'.format(
        #     folder=data_folder, case_name=case_name, compare_type=compare_type, exp_type=exp_type,
        #     test_result_type=test_result_type, exp_id=exp_id)
        # plt.savefig(file_path)
        # plt.close()

    file_path = '{folder}/Learning_curve_{case_name}_{compare_type}_{exp_type}_{test_result_type}_exp_{exp_id}.png'.format(
        folder=data_folder, case_name=case_name,compare_type=compare_type, exp_type=exp_type, test_result_type=test_result_type,exp_id=exp_id)
    plt.savefig(file_path, dpi=300)
    plt.close()
    return

def draw_all_trace_TD3_action_num_each_exp_RP2(data_folder,compare_type,test_result_type, exp_id, trace_index_df,
                                               all_train_round_list,xaxis_interval,total_trial_num, case_name):
    data_path = '{data_folder}/metric_summary_each_trace_{compare_type}_exp_{exp_id}_{test_result_type}.csv'.format(data_folder=data_folder,
                                                                                                                  compare_type=compare_type,
                                                                                              test_result_type=test_result_type,exp_id=exp_id)
    data_df = pd.read_csv(data_path, index_col=0)
    all_df_list = []
    fontsize = 30
    alpha = 1.0
    for t_index in range(1, total_trial_num + 1):
        fig = plt.figure(figsize=(20, 15))
        plt.ticklabel_format(style='plain')
        plt.rcParams['font.size'] = fontsize
        for item, row in trace_index_df.iterrows():
            if case_name=='lunar_lander':
                ENV_ID, orig_trace_episode, orig_end_step = row['gravity'], row['orig_episode'], row['orig_end_time_index']
            else:
                ENV_ID, orig_trace_episode, orig_end_step = row['ENV_NAME'], row['orig_episode'], row['orig_end_time_index']

            x_up = len(all_train_round_list)  # 30
            x_value_list = list(range(1, x_up + 1))
            x_value = [x * (xaxis_interval) for x in x_value_list]
            average_TD3_action_num_this_seg = data_df[(data_df['ENV_ID']==ENV_ID)
                                                              & (data_df['orig_trace_episode']==orig_trace_episode)
                                                              &(data_df['orig_end_step']==orig_end_step)
                                                                &(data_df['trial_index']==t_index)]['aveg_TD3_action_count'].tolist()[:x_up]
            label = '{ENV_ID}_{orig_trace_episode}_{orig_end_step}'.format(ENV_ID=ENV_ID,orig_trace_episode=orig_trace_episode, orig_end_step=orig_end_step)
            plt.plot(x_value, average_TD3_action_num_this_seg, label=label, alpha=alpha, linewidth=1.0)
            plt.xlabel('Train Step')
            plt.ylabel('TD3_Action_Num')
            #plt.legend()
            new_df = pd.DataFrame()
            new_df['ENV_ID'] = [ENV_ID]*len(x_value)
            new_df['orig_trace_episode'] = [orig_trace_episode] * len(x_value)
            new_df['orig_end_step'] = [orig_end_step] * len(x_value)
            new_df['trial_index'] = [t_index] * len(x_value)
        file_path = '{folder}/TD3_Action_Num_all_trace_{case_name}_{compare_type}_{exp_type}_{test_result_type}_exp_{exp_id}_t_{t_index}.png'.format(
                folder=data_folder, case_name=case_name, compare_type=compare_type, exp_type=exp_type,
                test_result_type=test_result_type, exp_id=exp_id, t_index=t_index)
        plt.savefig(file_path)
        plt.close()
    for item, row in trace_index_df.iterrows():
        fig = plt.figure(figsize=(20, 15))
        plt.ticklabel_format(style='plain')
        plt.rcParams['font.size'] = fontsize
        new_df = pd.DataFrame()
        if case_name == 'lunar_lander':
            ENV_ID, orig_trace_episode, orig_end_step = row['gravity'], row['orig_episode'], row['orig_end_time_index']
        else:
            ENV_ID, orig_trace_episode, orig_end_step = row['ENV_NAME'], row['orig_episode'], row['orig_end_time_index']
        #ENV_ID, orig_trace_episode, orig_end_step = row['gravity'], row['orig_episode'], row['orig_end_time_index']
        label = '{ENV_ID}_{orig_trace_episode}_{orig_end_step}'.format(ENV_ID=ENV_ID,
                                                                       orig_trace_episode=orig_trace_episode,
                                                                       orig_end_step=orig_end_step)
        for t_index in range(1, total_trial_num + 1):
            x_up = len(all_train_round_list)  # 30
            x_value_list = list(range(1, x_up + 1))
            x_value = [x * (xaxis_interval) for x in x_value_list]
            average_TD3_action_num_this_seg = data_df[(data_df['ENV_ID'] == ENV_ID)
                                                      & (data_df['orig_trace_episode'] == orig_trace_episode)
                                                      & (data_df['orig_end_step'] == orig_end_step)
                                                      & (data_df['trial_index'] == t_index)]['aveg_TD3_action_count'].tolist()[:x_up]
            plt.plot(x_value, average_TD3_action_num_this_seg, label='trial_{t_index}'.format(t_index=t_index), alpha=alpha, linewidth=1.0)
            plt.xlabel('Train Step')
            plt.ylabel('TD3_Action_Num')
            plt.legend()
            new_df['ENV_ID'] = [ENV_ID] * len(x_value)
            new_df['orig_trace_episode'] = [orig_trace_episode] * len(x_value)
            new_df['orig_end_step'] = [orig_end_step] * len(x_value)
            new_df['average_TD3_action_num_trial_{t_index}'.format(t_index=t_index)] = average_TD3_action_num_this_seg
            new_df = new_df.loc[:, ~new_df.columns.duplicated()]
        file_path = '{folder}/TD3_Action_Num_trace_{label}_{exp_type}_{test_result_type}_exp_{exp_id}.png'.format(
            folder=data_folder, case_name=case_name, exp_type=exp_type,label=label,
            test_result_type=test_result_type, exp_id=exp_id)
        plt.savefig(file_path)
        plt.close()
        all_df_list.append(new_df)
    TD3_action_num_data_each_trace_df = pd.concat(all_df_list)
    col_name_list = ['average_TD3_action_num_trial_{t_index}'.format(t_index=t) for t in range(1, total_trial_num+1)]
    TD3_action_num_data_each_trace_df['mean_of_all_trial'] = TD3_action_num_data_each_trace_df[col_name_list].mean(axis=1)
    TD3_action_num_data_each_trace_df['SD_of_all_trial'] = TD3_action_num_data_each_trace_df[col_name_list].std(axis=1)
    TD3_action_num_data_each_trace_df['SE_of_all_trial'] = TD3_action_num_data_each_trace_df[col_name_list].std(axis=1)/np.sqrt(total_trial_num)
    TD3_action_num_data_each_trace_df.to_csv('{data_folder}/TD3_action_num_data_each_trace.csv'.format(data_folder=data_folder))
    for item, row in trace_index_df.iterrows():
        alpha = 0.4
        color='green'
        fig = plt.figure(figsize=(20, 15))
        plt.ticklabel_format(style='plain')
        plt.rcParams['font.size'] = fontsize
        if case_name == 'lunar_lander':
            ENV_ID, orig_trace_episode, orig_end_step = row['gravity'], row['orig_episode'], row['orig_end_time_index']
        else:
            ENV_ID, orig_trace_episode, orig_end_step = row['ENV_NAME'], row['orig_episode'], row['orig_end_time_index']
        #ENV_ID, orig_trace_episode, orig_end_step = row['gravity'], row['orig_episode'], row['orig_end_time_index']
        label = '{ENV_ID}_{orig_trace_episode}_{orig_end_step}'.format(ENV_ID=ENV_ID,
                                                                       orig_trace_episode=orig_trace_episode,
                                                                       orig_end_step=orig_end_step)
        x_up = len(all_train_round_list)  # 30
        x_value_list = list(range(1, x_up + 1))
        x_value = [x * (xaxis_interval) for x in x_value_list]
        mean_TD3_action_num_this_seg = TD3_action_num_data_each_trace_df[(TD3_action_num_data_each_trace_df['ENV_ID'] == ENV_ID)
                                                  & (TD3_action_num_data_each_trace_df['orig_trace_episode'] == orig_trace_episode)
                                                  & (TD3_action_num_data_each_trace_df['orig_end_step'] == orig_end_step)]['mean_of_all_trial'].tolist()[:x_up]
        SD_TD3_action_num_this_seg = TD3_action_num_data_each_trace_df[(TD3_action_num_data_each_trace_df['ENV_ID'] == ENV_ID)
                                          & (TD3_action_num_data_each_trace_df[
                                                 'orig_trace_episode'] == orig_trace_episode)
                                          & (TD3_action_num_data_each_trace_df['orig_end_step'] == orig_end_step)]['SD_of_all_trial'].tolist()[:x_up]
        SE_TD3_action_num_this_seg = TD3_action_num_data_each_trace_df[(TD3_action_num_data_each_trace_df['ENV_ID'] == ENV_ID)
                                          & (TD3_action_num_data_each_trace_df['orig_trace_episode'] == orig_trace_episode)
                                          & (TD3_action_num_data_each_trace_df['orig_end_step'] == orig_end_step)]['SE_of_all_trial'].tolist()[:x_up]

        plt.plot(x_value, mean_TD3_action_num_this_seg, label='mean_of_all_trial', color=color, alpha=1.0,linewidth=1.0)
        r1 = list(map(lambda x: x[0] - x[1], zip(mean_TD3_action_num_this_seg, SE_TD3_action_num_this_seg)))
        r2 = list(map(lambda x: x[0] + x[1], zip(mean_TD3_action_num_this_seg, SE_TD3_action_num_this_seg)))
        for i in range(0,len(r1)):
            if r1[i]<0:
                r1[i] = 0
            if r2[i]>20.0:
                r2[i] = 20.0
        plt.fill_between(x_value, r1, r2, alpha=alpha, color=color)
        r1 = list(map(lambda x: x[0] - x[1], zip(mean_TD3_action_num_this_seg, SD_TD3_action_num_this_seg)))
        r2 = list(map(lambda x: x[0] + x[1], zip(mean_TD3_action_num_this_seg, SD_TD3_action_num_this_seg)))
        for i in range(0,len(r1)):
            if r1[i]<0:
                r1[i] = 0
            if r2[i]>20.0:
                r2[i] = 20.0
        plt.fill_between(x_value, r1, r2, alpha=alpha*0.8, color=color)
        plt.xlabel('Train Step')
        plt.ylabel('TD3_Action_Num')
        plt.legend()
        file_path = '{folder}/TD3_Action_Num_Aveg_trace_{label}_{exp_type}_{test_result_type}_exp_{exp_id}.png'.format(
            folder=data_folder, case_name=case_name, exp_type=exp_type, label=label,
            test_result_type=test_result_type, exp_id=exp_id)
        plt.savefig(file_path)
        plt.close()


    return

def del_unused_baseline_data(data_folder, all_train_round_list,total_trial_num, test_result_type):
    for t in range(1,total_trial_num+1):
        folder = '{data_folder}/td3_cf_results/trial_{t}/{test_result_type}'.format(data_folder=data_folder, t=t, test_result_type=test_result_type)
        for train_round in all_train_round_list:
            file_1 = '{folder}/cf_traces_test_baseline_r_{train_round}.pkl'.format(folder=folder, train_round=train_round)
            file_2 = '{folder}/test_result_baseline_r_{train_round}.pkl'.format(folder=folder,
                                                                                   train_round=train_round)
            file_3 = '{folder}/test_statistic_baseline_r_{train_round}.pkl'.format(folder=folder,
                                                                                   train_round=train_round)
            #print(file_1)
            if os.path.exists(file_1):
                #print('exist')
                os.remove(file_1)
            if os.path.exists(file_2):
                #print('exist')
                os.remove(file_2)
            if os.path.exists(file_3):
                #print('exist')
                os.remove(file_3)

    return

def draw_multiple_Phr_final_figure(data_folder, s_thre_list, all_train_round_list_EXP1,all_train_round_list_EXP3,
                                   exp_type, xaxis_interval,compare_type,test_result_type, case_name):
    #color_list = ['red', 'orange','blue', 'purple','green']
    def format_ticks(x, pos):
        return f'{x:.1f}'

    if case_name=='lunar_lander':
        ylim=(0.0, 1.0)
    else:
        ylim=(0.35, 0.6)
    if exp_type == 'Across_RP':
        color_list = ['green', 'orangered', 'purple']#final figure
    else:
        color_list = ['orange', 'blue', 'purple', 'green']

    #for exp_id in [1,3]:
    aveg_test_result_df_dict_exp1 = {}
    aveg_test_result_df_dict_exp3 = {}
    RP_list = []
    x_scaler_dict={'EXP1':10000, 'EXP3':10000}
    label_dict = {'1':'P1', '2':'P2-base', '3':'P2-fixed'}
    #linestyle_dict = {'1':':', '2':'--', '3':'-.'}
    linestyle_dict = {'1': '-', '2': '-', '3': '-'}
    for s_thre in s_thre_list:
        RP_id = s_thre[-1]
        RP_list.append(RP_id)
        df_exp1 = pd.read_csv('{data_folder}/all_metric_aveg_{compare_type}_{test_result_type}_{s_thre}_EXP1.csv'.format(s_thre=s_thre, compare_type=compare_type,
                                                                                                               test_result_type=test_result_type,
                                                                                                      data_folder=data_folder), index_col=0)
        df_exp3 = pd.read_csv('{data_folder}/all_metric_aveg_{compare_type}_{test_result_type}_{s_thre}_EXP3.csv'.format(s_thre=s_thre,
                                                                                                  compare_type=compare_type,
                                                                                                  test_result_type=test_result_type,
                                                                                                  data_folder=data_folder), index_col=0)
        aveg_test_result_df_dict_exp1[RP_id] = df_exp1
        aveg_test_result_df_dict_exp3[RP_id] = df_exp3

    fontsize = 30 #25
    fontsize_legend = 25#22
    fontweight = 'bold'  # fontsize=fontsize, fontweight=fontweight
    alpha = 0.15
    ncol = 1
    linewidth = 2.0
    #fig = plt.figure(figsize=(20, 20))
    fig = plt.figure(figsize=(20, 30))
    plt.ticklabel_format(style='plain')
    plt.rcParams['font.size'] = fontsize
    marker_list = ['+', 'x', '^']
    for k in range(len(s_thre_list)):
        if exp_type=='Across_RP' and (k in [0, 2]):#[0, 1, 4]
            line_style = '-'#'solid'
            #marker =
        elif exp_type=='Across_RP' and (k in [1]):#[2, 3]
            line_style = '-'#'dashdot': -.
        else:
            line_style = '-'

        RP_id = RP_list[k]
        aveg_P_hr_ddpg_list_exp1 = aveg_test_result_df_dict_exp1[RP_id]['P_hr_ddpg'].tolist()[:len(all_train_round_list_EXP1)]
        P_hr_ddpg_SE_list_exp1 = aveg_test_result_df_dict_exp1[RP_id]['P_hr_ddpg_SE'].tolist()[:len(all_train_round_list_EXP1)]
        # compare_count_aveg_list_exp1 = aveg_test_result_df_dict_exp1[s_thre]['compare_count'].tolist()
        # compare_count_SE_list_exp1 = aveg_test_result_df_dict_exp1[s_thre]['compare_count_SE'].tolist()
        # compare_ratio_aveg_list_exp1 = aveg_test_result_df_dict_exp1[s_thre]['compare_count_perc'].tolist()
        # compare_ratio_SE_list_exp1 = aveg_test_result_df_dict_exp1[s_thre]['compare_count_perc_SE'].tolist()
        aveg_P_hr_baseline_list_exp1 = aveg_test_result_df_dict_exp1[RP_id]['P_hr_baseline'].tolist()[:len(all_train_round_list_EXP1)]
        P_hr_baseline_SE_list_exp1 = aveg_test_result_df_dict_exp1[RP_id]['P_hr_baseline_SE'].tolist()[:len(all_train_round_list_EXP1)]
        #print(RP_id, len(aveg_P_hr_ddpg_list_exp1), len(aveg_P_hr_baseline_list_exp1))

        aveg_P_hr_ddpg_list_exp3 = aveg_test_result_df_dict_exp3[RP_id]['P_hr_ddpg'].tolist()[:len(all_train_round_list_EXP3)]
        P_hr_ddpg_SE_list_exp3 = aveg_test_result_df_dict_exp3[RP_id]['P_hr_ddpg_SE'].tolist()[:len(all_train_round_list_EXP3)]
        aveg_P_hr_baseline_list_exp3 = aveg_test_result_df_dict_exp3[RP_id]['P_hr_baseline'].tolist()[:len(all_train_round_list_EXP3)]
        P_hr_baseline_SE_list_exp3 = aveg_test_result_df_dict_exp3[RP_id]['P_hr_baseline_SE'].tolist()[:len(all_train_round_list_EXP3)]
        #print(RP_id, len(aveg_P_hr_ddpg_list_exp3), len(aveg_P_hr_baseline_list_exp3))

        color_baseline = 'black'
        color = color_list[k]
        plt.subplot(2, 1, 1)  # Phr_Exp1
        x_up = len(all_train_round_list_EXP1)  # 30
        x_value_list = list(range(1, x_up + 1))
        # 计算 x_value 和 x_tick
        x_value = [x * (xaxis_interval) for x in x_value_list]
        x_tick = [round(x * (xaxis_interval) / x_scaler_dict['EXP1'], 1) for x in x_value_list]

        if k == 0:
            plt.plot(x_value, aveg_P_hr_baseline_list_exp1[:x_up], label='Baseline', alpha=1.0, linewidth=linewidth,
                     color=color_baseline, linestyle='-')
            r1 = list(
                map(lambda x: x[0] - x[1], zip(aveg_P_hr_baseline_list_exp1[:x_up], P_hr_baseline_SE_list_exp1[:x_up])))
            r2 = list(
                map(lambda x: x[0] + x[1], zip(aveg_P_hr_baseline_list_exp1[:x_up], P_hr_baseline_SE_list_exp1[:x_up])))
            plt.fill_between(x_value, r1, r2, color=color_baseline, alpha=alpha)

        plt.plot(x_value, aveg_P_hr_ddpg_list_exp1[:x_up], label=label_dict[RP_id],#'Proposed_{RP_id}'.format(RP_id=RP_id),
                 linestyle=linestyle_dict[RP_id],
                 alpha=1.0, linewidth=linewidth, color=color)
        r1 = list(map(lambda x: x[0] - x[1], zip(aveg_P_hr_ddpg_list_exp1[:x_up], P_hr_ddpg_SE_list_exp1[:x_up])))
        r2 = list(map(lambda x: x[0] + x[1], zip(aveg_P_hr_ddpg_list_exp1[:x_up], P_hr_ddpg_SE_list_exp1[:x_up])))
        plt.fill_between(x_value, r1, r2, color=color, alpha=alpha)

        # 添加浅色网格线
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)  # 设置浅色网格线
        plt.xlabel('Time steps', fontsize=fontsize, fontweight=fontweight)
        plt.ylabel('Percentage', fontsize=fontsize+10, fontweight=fontweight)
        plt.ylim(ylim)
        # 使用FuncFormatter将x_tick转换为科学计数法
        #plt.gca().xaxis.set_major_formatter(FuncFormatter(scientific))
        #plt.xticks(x_tick, size=fontsize, weight=fontweight)  # 使用缩小后的 x_tick
        #plt.xticks(x_tick, size=fontsize, weight=fontweight)  # 使用缩小后的 x_tick
        ax = plt.gca()
        ax.ticklabel_format(style='sci', scilimits=(-1, 4), axis='x')
        #ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))  # 格式化X轴刻度
        ax.xaxis.get_offset_text().set_fontsize(fontsize)
        plt.xticks(size=fontsize, weight=fontweight)  # 使用缩小后的 x_tick
        plt.yticks(size=fontsize, weight=fontweight)
        plt.legend(prop={'size': fontsize_legend, 'weight': fontweight}, ncol=ncol,loc="lower right")
        plt.title('(a) Single-Environment', y=-0.16, fontdict={'fontname': 'Times New Roman', 'fontsize': fontsize+3, 'weight': 'bold'})#y=-0.2

        # 第二个子图 (b) Multiple environment
        plt.subplot(2, 1, 2)  # Phr_Exp3
        x_up = len(all_train_round_list_EXP3)
        x_value_list = list(range(1, x_up + 1))

        # 计算 x_value 和 x_tick
        x_value = [x * (xaxis_interval) for x in x_value_list]
        x_tick = [round(x * (xaxis_interval) / x_scaler_dict['EXP3'], 1) for x in x_value_list]

        if k == 0:
            plt.plot(x_value, aveg_P_hr_baseline_list_exp3[:x_up], label='Baseline', alpha=1.0, linewidth=linewidth,
                     color=color_baseline,linestyle='-')
            r1 = list(
                map(lambda x: x[0] - x[1], zip(aveg_P_hr_baseline_list_exp3[:x_up], P_hr_baseline_SE_list_exp3[:x_up])))
            r2 = list(
                map(lambda x: x[0] + x[1], zip(aveg_P_hr_baseline_list_exp3[:x_up], P_hr_baseline_SE_list_exp3[:x_up])))
            plt.fill_between(x_value, r1, r2, color=color_baseline, alpha=alpha)

        plt.plot(x_value, aveg_P_hr_ddpg_list_exp3[:x_up], label=label_dict[RP_id],#'Proposed_{RP_id}'.format(RP_id=RP_id),
                 linestyle=linestyle_dict[RP_id], alpha=1.0, linewidth=linewidth, color=color)
        r1 = list(map(lambda x: x[0] - x[1], zip(aveg_P_hr_ddpg_list_exp3[:x_up], P_hr_ddpg_SE_list_exp3[:x_up])))
        r2 = list(map(lambda x: x[0] + x[1], zip(aveg_P_hr_ddpg_list_exp3[:x_up], P_hr_ddpg_SE_list_exp3[:x_up])))
        plt.fill_between(x_value, r1, r2, color=color, alpha=alpha)

        # 添加浅色网格线
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)  # 设置浅色网格线
        plt.xlabel('Time steps', fontsize=fontsize, fontweight=fontweight)
        plt.ylabel('Percentage', fontsize=fontsize + 10, fontweight=fontweight)
        plt.ylim(ylim)
        #plt.xticks(x_tick, size=fontsize, weight=fontweight)  # 使用缩小后的 x_tick
        ## 使用FuncFormatter将x_tick转换为科学计数法
        ax = plt.gca()
        ax.ticklabel_format(style='sci', scilimits=(-1, 4), axis='x')
        #ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))  # 格式化X轴刻度
        ax.xaxis.get_offset_text().set_fontsize(fontsize)
        plt.xticks(size=fontsize, weight=fontweight)  # 使用缩小后的 x_tick
        plt.yticks(size=fontsize, weight=fontweight)
        plt.title('(b) Multi-Environment', y=-0.16, fontdict={'fontname': 'Times New Roman', 'fontsize': fontsize+3, 'weight': 'bold'})#y=-0.2
        plt.legend(prop={'size': fontsize_legend, 'weight': fontweight}, ncol=ncol)

        # 调整子图之间的垂直间距和底部空白
        plt.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.2)  # 增加 bottom 以增加底部空白

    file_path = '{folder}/Learning_curve_{case_name}_{compare_type}.png'.format(folder=data_folder, case_name=case_name,compare_type=compare_type)
    plt.savefig(file_path, dpi=300)
    plt.close()
    return

#==============================Assign Parameter===================================#
encoder = 0
dist_func = 'dist_pairwise'
dist_func_1 = 'cf_pairwise_distance'
mode = 'ddpg_cf'
case_name = 'diabetic'#'lunar_lander'#'diabetic' # assign case name
lr = '1e-05'
start_index = 20
interval = 20
xaxis_interval = 400
all_metric_df = pd.DataFrame()
all_metric_summary_list = []

##===============basic postprocess to get all metric data=========================#
if case_name=='diabetic':
    baseline_result_folder = r'Project\TD3_results_slurm_e3_RP3_S_UET_id6\1-0'
    baseline_result_exp_list = [29]
    all_result_folder = r'Project\TD3_results_slurm_e3_RP3_S_UET_id6\5-100-0.03'
    all_result_save_path = '{all_result_folder}/{case_name}_case_metric_server.csv'.format(
        all_result_folder=all_result_folder, case_name=case_name)
    all_earlist_train_round_save_path = '{all_result_folder}/{case_name}_case_earlist_train_round_server.csv'.format(
        all_result_folder=all_result_folder, case_name=case_name)
    aveg_earlist_train_round_save_path = '{all_result_folder}/{case_name}_case_aveg_earlist_train_round_server.csv'.format(
        all_result_folder=all_result_folder, case_name=case_name)
    all_trace_summary_save_path = '{all_result_folder}/{case_name}_case_all_trace_summary_server.csv'.format(
        all_result_folder=all_result_folder, case_name=case_name)
    # diabetic case
    exp_type_list = [
        'multi_env_multi_trace_no_action_thre_diff', # for multi_env
        #'one_env_multi_trace_no_action_thre_p7', # for single-env
    ]
    exp_trace_len_dict = {
                    'multi_env_multi_trace_no_action_thre_diff':1000,
                   'one_env_multi_trace_no_action_thre_p7':1000,
                   }
    exp_id_dict = {
                    'multi_env_multi_trace_no_action_thre_diff':[1],
                   'one_env_multi_trace_no_action_thre_p7':[29],
                   }
    max_train_round_dict={
                    'multi_env_multi_trace_no_action_thre_diff':[2000]*100,
                   'one_env_multi_trace_no_action_thre_p7':[2000]*100 ,
    }
    total_trial_num_dict = {
        'multi_env_multi_trace_no_action_thre_diff':[7]*100,
        'one_env_multi_trace_no_action_thre_p7': [7]*100,
    }
    grad_dict = {
        'multi_env_multi_trace_no_action_thre_diff': [50]*100,
        'one_env_multi_trace_no_action_thre_p7': [50]*100,
    }

elif case_name=='lunar_lander':
    baseline_result_folder = r'Project\TD3_results_slurm_e3_RP3_S_UET_id6\1-0.0'
    baseline_result_exp_list = [29]
    all_result_folder = r'Project\TD3_results_slurm_e3_RP3_S_UET_id6\2-0.18-0_0'
    all_result_save_path = '{all_result_folder}/{case_name}_case_metric_server.csv'.format(all_result_folder=all_result_folder, case_name=case_name)
    all_earlist_train_round_save_path = '{all_result_folder}/{case_name}_case_earlist_train_round_server.csv'.format(
        all_result_folder=all_result_folder, case_name=case_name)
    aveg_earlist_train_round_save_path = '{all_result_folder}/{case_name}_case_aveg_earlist_train_round_server.csv'.format(
        all_result_folder=all_result_folder, case_name=case_name)
    all_trace_summary_save_path = '{all_result_folder}/{case_name}_case_all_trace_summary_server.csv'.format(
        all_result_folder=all_result_folder, case_name=case_name)
    #Lunar Lander
    exp_type_list = ['multi_env_multi_trace_no_action_thre_diff', 'one_env_multi_trace_with_action_thre_g10']
    exp_trace_len_dict = {
                    'multi_env_multi_trace_no_action_thre_diff':100,
                   'one_env_multi_trace_no_action_thre_g10':1000,
                   }
    exp_id_dict = {
                    'multi_env_multi_trace_no_action_thre_diff': [57],#[1], #[8], #[15], #[22],#[29],36,43,50,57
                   'one_env_multi_trace_no_action_thre_g10': [],#[1], #[8], #[15], #[22],#[29],
                   }
    max_train_round_dict={
                        'multi_env_multi_trace_no_action_thre_diff': [4000]*180,
                       'one_env_multi_trace_no_action_thre_g10':[4000]*100,
    }
    total_trial_num_dict = {
        'multi_env_multi_trace_no_action_thre_diff': [7] * 180,
        'one_env_multi_trace_no_action_thre_g10': [7] * 90,
    }
    grad_dict = {
        'multi_env_multi_trace_no_action_thre_diff': [20]*180,
        'one_env_multi_trace_no_action_thre_g10': [20] * 90,
    }

earlist_training_round_good_cf_summary_df_all = pd.DataFrame()
earlist_training_round_good_cf_summary_df_list = []
trace_summary_df_all = pd.DataFrame()
trace_summary_df_list = []
snapshot_result_list = []
for exp_type in exp_type_list:
    id_list = exp_id_dict[exp_type]
    print(exp_type, id_list)
    if exp_type in ['multi_env_multi_trace_with_action_thre','multi_env_multi_trace_with_action_thre_diff',
                    'one_env_multi_trace_with_action_thre_p1', 'one_env_multi_trace_with_action_thre_p2',
                    'one_env_multi_trace_with_action_thre_p3','one_env_multi_trace_with_action_thre_p4',
                    'one_env_multi_trace_with_action_thre_p5', 'one_env_multi_trace_with_action_thre_p6', 'one_env_multi_trace_with_action_thre_p7',
                    'one_env_multi_trace_with_action_thre_p8','one_env_multi_trace_with_action_thre_p9', 'one_env_multi_trace_with_action_thre_g10']:
        if case_name == 'diabetic':
            action_threshold = 0.08
        else:
            action_threshold = 0.0
    else:
        if case_name == 'diabetic':
            action_threshold = -1.0
        else:
            action_threshold = -99
    for id in id_list:
        id_index = id_list.index(id)
        whole_trace_len = exp_trace_len_dict[exp_type]
        max_train_round_index = max_train_round_dict[exp_type][id_index]
        total_trial_num = total_trial_num_dict[exp_type][id_index]
        grad = grad_dict[exp_type][id_index]
        baseline_result_exp_id = baseline_result_exp_list[id_index]
        # get result summary
        for trial_index in range(1, total_trial_num+1):
            print('exp_type: ', exp_type, ' ID: ', id, 'baseline_result_exp_id: ', baseline_result_exp_id,' max_train_round_index: ', max_train_round_index, 'trial_index: ', trial_index)
            get_final_result(whole_trace_len, id, encoder, dist_func, dist_func_1, case_name, lr, grad, max_train_round_index,
                            exp_type, action_threshold, trial_index, start_index=start_index,
                             interval=interval, xaxis_interval=xaxis_interval, all_result_folder=all_result_folder,
                             baseline_result_folder=baseline_result_folder, baseline_result_exp_id=baseline_result_exp_id)
        # draw successful rate curve
        if case_name == 'diabetic':
            result_folder = '{all_result_folder}/all_trace_len_{whole_trace_len}_{dist_func}_lr_{lr}_grad_{grad}_{id}'.format(
                all_result_folder=all_result_folder,
                whole_trace_len=whole_trace_len, lr=lr, grad=grad,
                dist_func=dist_func, id=id)
            all_test_index_file_path = '{result_folder}/test_index_file.csv'.format(result_folder=result_folder)
            all_test_index_df = pd.read_csv(all_test_index_file_path, index_col=0).drop_duplicates()
            all_train_index_file_path = '{result_folder}/train_index_file.csv'.format(result_folder=result_folder)
            all_train_index_df = pd.read_csv(all_train_index_file_path, index_col=0).drop_duplicates()
        else:
            result_folder = '{all_result_folder}/all_trace_len_{whole_trace_len}_{dist_func}_lr_{lr}_grad_{grad}_{id}'.format(all_result_folder=all_result_folder,
                whole_trace_len=whole_trace_len,lr=lr,grad=grad,
                dist_func=dist_func, id=id)
            all_test_index_file_path = '{result_folder}/test_index_file.csv'.format(result_folder=result_folder)
            all_test_index_df = pd.read_csv(all_test_index_file_path, index_col=0).drop_duplicates()
            all_train_index_file_path = '{result_folder}/train_index_file.csv'.format(result_folder=result_folder)
            all_train_index_df = pd.read_csv(all_train_index_file_path, index_col=0).drop_duplicates()
        total_test_trace_num = len(all_test_index_df)
        all_train_round_list = np.arange(start_index, max_train_round_index + 1, interval).tolist()
        all_test_index_file_path = '{result_folder}/test_index_file.csv'.format(result_folder=result_folder)
        all_test_index_df = pd.read_csv(all_test_index_file_path, index_col=0).drop_duplicates()
        # # calculate_successful_rate(total_trial_number=total_trial_num, all_train_round_list=all_train_round_list,
        # #                           result_folder=result_folder, save_folder=result_folder, exp_id=id, baseline_result_df=baseline_result_df)
        draw_successful_rate_all_trial(data_folder=result_folder, total_trial_number=total_trial_num, all_train_round_list=all_train_round_list,
                                               save_folder=result_folder, exp_id=id, all_test_index_df=all_test_index_df, xaxis_interval=xaxis_interval,
                                               test_result_type='test',
                                               case_name=case_name)
        draw_successful_rate_all_trial(data_folder=result_folder, total_trial_number=total_trial_num,
                                               all_train_round_list=all_train_round_list,
                                               save_folder=result_folder, exp_id=id, all_test_index_df=all_train_index_df,
                                               xaxis_interval=xaxis_interval,
                                               test_result_type='train',
                                               case_name=case_name)
        # get the metric values of pointwise comparsion
        chosen_test_result_snapshot_this_exp = get_metric_each_trial(data_folder=result_folder, total_trial_number=total_trial_num, all_train_round_list=all_train_round_list,
                                      save_folder=result_folder, exp_id=id,all_test_index_df=all_test_index_df,
                                      total_test_trace_num=total_test_trace_num, xaxis_interval=xaxis_interval,
                                      total_trial_num=total_trial_num,
                                      exp_type=exp_type, case_name=case_name,start_index=start_index,interval=interval,
                                      compare_type='with_pointwise', best_baseline_test_index=None, best_baseline_trial_index=None,
                                                                             test_result_type='test')
        chosen_test_result_snapshot_this_exp = get_metric_each_trial(data_folder=result_folder,
                                                                             total_trial_number=total_trial_num,
                                                                             all_train_round_list=all_train_round_list,
                                                                             save_folder=result_folder, exp_id=id,
                                                                             all_test_index_df=all_train_index_df,
                                                                             total_test_trace_num=total_test_trace_num,
                                                                             xaxis_interval=xaxis_interval,
                                                                             total_trial_num=total_trial_num,
                                                                             exp_type=exp_type, case_name=case_name,
                                                                             start_index=start_index, interval=interval,
                                                                             compare_type='with_pointwise',
                                                                             best_baseline_test_index=None,
                                                                             best_baseline_trial_index=None,
                                                                             test_result_type='train')
        # snapshot_result_list.append(chosen_test_result_snapshot_this_exp)

        all_metric_aveg_df = pd.read_csv('{save_folder}/all_metric_aveg_with_pointwise_test.csv'.format(save_folder=result_folder))
        metric_all_trace_with_pointwise_df = pd.read_csv(
                    '{save_folder}/metric_summary_all_trace_with_pointwise_exp_{id}_test.csv'.format(save_folder=result_folder,id=id))
        metric_each_trace_with_pointwise_df = pd.read_csv(
                    '{save_folder}/metric_summary_each_trace_with_pointwise_exp_{id}_test.csv'.format(save_folder=result_folder,id=id))
        test_index_best_aveg_P_hr_baseline, trial_id_best_P_hr_baseline = get_metric_with_best_baseline(all_metric_aveg_df, metric_all_trace_with_pointwise_df)
        print('test_index_best_aveg_P_hr_baseline: ', test_index_best_aveg_P_hr_baseline, 'trial_id_best_P_hr_baseline: ', trial_id_best_P_hr_baseline)
        # get the metric values of best baseline comparison
        chosen_test_result_snapshot_this_exp_1 = get_metric_each_trial(data_folder=result_folder,
                                                                             total_trial_number=total_trial_num,
                                                                             all_train_round_list=all_train_round_list,
                                                                             save_folder=result_folder, exp_id=id,
                                                                             all_test_index_df=all_test_index_df,
                                                                             total_test_trace_num=total_test_trace_num,
                                                                             xaxis_interval=xaxis_interval,
                                                                             total_trial_num=total_trial_num,
                                                                             exp_type=exp_type, case_name=case_name,
                                                                             start_index=start_index, interval=interval,
                                                                             compare_type='best_baseline',
                                                                             best_baseline_test_index=test_index_best_aveg_P_hr_baseline,
                                                                             best_baseline_trial_index=trial_id_best_P_hr_baseline,
                                                                               test_result_type='test')
        all_metric_aveg_df = pd.read_csv(
                    '{save_folder}/all_metric_aveg_with_pointwise_train.csv'.format(save_folder=result_folder))
        metric_all_trace_with_pointwise_df = pd.read_csv(
                    '{save_folder}/metric_summary_all_trace_with_pointwise_exp_{id}_train.csv'.format(save_folder=result_folder,
                                                                                                     id=id))
        metric_each_trace_with_pointwise_df = pd.read_csv(
                    '{save_folder}/metric_summary_each_trace_with_pointwise_exp_{id}_train.csv'.format(save_folder=result_folder,
                                                                                                      id=id))
        test_index_best_aveg_P_hr_baseline, trial_id_best_P_hr_baseline = get_metric_with_best_baseline(
                    all_metric_aveg_df, metric_all_trace_with_pointwise_df)
        print('test_index_best_aveg_P_hr_baseline: ', test_index_best_aveg_P_hr_baseline,
                      'trial_id_best_P_hr_baseline: ', trial_id_best_P_hr_baseline)
        # get the metric values of best baseline comparison
        chosen_test_result_snapshot_this_exp_1 = get_metric_each_trial(data_folder=result_folder,
                                                                               total_trial_number=total_trial_num,
                                                                               all_train_round_list=all_train_round_list,
                                                                               save_folder=result_folder, exp_id=id,
                                                                               all_test_index_df=all_train_index_df,
                                                                               total_test_trace_num=total_test_trace_num,
                                                                               xaxis_interval=xaxis_interval,
                                                                               total_trial_num=total_trial_num,
                                                                               exp_type=exp_type, case_name=case_name,
                                                                               start_index=start_index, interval=interval,
                                                                               compare_type='best_baseline',
                                                                               best_baseline_test_index=test_index_best_aveg_P_hr_baseline,
                                                                               best_baseline_trial_index=trial_id_best_P_hr_baseline,
                                                                               test_result_type='train')
        # draw_bar_chart_compare(all_metric_aveg_df, metric_all_trace_df, metric_each_trace_df, result_folder,
        #                        case_name=case_name, exp_type=exp_type, exp_id=id)
        draw_all_trace_TD3_action_num_each_exp_RP2(data_folder=result_folder, compare_type='with_pointwise',
                                                           test_result_type='test',
                                                           exp_id=id, trace_index_df=all_test_index_df,
                                                           all_train_round_list=all_train_round_list,
                                                           xaxis_interval=xaxis_interval,total_trial_num=total_trial_num,case_name=case_name)
        draw_all_trace_TD3_action_num_each_exp_RP2(data_folder=result_folder, compare_type='with_pointwise',
                                                           test_result_type='train',
                                                           exp_id=id, trace_index_df=all_train_index_df,
                                                           all_train_round_list=all_train_round_list,
                                                           xaxis_interval=xaxis_interval,total_trial_num=total_trial_num,case_name=case_name)
        # del_unused_baseline_data(data_folder=result_folder, all_train_round_list=all_train_round_list,
        #                          total_trial_num=total_trial_num, test_result_type='train')
        # del_unused_baseline_data(data_folder=result_folder, all_train_round_list=all_train_round_list,
        #                          total_trial_num=total_trial_num, test_result_type='test')
#===============END basic postprocess to get all metric data=========================#


# #============================draw final learning curve==================================#
case_name = 'diabetic'#'lunar_lander'#'diabetic'
exp_type = 'Across_RP'
xaxis_interval = 400
test_result_type = 'test'
metric_name='rho_adv' #rho_plus: Phr, rho_adv:aveg_compare_count
if case_name == 'diabetic':
    exp_1_index = 1000
    exp_3_index = 1240
    data_folder = r'D:\ShuyangDongDocument\UVA\UVA-Research\Project\Counterfactual_Explanation\code\diabetic_example\server\across_rp\final_phr_data'
    s_thre_list = ['0_RP1', '100_RP2', '100-0.03_RP3']  # final
else:
    exp_1_index = 500
    exp_3_index = 2100
    data_folder = r'D:\ShuyangDongDocument\UVA\UVA-Research\Project\Counterfactual_Explanation\code\OpenAI_example\lunar_lander\server\across_rp\final_phr_data'
    s_thre_list = ['0.0_RP1','0.18_RP2','0.18-0_0_RP3'] #final
if metric_name=='rho_adv': #'with_pointwise': for p_hr (rho+), best_baseline: for rho_adv
    compare_type = 'best_baseline'
elif metric_name=='rho_plus':
    compare_type = 'with_pointwise'
all_train_round_list_EXP1 = np.arange(20, exp_1_index + 1, interval).tolist() #LL-EXP1:500, LL-EXP3-2100; Diabetic EXP1-1000, Diabetic EXP3-1240
all_train_round_list_EXP3 = np.arange(20, exp_3_index + 1, interval).tolist()
draw_multiple_Phr_final_figure(data_folder, s_thre_list, all_train_round_list_EXP1, all_train_round_list_EXP3,
                               exp_type, xaxis_interval,compare_type,test_result_type, case_name,metric_name)
# #==================================End draw final learning curve: Phr figure=================================================#


# #===============draw summary curve for all s_thre values=========================#
# case_name = 'lunar_lander'#'lunar_lander'#'diabetic'
# exp_type_id = 3
# RP_id = 2
# trace_group_id = 1
# summary_folder = r'\across_rp\TD3_results_slurm_e{exp_type_id}_RP{RP_id}_S_UET_id{trace_group_id}\summary_result'.format(exp_type_id=exp_type_id, RP_id=RP_id, trace_group_id=trace_group_id)
# ##Compare within RP:
# # s_thre_list = ['120', '100', '80', '0'] #RP2
# # s_thre_list = ['0.18', '0.13', '0.08', '0.0']
# # s_thre_list = ['80-0.03', '100-0.03','120-0.03', '0'] #RP3
# # s_thre_list = ['0.18-0.3_0', '0.13-0.3_0', '0.08-0.3_0', '0.0']
# # # Compare across RP:
# # s_thre_list = ['0.18_RP2', '0.13_RP2', '0.08_RP2', '0.18-0_0_RP3', '0.13-0_0_RP3', '0.08-0_0_RP3', '0.0_RP1']
# # s_thre_list = ['0.13_RP2', '0.18_RP2', '0.13-0_0_RP3', '0.18-0_0_RP3', '0.0_RP1']
# s_thre_list = ['0.18_RP2', '0.18-0_0_RP3', '0.0_RP1'] #final
# # s_thre_list = ['120_RP2', '100_RP2', '80_RP2', '120-0.03_RP3', '100-0.03_RP3', '80-0.03_RP3', '0_RP1']
# # s_thre_list = ['100_RP2', '80_RP2', '100-0.03_RP3', '80-0.03_RP3', '0_RP1']
# #s_thre_list = ['100_RP2', '100-0.03_RP3', '0_RP1'] #final
# all_train_round_list = np.arange(20, 2100 + 1, interval).tolist() #LL-EXP1:760, LL-EXP3-2100; Diabetic EXP1-2000, Diabetic EXP3-2000
# compare_type_list = ['with_pointwise', 'best_baseline']
# test_result_type_list = ['train', 'test']
# exp_type = 'Across_RP'
# exp_id = '{exp_type_id}_RP{RP_id}_UET_id{trace_group_id}'.format(exp_type_id=exp_type_id, RP_id=RP_id, trace_group_id=trace_group_id)
# for compare_type in compare_type_list:
#     for test_result_type in test_result_type_list:
#         #for id in id_list:
#         data_folder = '{summary_folder}/1'.format(summary_folder=summary_folder)
#         draw_multiple_test_result_RP2(data_folder, s_thre_list, all_train_round_list, exp_type,
#                                       exp_id=exp_id, xaxis_interval=xaxis_interval,compare_type=compare_type,test_result_type=test_result_type)
# #===============END draw summary curve for all s_thre values=========================#


#===============draw diabetic running example trace=========================#
# case_name = 'diabetic'#'lunar_lander'#'diabetic'
# data_folder = r'Project\running_example_diabetic\TD3_results_slurm_e1_RP3_S_UET_id14'
# max_train_round_index = 100
# #trial_id = 1
# def draw_running_example_trace(data_folder, max_train_round_index, trial_id, case_name):
#     P1_data_folder = '{data_folder}/RP1/1-0/all_trace_len_1000_dist_pairwise_lr_1e-05_grad_50_29/td3_cf_results/trial_{trial_id}/test'.format(data_folder=data_folder,trial_id=trial_id)
#     P2_data_folder = '{data_folder}/RP2/100/all_trace_len_1000_dist_pairwise_lr_1e-05_grad_50_8/td3_cf_results/trial_{trial_id}/test'.format(data_folder=data_folder,trial_id=trial_id)
#     P3_data_folder = '{data_folder}/RP3/100-0.03/all_trace_len_1000_dist_pairwise_lr_1e-05_grad_50_22/td3_cf_results/trial_{trial_id}/test'.format(data_folder=data_folder,trial_id=trial_id)
#
#     all_test_index_file_path = '{data_folder}/ppo_1_adult_p_7_group_14/test_index_file.csv'.format(data_folder=data_folder)
#     all_test_index_df = pd.read_csv(all_test_index_file_path, index_col=0).drop_duplicates()
#
#     for item, row in all_test_index_df.iterrows():
#         for r in range(100, max_train_round_index+1, 20):
#             if case_name == 'diabetic':
#                 trace_eps, trace_current_step, patient_type, patient_id = row['orig_episode'], row['orig_end_time_index'], \
#                                                                           row['patient_type'], row['patient_id']
#                 trace_step_list = list(range(trace_current_step-20+1, trace_current_step+1))
#                 print(trace_eps, trace_current_step, trace_step_list)
#                 # save figure
#                 figure_folder = '{figure_folder_0}/figures_trial_{trial_id}/{trace_eps}_{trace_current_step}'.format(
#                     figure_folder_0=data_folder,trial_id=trial_id,
#                     trace_eps=trace_eps, trace_current_step=trace_current_step)
#                 mkdir(figure_folder)
#                 # get trace data for this segment
#                 orig_trace_file = '{data_folder}/ppo_1_adult_p_7_group_14/patient_trace_adult#{patient_id}_rf_0_step.csv'.format(data_folder=data_folder,patient_id=patient_id)
#                 P1_ddpg_test_traces = '{P1_data_folder}/cf_traces_test_ddpg_r_{round}.pkl'.format(P1_data_folder=P1_data_folder, trace_eps=trace_eps, trace_current_step=trace_current_step, round=r)
#                 P1_ddpg_test_result = '{P1_data_folder}/all_test_result_ddpg.csv'.format(P1_data_folder=P1_data_folder)
#                 P2_ddpg_test_traces = '{P2_data_folder}/cf_traces_test_ddpg_r_{round}.pkl'.format(
#                                                     P2_data_folder=P2_data_folder, trace_eps=trace_eps, trace_current_step=trace_current_step, round=r)
#                 P2_ddpg_test_result = '{P2_data_folder}/all_test_result_ddpg.csv'.format(P2_data_folder=P2_data_folder)
#                 P3_ddpg_test_traces = '{P3_data_folder}/cf_traces_test_ddpg_r_{round}.pkl'.format(
#                                                     P3_data_folder=P3_data_folder, trace_eps=trace_eps, trace_current_step=trace_current_step, round=r)
#                 P3_ddpg_test_result = '{P3_data_folder}/all_test_result_ddpg.csv'.format(P3_data_folder=P3_data_folder)
#
#                 P1_ddpg_test_result_df = pd.read_csv(P1_ddpg_test_result, index_col=0)
#                 P1_ddpg_test_trace_df = pd.read_pickle(P1_ddpg_test_traces)
#                 P2_ddpg_test_result_df = pd.read_csv(P2_ddpg_test_result, index_col=0)
#                 P2_ddpg_test_trace_df = pd.read_pickle(P2_ddpg_test_traces)
#                 P3_ddpg_test_result_df = pd.read_csv(P3_ddpg_test_result, index_col=0)
#                 P3_ddpg_test_trace_df = pd.read_pickle(P3_ddpg_test_traces)
#                 all_orig_trace_df = pd.read_csv(orig_trace_file, index_col=0)
#                 orig_trace_df = all_orig_trace_df[(all_orig_trace_df['episode']==trace_eps) &
#                                                     (all_orig_trace_df['episode_step'].isin(trace_step_list))]
#                 # Diabetic
#                 env_id = 'simglucose-{patient_type}{patient_id}-v0'.format(patient_type=patient_type,
#                                                                            patient_id=patient_id)
#                 P1_ddpg_test_result_df = P1_ddpg_test_result_df[(P1_ddpg_test_result_df['orig_trace_episode'] == trace_eps)
#                                                           & (P1_ddpg_test_result_df['orig_end_step'] == trace_current_step)
#                                                           & (P1_ddpg_test_result_df['ENV_ID'] == env_id)
#                                                           & (P1_ddpg_test_result_df['train_round'] == r)]
#                 P1_ddpg_test_trace_df = P1_ddpg_test_trace_df[(P1_ddpg_test_trace_df['orig_trace_episode'] == trace_eps)
#                                                         & (P1_ddpg_test_trace_df['orig_end_step'] == trace_current_step)
#                                                         & (P1_ddpg_test_trace_df['ENV_ID'] == env_id)]
#                 P2_ddpg_test_result_df = P2_ddpg_test_result_df[(P2_ddpg_test_result_df['orig_trace_episode'] == trace_eps)
#                                                             & (P2_ddpg_test_result_df['orig_end_step'] == trace_current_step)
#                                                             & (P2_ddpg_test_result_df['ENV_ID'] == env_id)
#                                                             & (P2_ddpg_test_result_df['train_round'] == r)]
#                 P2_ddpg_test_trace_df = P2_ddpg_test_trace_df[(P2_ddpg_test_trace_df['orig_trace_episode'] == trace_eps)
#                                                               & (P2_ddpg_test_trace_df['orig_end_step'] == trace_current_step)
#                                                               & (P2_ddpg_test_trace_df['ENV_ID'] == env_id)]
#                 P3_ddpg_test_result_df = P3_ddpg_test_result_df[(P3_ddpg_test_result_df['orig_trace_episode'] == trace_eps)
#                                                         & (P3_ddpg_test_result_df['orig_end_step'] == trace_current_step)
#                                                         & (P3_ddpg_test_result_df['ENV_ID'] == env_id)
#                                                         & (P3_ddpg_test_result_df['train_round'] == r)]
#                 P3_ddpg_test_trace_df = P3_ddpg_test_trace_df[(P3_ddpg_test_trace_df['orig_trace_episode'] == trace_eps)
#                                                               & (P3_ddpg_test_trace_df['orig_end_step'] == trace_current_step)
#                                                               & (P3_ddpg_test_trace_df['ENV_ID'] == env_id)]
#
#                 orig_obs_trace_this_seg = orig_trace_df['observation_CGM'].tolist()
#                 orig_action_trace_this_seg = orig_trace_df['action'].tolist()
#                 print('orig_action_trace_this_seg: ', orig_action_trace_this_seg)
#                 orig_reward_trace_this_seg = orig_trace_df['reward'].tolist()
#                 J0 = round(sum(orig_trace_df['reward'].tolist()),2)
#                 x_tick = range(1, len(orig_obs_trace_this_seg) + 1)
#                 total_eps_num = 1
#                 # print('best_result_for_lowest_distance_list_train: ', best_result_for_lowest_distance_list_train)
#                 # print('best_result_for_lowest_distance_list_test: ', best_result_for_lowest_distance_list_test)
#
#                 fig = plt.figure(figsize=(20, 12))
#                 fontsize = 27
#                 size_1 = 20
#                 fontweight = 'bold' #fontsize=fontsize, fontweight=fontweight
#                 linewight = 2.5
#                 # shorten vertical distance
#                 plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.15, wspace=0.2)
#                 # Plot Obs
#                 plt.subplot(2, 1, 1)
#                 plt.ticklabel_format(style='plain')
#                 plt.axhline(y=100, color='black', linestyle='--', alpha=.8)
#                 plt.plot(x_tick, orig_obs_trace_this_seg, color='b', linewidth=linewight, marker='.', label='Observed')
#                 for eps in range(0, total_eps_num):
#                     # print(trace_eps, trace_current_step, eps)
#                     P1_ddpg_reward_sum_this_seg = round(sum(P1_ddpg_test_trace_df[P1_ddpg_test_trace_df['episode'] == eps]['reward'].tolist()), 2)
#                     P2_ddpg_reward_sum_this_seg = round(sum(P2_ddpg_test_trace_df[P2_ddpg_test_trace_df['episode'] == eps]['reward'].tolist()), 2)
#                     P3_ddpg_reward_sum_this_seg = round(sum(P3_ddpg_test_trace_df[P3_ddpg_test_trace_df['episode'] == eps]['reward'].tolist()), 2)
#
#                     P1_ddpg_obs_trace_this_seg = P1_ddpg_test_trace_df[P1_ddpg_test_trace_df['episode'] == eps]['observation_CGM'].tolist()
#                     P2_ddpg_obs_trace_this_seg = P2_ddpg_test_trace_df[P2_ddpg_test_trace_df['episode'] == eps]['observation_CGM'].tolist()
#                     P3_ddpg_obs_trace_this_seg = P3_ddpg_test_trace_df[P1_ddpg_test_trace_df['episode'] == eps]['observation_CGM'].tolist()
#                     if len(P1_ddpg_obs_trace_this_seg) > 20:
#                         P1_ddpg_obs_trace_this_seg = P1_ddpg_obs_trace_this_seg[:20]
#                     ddpg_tick = range(1, len(P1_ddpg_obs_trace_this_seg) + 1)
#                     plt.plot(ddpg_tick, P3_ddpg_obs_trace_this_seg, linewidth=linewight, color='purple', linestyle='dotted',
#                              marker='^', label='Counterfactual_1'.format(tr=P3_ddpg_reward_sum_this_seg), alpha=.8)
#                     plt.plot(ddpg_tick, P2_ddpg_obs_trace_this_seg, linewidth=linewight, color='red', linestyle='-.',
#                              marker='x', label='Counterfactual_2'.format(tr=P2_ddpg_reward_sum_this_seg), alpha=.8)
#                     plt.plot(ddpg_tick, P1_ddpg_obs_trace_this_seg, linewidth=linewight,color='green', linestyle='--',
#                              marker='+',label='Counterfactual_3'.format(tr=P1_ddpg_reward_sum_this_seg), alpha=.8)
#
#
#                 #plt.title('obs_trace_compare_round_{round}_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}'.format(
#                 #                   round=r, trace_eps=trace_eps, trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id))
#                 plt.ylabel('Glucose (mg/dL)',fontsize=fontsize, fontweight=fontweight)
#                 plt.xticks(x_tick, size=size_1, weight=fontweight)
#                 plt.yticks(size=fontsize, weight=fontweight)
#                 #plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
#                 #plt.xlabel('Time Step', fontsize=fontsize, fontweight=fontweight)
#                 plt.legend(prop={'size':size_1,'weight':fontweight}, ncol=1, loc='upper right')
#
#                 # plot action
#                 x_tick = np.arange(1, 21)
#                 plt.subplot(2, 1, 2)
#                 plt.ticklabel_format(style='plain')
#                 for eps in range(0, total_eps_num):
#                     P1_ddpg_action_trace_this_seg = P1_ddpg_test_trace_df[P1_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                     P1_ddpg_action_dist_this_seg = round(P1_ddpg_test_result_df['cf_pairwise_distance'].tolist()[eps],2)
#                     P2_ddpg_action_trace_this_seg = P2_ddpg_test_trace_df[P2_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                     P3_ddpg_action_trace_this_seg = P3_ddpg_test_trace_df[P3_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                     print('P1_ddpg_action_trace_this_seg: ', P1_ddpg_action_trace_this_seg)
#                     if len(P1_ddpg_action_trace_this_seg) > 20:
#                         P1_ddpg_action_trace_this_seg = P1_ddpg_action_trace_this_seg[:20]
#                     data = [orig_action_trace_this_seg, P3_ddpg_action_trace_this_seg, P2_ddpg_action_trace_this_seg, P1_ddpg_action_trace_this_seg]
#                     data_labels = ['Observed', 'Counterfactual_1','Counterfactual_2','Counterfactual_3']
#                     color_list = ['blue', 'purple', 'red', 'green']
#                     hatch_patterns = ['/', '-', '.', '\\']
#                     bar_width = 0.9 / len(data)
#                     for i in range(len(data)):
#                         plt.bar(x_tick + (i - (len(data) - 1) / 2) * bar_width, data[i], width=bar_width,
#                                 label=data_labels[i], color=color_list[i], hatch=hatch_patterns[i], alpha=0.6)
#                 # plt.title('action_compare_P1')
#                 plt.ylabel('Insulin (U/min)', fontsize=fontsize, fontweight=fontweight)
#                 plt.xticks(x_tick, size=size_1, weight=fontweight)
#                 plt.yticks(size=fontsize, weight=fontweight)
#                 plt.xlabel('Time steps', fontsize=fontsize, fontweight=fontweight)
#                 # plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
#                 plt.ylim(0.0, 0.4)
#                 y_major_locator = MultipleLocator(0.1)
#                 ax = plt.gca()
#                 ax.yaxis.set_major_locator(y_major_locator)
#                 plt.legend(prop={'size':size_1,'weight':fontweight}, ncol=4)
#
#                 # # plot P1-ORIG action
#                 # x_tick = np.arange(1, 21)
#                 # plt.subplot(4, 1, 2)
#                 # plt.ticklabel_format(style='plain')
#                 # for eps in range(0, total_eps_num):
#                 #     P1_ddpg_action_trace_this_seg = P1_ddpg_test_trace_df[P1_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                 #     P1_ddpg_action_dist_this_seg = round(P1_ddpg_test_result_df['cf_pairwise_distance'].tolist()[eps], 2)
#                 #     #P2_ddpg_action_trace_this_seg = P2_ddpg_test_trace_df[P2_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                 #     #P3_ddpg_action_trace_this_seg = P3_ddpg_test_trace_df[P3_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                 #     print('P1_ddpg_action_trace_this_seg: ', P1_ddpg_action_trace_this_seg)
#                 #     if len(P1_ddpg_action_trace_this_seg) > 20:
#                 #         P1_ddpg_action_trace_this_seg = P1_ddpg_action_trace_this_seg[:20]
#                 #     data = [P1_ddpg_action_trace_this_seg, orig_action_trace_this_seg]
#                 #     data_labels = ['CF1 (distance: {dist})'.format(dist=P1_ddpg_action_dist_this_seg), 'Original trace']
#                 #     color_list = ['green', 'red']
#                 #     hatch_patterns = ['/', '-']
#                 #     bar_width = 0.8 / len(data)
#                 #     for i in range(len(data)):
#                 #         plt.bar(x_tick + (i - (len(data) - 1) / 2) * bar_width, data[i], width=bar_width, label=data_labels[i], color=color_list[i],hatch=hatch_patterns[i])
#                 # #plt.title('action_compare_P1')
#                 # plt.ylabel('Insulin', fontsize=fontsize, fontweight=fontweight)
#                 # plt.xticks(x_tick, size=fontsize, weight=fontweight)
#                 # plt.yticks(size=fontsize, weight=fontweight)
#                 # plt.xlabel('Time Step',fontsize=fontsize, fontweight=fontweight)
#                 # #plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
#                 # plt.ylim(0.0, 1.5)
#                 # plt.legend(fontsize=fontsize,  prop={'weight':fontweight})
#                 #
#                 # # plot P2-ORIG action
#                 # plt.subplot(4, 1, 3)
#                 # plt.ticklabel_format(style='plain')
#                 # for eps in range(0, total_eps_num):
#                 #     #P1_ddpg_action_trace_this_seg = P1_ddpg_test_trace_df[P1_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                 #     P2_ddpg_action_trace_this_seg = P2_ddpg_test_trace_df[P2_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                 #     P2_ddpg_action_dist_this_seg = round(P2_ddpg_test_result_df['cf_pairwise_distance'].tolist()[eps], 2)
#                 #     # P3_ddpg_action_trace_this_seg = P3_ddpg_test_trace_df[P3_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                 #     print('P2_ddpg_action_trace_this_seg: ', P2_ddpg_action_trace_this_seg)
#                 #     if len(P2_ddpg_action_trace_this_seg) > 20:
#                 #         P2_ddpg_action_trace_this_seg = P2_ddpg_action_trace_this_seg[:20]
#                 #     data = [P2_ddpg_action_trace_this_seg, orig_action_trace_this_seg]
#                 #     data_labels = ['CF2 (distance: {dist})'.format(dist=P2_ddpg_action_dist_this_seg), 'Original trace']
#                 #     color_list = ['green', 'red']
#                 #     hatch_patterns = ['/', '-']
#                 #     bar_width = 0.8 / len(data)
#                 #     for i in range(len(data)):
#                 #         plt.bar(x_tick + (i - (len(data) - 1) / 2) * bar_width, data[i], width=bar_width, label=data_labels[i],
#                 #                 color=color_list[i], hatch=hatch_patterns[i])
#                 # #plt.title('action_compare_P2')
#                 # plt.ylabel('Insulin',fontsize=fontsize, fontweight=fontweight)
#                 # plt.xticks(x_tick, size=fontsize, weight=fontweight)
#                 # plt.yticks(size=fontsize, weight=fontweight)
#                 # plt.xlabel('Time Step',fontsize=fontsize, fontweight=fontweight)
#                 # # plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
#                 # plt.ylim(0.0, 1.5)
#                 # plt.legend(fontsize=fontsize, prop={'weight':fontweight})
#                 #
#                 # # plot P1-ORIG action
#                 # plt.subplot(4, 1, 4)
#                 # plt.ticklabel_format(style='plain')
#                 # for eps in range(0, total_eps_num):
#                 #     # P1_ddpg_action_trace_this_seg = P1_ddpg_test_trace_df[P1_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                 #     # P2_ddpg_action_trace_this_seg = P2_ddpg_test_trace_df[P2_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                 #     P3_ddpg_action_trace_this_seg = P3_ddpg_test_trace_df[P3_ddpg_test_trace_df['episode'] == eps]['action'].tolist()
#                 #     P3_ddpg_action_dist_this_seg = round(P3_ddpg_test_result_df['cf_pairwise_distance'].tolist()[eps], 2)
#                 #     print('P3_ddpg_action_trace_this_seg: ', P3_ddpg_action_trace_this_seg)
#                 #     if len(P3_ddpg_action_trace_this_seg) > 20:
#                 #         P3_ddpg_action_trace_this_seg = P3_ddpg_action_trace_this_seg[:20]
#                 #     data = [P3_ddpg_action_trace_this_seg, orig_action_trace_this_seg]
#                 #     data_labels = ['CF3 (distance: {dist})'.format(dist=P3_ddpg_action_dist_this_seg), 'Original trace']
#                 #     color_list = ['green', 'red']
#                 #     hatch_patterns = ['/', '-']
#                 #     bar_width = 0.8 / len(data)
#                 #     for i in range(len(data)):
#                 #         plt.bar(x_tick + (i - (len(data) - 1) / 2) * bar_width, data[i], width=bar_width, label=data_labels[i],
#                 #                 color=color_list[i], hatch=hatch_patterns[i])
#                 # #plt.title('action_compare_P3')
#                 # plt.ylabel('Insulin',fontsize=fontsize, fontweight=fontweight)
#                 # plt.xticks(x_tick, size=fontsize, weight=fontweight)
#                 # plt.yticks(size=fontsize, weight=fontweight)
#                 # plt.xlabel('Time Step',fontsize=fontsize, fontweight=fontweight)
#                 # # plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
#                 # plt.ylim(0.0, 1.5)
#                 # plt.legend(fontsize=fontsize, prop={'weight':fontweight})
#
#
#                 file_path = '{folder}/trace_compare_trial_{trial_id}_round_{round}_{patient_type}_{patient_id}_{trace_eps}_{trace_current_step}.png'.format(
#                     folder=figure_folder, round=r, trace_eps=trace_eps,trial_id=trial_id,
#                     trace_current_step=trace_current_step, patient_type=patient_type, patient_id=patient_id)
#                 plt.savefig(file_path,dpi=300)
#                 plt.close()
#             else:
#                 trace_eps, trace_current_step, gravity, patient_type, patient_id = row['orig_episode'], row[
#                     'orig_end_time_index'], row['gravity'], None, None
#                 figure_folder = '{figure_folder_0}/{trace_eps}_{trace_current_step}'.format(
#                     figure_folder_0=figure_folder_0,
#                     trace_eps=trace_eps, trace_current_step=trace_current_step)
#                 mkdir(figure_folder)
#                 # TODO: to be removed
#                 # trace_eps_int_8 = np.int8(trace_eps)
#                 # print(trace_eps, trace_current_step, r)
#                 ddpg_test_traces = '{result_folder}/test/cf_traces_test_r_{round}.pkl'.format(
#                     result_folder=ddpg_trace_folder,
#                     trace_eps=trace_eps,
#                     trace_current_step=trace_current_step, round=r)
#                 ddpg_test_result = '{result_folder}/test/all_test_result.csv'.format(result_folder=ddpg_trace_folder)
#                 # Lunar lander
#                 baseline_traces = '{result_folder}/cf_trace_file_test_baseline_trained_0_g_{gravity}_{trace_eps}_{trace_current_step}.csv'.format(
#                     result_folder=baseline_trace_folder, trace_eps=trace_eps, trace_current_step=trace_current_step,
#                     gravity=gravity)
#                 baseline_results = '{result_folder}/accumulated_reward_test_baseline_trained_0_g_{gravity}_{trace_eps}_{trace_current_step}.csv'.format(
#                     result_folder=baseline_trace_folder, trace_eps=trace_eps, trace_current_step=trace_current_step,
#                     gravity=gravity)
#                 orig_trace_file = '{result_folder}/orig_trace_test_g_{gravity}_{trace_eps}_{trace_current_step}.csv'.format(
#                     result_folder=baseline_trace_folder, trace_eps=trace_eps, trace_current_step=trace_current_step,
#                     gravity=gravity)
#                 ddpg_test_result_df = pd.read_csv(ddpg_test_result, index_col=0)
#                 baseline_result_df = pd.read_csv(baseline_results, index_col=0)
#                 ddpg_test_trace_df = pd.read_pickle(ddpg_test_traces)
#                 # ddpg_test_trace_df['orig_trace_episode'].astype('int64')
#                 # ddpg_test_trace_df['orig_end_step'].astype('int64')
#                 # ddpg_test_trace_df['step'].astype('int64')
#                 # ddpg_test_trace_df['episode'].astype('int64')
#                 # ddpg_test_trace_df['episode_step'].astype('int64')
#                 # print(ddpg_test_trace_df.info(),ddpg_test_trace_df['orig_trace_episode'].tolist())
#                 ddpg_test_trace_df.to_csv(
#                     '{result_folder}/test/cf_traces_test_r_{round}.csv'.format(result_folder=ddpg_trace_folder,
#                                                                                trace_eps=trace_eps,
#                                                                                trace_current_step=trace_current_step,
#                                                                                round=r))
#                 baseline_trace_df = pd.read_csv(baseline_traces, index_col=0)
#                 orig_trace_df = pd.read_csv(orig_trace_file, index_col=0)
#                 # Lunar lander
#                 ddpg_test_result_df = ddpg_test_result_df[(ddpg_test_result_df['orig_trace_episode'] == trace_eps)
#                                                           & (ddpg_test_result_df['orig_end_step'] == trace_current_step)
#                                                           & (ddpg_test_result_df['gravity'] == gravity)
#                                                           & (ddpg_test_result_df['train_round'] == r)]
#                 ddpg_test_trace_df = ddpg_test_trace_df[(ddpg_test_trace_df['orig_trace_episode'] == trace_eps)
#                                                         & (ddpg_test_trace_df['orig_end_step'] == trace_current_step)
#                                                         & (ddpg_test_trace_df['gravity'] == gravity)]
#                 # print('df info: ',ddpg_test_trace_df.info())
#
#                 orig_obs_trace_this_seg = orig_trace_df['observation'].tolist()
#                 orig_obs_trace_dict_this_seg = get_lunar_lander_trace_list(orig_obs_trace_this_seg, trace_type='obs',
#                                                                            exp_type='orig_trace')
#                 orig_action_trace_this_seg = orig_trace_df['action'].tolist()
#                 orig_action_trace_dict_this_seg = get_lunar_lander_trace_list(orig_action_trace_this_seg,
#                                                                               trace_type='action',
#                                                                               exp_type='orig_trace')
#                 orig_reward_trace_this_seg = orig_trace_df['reward'].tolist()
#                 J0 = sum(orig_trace_df['reward'].tolist())
#                 x_tick = range(1, len(orig_obs_trace_this_seg) + 1)
#                 total_eps_num = 10
#                 # print('best_result_for_lowest_distance_list_train: ', best_result_for_lowest_distance_list_train)
#                 # print('best_result_for_lowest_distance_list_test: ', best_result_for_lowest_distance_list_test)
#                 fig = plt.figure(figsize=(10, 80))
#                 plt.subplots_adjust(left=None, bottom=None, top=None, right=None, hspace=0.3, wspace=None)
#                 num_sub_polt = 10
#                 for obs_index in range(1, 7):
#                     plt.subplot(num_sub_polt, 1, obs_index)
#                     plt.ticklabel_format(style='plain')
#                     plt.plot(x_tick, orig_obs_trace_dict_this_seg[obs_index - 1], color='b', linewidth=1.0,
#                              label='orig_obs_{obs_index}_{trace_eps}_{trace_current_step}_r_{round}'.format(
#                                  obs_index=obs_index,
#                                  trace_eps=trace_eps, trace_current_step=trace_current_step, round=r))
#                     for eps in range(0, total_eps_num):
#                         # print(trace_eps, trace_current_step, eps)
#                         # print('Begin DDPG trace.')
#                         ddpg_obs_trace_this_seg = ddpg_test_trace_df[ddpg_test_trace_df['episode'] == eps][
#                             'observation'].tolist()
#                         if len(ddpg_obs_trace_this_seg) > 20:
#                             ddpg_obs_trace_this_seg = ddpg_obs_trace_this_seg[:20]
#                         # print('ddpg_obs_trace_this_seg: ', ddpg_obs_trace_this_seg)
#                         for a in range(len(ddpg_obs_trace_this_seg)):
#                             ddpg_obs_trace_this_seg[a] = ddpg_obs_trace_this_seg[a].tolist()
#                         ddpg_obs_trace_dict_this_seg = get_lunar_lander_trace_list(ddpg_obs_trace_this_seg,
#                                                                                    trace_type='obs',
#                                                                                    exp_type='ddpg_trace')
#                         plt.plot(x_tick, ddpg_obs_trace_dict_this_seg[obs_index - 1], linewidth=0.2,
#                                  marker='.',
#                                  label='ddpg_obs_{obs_index}_test_{trace_eps}_{trace_current_step}_r_{round}'.format(
#                                      obs_index=obs_index,
#                                      trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.5)
#                     plt.title(
#                         'obs_{obs_index}_trace_compare_round_{round}_{gravity}_{trace_eps}_{trace_current_step}'.format(
#                             obs_index=obs_index, round=r, trace_eps=trace_eps,
#                             trace_current_step=trace_current_step, gravity=gravity))
#                     plt.ylabel('Value')
#                     plt.xticks(x_tick)
#                     plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
#                     # plt.legend()
#                 ## Action
#                 for action_index in range(7, 9):
#                     plt.subplot(num_sub_polt, 1, action_index)
#                     plt.ticklabel_format(style='plain')
#                     plt.plot(x_tick, orig_action_trace_dict_this_seg[action_index - 7], color='b', linewidth=1.0,
#                              label='orig_action_{action_index}_{trace_eps}_{trace_current_step}_r_{round}'.format(
#                                  action_index=action_index - 6,
#                                  trace_eps=trace_eps, trace_current_step=trace_current_step, round=r))
#                     for eps in range(0, total_eps_num):
#                         ddpg_action_trace_this_seg = ddpg_test_trace_df[ddpg_test_trace_df['episode'] == eps][
#                             'action'].tolist()
#                         if len(ddpg_action_trace_this_seg) > 20:
#                             ddpg_action_trace_this_seg = ddpg_action_trace_this_seg[:20]
#                         for a in range(len(ddpg_action_trace_this_seg)):
#                             ddpg_action_trace_this_seg[a] = ddpg_action_trace_this_seg[a].tolist()
#                         ddpg_action_trace_dict_this_seg = get_lunar_lander_trace_list(ddpg_action_trace_this_seg,
#                                                                                       trace_type='action',
#                                                                                       exp_type='ddpg_trace')
#                         plt.plot(x_tick, ddpg_action_trace_dict_this_seg[action_index - 7], linewidth=0.2,
#                                  marker='.',
#                                  label='ddpg_action_{action_index}_test_{trace_eps}_{trace_current_step}_r_{round}'.format(
#                                      action_index=action_index - 7,
#                                      trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.5)
#                     plt.title(
#                         'action_{action_index}_trace_compare_round_{round}_{gravity}_{trace_eps}_{trace_current_step}'.format(
#                             action_index=action_index - 7, round=r, trace_eps=trace_eps,
#                             trace_current_step=trace_current_step, gravity=gravity))
#                     plt.ylabel('Value')
#                     plt.xticks(x_tick)
#                     plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
#                 # plt.legend()
#                 ## Reward
#                 plt.subplot(num_sub_polt, 1, 10)
#                 plt.ticklabel_format(style='plain')
#                 plt.plot(x_tick, orig_reward_trace_this_seg, color='b', linewidth=1.0,
#                          label='orig_reward_{trace_eps}_{trace_current_step}_r_{round}'.format(
#                              trace_eps=trace_eps, trace_current_step=trace_current_step, round=r))
#                 for eps in range(0, total_eps_num):
#                     ddpg_reward_trace_this_seg = ddpg_test_trace_df[ddpg_test_trace_df['episode'] == eps][
#                         'reward'].tolist()
#                     if len(ddpg_reward_trace_this_seg) > 20:
#                         ddpg_reward_trace_this_seg = ddpg_reward_trace_this_seg[:20]
#                     plt.plot(x_tick, ddpg_reward_trace_this_seg, linewidth=0.2,
#                              marker='.',
#                              label='ddpg_reward_test_{trace_eps}_{trace_current_step}_r_{round}'.format(
#                                  trace_eps=trace_eps, trace_current_step=trace_current_step, round=r), alpha=.5)
#                 plt.title(
#                     'reward_trace_compare_round_{round}_{gravity}_{trace_eps}_{trace_current_step}'.format(
#                         round=r, trace_eps=trace_eps,
#                         trace_current_step=trace_current_step, gravity=gravity))
#                 plt.ylabel('Value')
#                 plt.xticks(x_tick)
#                 plt.xlabel('Time step (J0 = {j0})'.format(j0=round(J0, 4)))
#                 # plt.legend()
#                 file_path = '{folder}/trace_compare_round_{round}_{gravity}_{trace_eps}_{trace_current_step}.png'.format(
#                     folder=figure_folder, round=r, trace_eps=trace_eps,
#                     trace_current_step=trace_current_step, gravity=gravity)
#                 plt.savefig(file_path)
#                 plt.close()
#     return
# for trial_id in range(4, 5):
#     draw_running_example_trace(data_folder, max_train_round_index, trial_id, case_name)
#===============END draw diabetic running example trace=========================#

# #all_result_folder = r'Project\Counterfactual_Explanation\code\OpenAI_example\lunar_lander\DDPG_results_final'
# snapshot_df = pd.concat(snapshot_result_list)
# snapshot_df.to_csv('{result_folder}/Chosen_exp_result_{case_name}.csv'.format(result_folder=all_result_folder,case_name=case_name))



# save_folder=r'{all_result_folder}\all_trace_len_1500_dist_pairwise_lr_1e-05_grad_20_26'.format(all_result_folder=all_result_folder)
# all_metric_aveg_df=pd.read_csv('{save_folder}/all_metric_aveg.csv'.format(save_folder=save_folder))
# metric_all_trace_df=pd.read_csv('{save_folder}/metric_summary_all_trace_exp_26.csv'.format(save_folder=save_folder))
# metric_each_trace_df=pd.read_csv('{save_folder}/metric_summary_each_trace_exp_26.csv'.format(save_folder=save_folder))
# draw_bar_chart_compare(all_metric_aveg_df, metric_all_trace_df, metric_each_trace_df, save_folder,case_name=case_name, exp_type='single_env', exp_id=26)




# all_metric_df = pd.concat(all_metric_summary_list)
# all_metric_df.to_csv(all_result_save_path)
# earlist_training_round_good_cf_summary_df_all = pd.concat(earlist_training_round_good_cf_summary_df_list)
# earlist_training_round_good_cf_summary_df_all.to_csv(all_earlist_train_round_save_path)
# all_trace_summary_df = pd.concat(trace_summary_df_list)
# all_trace_summary_df.to_csv(all_trace_summary_save_path)

#get_each_patient_result_separate(all_trace_summary_df, all_result_folder)

# all_trace_summary_df = pd.read_csv(all_trace_summary_save_path)
# figure_folder = r'Project\Counterfactual_Explanation\code\diabetic_example\server\DDPG_results_slurm'
# draw_distribution_no_better_cf_trace_first_state(all_trace_summary_df,figure_folder)

# # draw the aveg earlist training round for each exp type
# aveg_earlist_training_round_result_save_path = '{all_result_folder}/{case_name}_case_aveg_earlist_training_round_server.csv'.format(all_result_folder=all_result_folder,case_name=case_name)
# earlist_training_round_good_cf_summary_df_all = pd.read_csv(all_earlist_train_round_save_path, index_col=0)
# aveg_result_df_list = []
# for exp_type in exp_type_list:
#     new_df = get_earlist_training_round_aveg_one_exp_type(earlist_training_round_good_cf_summary_df_all, exp_type)
#     aveg_result_df_list.append(new_df)
# earlist_training_round_all_df = pd.concat(aveg_result_df_list, axis=1)
# #print('earlist_training_round_all_df: ', earlist_training_round_all_df)
# earlist_training_round_all_df.to_csv(aveg_earlist_training_round_result_save_path)
# #draw_earlist_training_round_aveg_all_exp_type(earlist_training_round_all_df, exp_type_list, all_result_folder)

# # draw each trace state, action and reward trace
# exp_type_list = ['one_env_multi_trace_no_action_thre_p9']
# for exp_type in exp_type_list:
#     for exp_id in exp_id_dict[exp_type]:
#         id_index = exp_id_dict[exp_type].index(exp_id)
#         max_train_round_index = max_train_round_dict[exp_type][id_index]
#         print('exp_id: ', exp_id, ' max_train_round_index: ', max_train_round_index)
#         draw_each_trace_compare(max_train_round_index=max_train_round_index, whole_trace_len=exp_trace_len_dict[exp_type], encoder=encoder,
#                                 dist_func=dist_func, id=exp_id, case_name=case_name)

############################################################
