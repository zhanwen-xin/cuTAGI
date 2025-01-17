import numpy as np
from src.RL_functions.kalman_filter import KalmanFilter
from src.RL_functions.generate_one_synthetic_time_series import generate_one_synthetic_time_series
import copy
import matplotlib.pyplot as plt
from matplotlib import gridspec

def _get_look_back_time_steps(current_step, step_look_back=0):
    look_back_step_list = [0]
    current = 1
    while current <= step_look_back:
        look_back_step_list.append(current)
        current *= 2
    look_back_step_list = [current_step - i for i in look_back_step_list]

    return look_back_step_list

def _hidden_states_collector(current_step, hidden_states_all_step_mu, hidden_states_all_step_var):
        hidden_states_all_step_numpy = {'mu': np.array(hidden_states_all_step_mu), \
                                'var': np.array(hidden_states_all_step_var)}
        look_back_steps_list = _get_look_back_time_steps(current_step)
        hidden_states_collected = {'mu': hidden_states_all_step_numpy['mu'][look_back_steps_list], \
                                    'var': hidden_states_all_step_numpy['var'][look_back_steps_list]}
        return hidden_states_collected

def estimate_hs_distribution(components, time_step_interval, hyperparameters, num_steps, x_init, x_init_d, kf, kf_d, num_ts = 10):
    # Collect samples
    ARd_samples = []
    LLd_samples = []
    LTd_samples = []
    x_samples = []
    for i in range(num_ts):
        np.random.seed(10+i)

        data_generator_i = generate_one_synthetic_time_series(components = components, 
                                                            time_step_interval = time_step_interval, 
                                                            hyperparameters = hyperparameters, 
                                                            num_steps = num_steps,
                                                            x_init = x_init)

        ############## Run two models ################
        x_last_step = x_init
        x_last_step_d = x_init_d
        LLd_mu, LTd_mu, ARd_mu = [], [], []
        LLd_var, LTd_var, ARd_var = [], [], []
        for i in range(num_steps):
            ############ Base model ############
            _, x_pred, _ = kf.predict(x_last_step)
            x_updated, _, _ = kf.update(data_generator_i.time_series['y'][i])

            ############ Meta-AR model ############
            pred_AR, xd_pred, _ = kf_d.predict(x_last_step_d)
            pred_AR['mu'] = pred_AR['mu'].item()
            pred_AR['var'] = pred_AR['var'].item()

            # Smoother equations to update
            target_AR = {'mu': x_pred['mu'][-1], 'var': x_pred['var'][-1,-1]}
            cov_x_AR = xd_pred['var'] @ kf_d.F.T
            xd_updated = kf_d.dist_update(target_AR, pred_AR, xd_pred, cov_x_AR)

            x_last_step = x_updated
            x_last_step_d = xd_updated

            LLd_mu.append(xd_pred['mu'][0])
            LTd_mu.append(xd_pred['mu'][1])
            ARd_mu.append(xd_pred['mu'][-1])
            LLd_var.append(xd_pred['var'][0,0])
            LTd_var.append(xd_pred['var'][1,1])
            ARd_var.append(xd_pred['var'][-1,-1])

            if i > 64:
                ARd_history = _hidden_states_collector(-1, ARd_mu, ARd_var)
                ARd_samples.append(ARd_history['mu'].tolist())
                LLd_history = _hidden_states_collector(-1, LLd_mu, LLd_var)
                LLd_samples.append(LLd_history['mu'].tolist())
                LTd_history = _hidden_states_collector(-1, LTd_mu, LTd_var)
                LTd_samples.append(LTd_history['mu'].tolist())
                x_samples.append(LTd_history['mu'].tolist())

    ARd_samples = np.array(ARd_samples)
    LLd_samples = np.array(LLd_samples)
    LTd_samples = np.array(LTd_samples)
    x_samples = np.array(x_samples)
    
    x_samples_mean = np.mean(x_samples, axis=0)
    x_samples_cov = np.cov(x_samples.T)


    return x_samples_mean, x_samples_cov

def critic(base_kf_model, drift_kf_model, x_last_step, x_last_step_d, intervention_hidden_state, drift_model_initial_state, observation, mv_normal_x):
    ## Critic
    ############## Store the x_last_step and x_last_step_d ################
    x_ls = copy.deepcopy(x_last_step)
    x_ls_d = copy.deepcopy(x_last_step_d)
    x_last_step_temp = copy.deepcopy(x_last_step)
    x_last_step_d_temp = copy.deepcopy(x_last_step_d)
    kf_intervene = copy.deepcopy(base_kf_model)
    kf_remain = copy.deepcopy(base_kf_model)
    kfd_intervene = copy.deepcopy(drift_kf_model)
    kfd_remain = copy.deepcopy(drift_kf_model)

    #####################################################
    ####### Likelihood when intervention is taken #######
    #####################################################
    # x_ls['mu'][0] += intervention_hidden_state['mu'][0]
    # x_ls['mu'][1] += intervention_hidden_state['mu'][1]
    # x_ls['var'][0,0] += intervention_hidden_state['var'][0,0]
    # x_ls['var'][1,1] += intervention_hidden_state['var'][1,1]
    # x_ls['mu'][-1] = intervention_hidden_state['mu'][-1]
    # x_ls['var'][-1,-1] = intervention_hidden_state['var'][-1,-1]

    x_ls = copy.deepcopy(intervention_hidden_state)

    x_ls_d = copy.deepcopy(drift_model_initial_state)
    x_ls_d['mu'][-1] = x_ls['mu'][-1]
    x_ls_d['var'][-1,-1] = x_ls['var'][-1,-1]

    # # Base model
    y_pred_a, x_pred_a, _ = kf_intervene.predict(x_ls)
    _, y_likelihood_a, _ = kf_intervene.update(observation)

    # # Drift model
    pred_AR_a, xd_pred_a, _ = kfd_intervene.predict(x_ls_d)
    pred_AR_a['mu'] = pred_AR_a['mu'].item()
    pred_AR_a['var'] = pred_AR_a['var'].item()

    target_AR = {'mu': x_pred_a['mu'][-1], 'var': x_pred_a['var'][-1,-1]}
    cov_x_AR = xd_pred_a['var'] @ kfd_intervene.F.T
    xd_updated_intervene = kfd_intervene.dist_update(target_AR, pred_AR_a, xd_pred_a, cov_x_AR)

    x_likelihood_a = mv_normal_x.pdf(xd_pred_a['mu'][1])

    #####################################################
    ##### Likelihood when intervention is not taken #####
    #####################################################
    x_ls = x_last_step_temp
    x_ls_d = x_last_step_d_temp

    # # Base model
    y_pred_na, x_pred_na, _ = kf_remain.predict(x_ls)
    _, y_likelihood_na, _ = kf_remain.update(observation)

    # # Drift model
    pred_AR_na, xd_pred_na, _ = kfd_remain.predict(x_ls_d)
    pred_AR_na['mu'] = pred_AR_na['mu'].item()
    pred_AR_na['var'] = pred_AR_na['var'].item()

    target_AR = {'mu': x_pred_na['mu'][-1], 'var': x_pred_na['var'][-1,-1]}
    cov_x_AR = xd_pred_na['var'] @ kfd_remain.F.T
    xd_updated_remain = kfd_remain.dist_update(target_AR, pred_AR_na, xd_pred_na, cov_x_AR)

    x_likelihood_na = mv_normal_x.pdf(xd_pred_na['mu'][1])

    return y_likelihood_a, y_likelihood_na, x_likelihood_a, x_likelihood_na, xd_updated_intervene, xd_updated_remain, x_pred_a, xd_pred_a, x_pred_na, xd_pred_na, y_pred_a, y_pred_na, pred_AR_a, pred_AR_na



def actor(base_kf_model, x_last_step, take_intervention, intervention_hidden_state, observation):
    ## Actor
    if take_intervention:
        # # Assign the drrift hidden states to the base ones
        # x_last_step['mu'][0] += intervention_hidden_state['mu'][0]
        # x_last_step['mu'][1] += intervention_hidden_state['mu'][1]
        # x_last_step['var'][0,0] += intervention_hidden_state['var'][0,0]
        # x_last_step['var'][1,1] += intervention_hidden_state['var'][1,1]
        # x_last_step['mu'][-1] = intervention_hidden_state['mu'][-1]
        # x_last_step['var'][-1,-1] = intervention_hidden_state['var'][-1,-1]

        x_last_step = copy.deepcopy(intervention_hidden_state)

    # Base model
    y_pred, x_pred, _ = base_kf_model.predict(x_last_step)
    x_updated, _, _ = base_kf_model.update(observation)

    return x_pred, y_pred, x_updated


    # LL_mu.append(x_pred['mu'][0])
    # LT_mu.append(x_pred['mu'][1])
    # AR_mu.append(x_pred['mu'][-1])
    # LL_var.append(x_pred['var'][0,0])
    # LT_var.append(x_pred['var'][1,1])
    # AR_var.append(x_pred['var'][-1,-1])
    # y_pred_mus.append(y_pred['mu'].item())
    # y_pred_vars.append(y_pred['var'].item())

    # LLd_mu.append(xd_pred['mu'][0])
    # LTd_mu.append(xd_pred['mu'][1])
    # ARd_mu.append(xd_pred['mu'][-1])
    # LLd_var.append(xd_pred['var'][0,0])
    # LTd_var.append(xd_pred['var'][1,1])
    # ARd_var.append(xd_pred['var'][-1,-1])
    # y_pred_mus_d.append(pred_AR['mu'])
    # y_pred_vars_d.append(pred_AR['var'])


    # # Plot prediction
    # plt.rcParams["figure.autolayout"] = True
    # fig = plt.figure(figsize=(12, 10))
    # gs = gridspec.GridSpec(8, 1)
    # ax0 = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1])
    # ax7 = plt.subplot(gs[2])
    # ax8 = plt.subplot(gs[3])
    # ax1 = plt.subplot(gs[4])
    # ax3 = plt.subplot(gs[5])
    # ax4 = plt.subplot(gs[6])
    # ax5 = plt.subplot(gs[7])


    # ############ Base model ############
    # ax0.plot(data_generator.time_series['timesteps'], y_pred_mus, label='Prediction')
    # ax0.fill_between(data_generator.time_series['timesteps'], np.array(y_pred_mus)-np.sqrt(y_pred_vars), np.array(y_pred_mus)+np.sqrt(y_pred_vars),color='gray', alpha=0.2)
    # ax0.plot(data_generator.time_series['timesteps'], LL_mu, 'k--')
    # ax0.fill_between(data_generator.time_series['timesteps'], np.array(LL_mu)-np.sqrt(LL_var), np.array(LL_mu)+np.sqrt(LL_var),color='gray', alpha=0.2)
    # ax0.plot(data_generator.time_series['timesteps'], data_generator.time_series['y'], label='True observation')
    # if anm_mag != 0:
    #     ax0.axvline(x=anm_pos, color='r', linestyle='--', label='Anomaly')
    # if len(trigger_pos) > 0:
    #     for trigger in trigger_pos:
    #         ax0.axvline(x=trigger, color='k', linestyle='--')
    # ax0.legend(loc='upper left', ncol = 2)
    # ax0.set_ylabel('obs.')
    # # Sum the rewards that include np.nan

    # ax0.set_title('Total reward {}'.format(np.round(np.nansum(prob_action_all), 2)))

    # # # Plot rewards
    # # ax6.plot(data_generator.time_series['timesteps'], likelihood_y_na, label='Reward')
    # # ax6.set_ylabel('L - y')
    # # # Use the same x-axis as ax0
    # # ax6.set_xlim(ax0.get_xlim())

    # # Plot HD LL
    # ax2.plot(data_generator.time_series['timesteps'], prob_action_all, label='Combined reward')
    # ax2.set_ylabel('Combined Likelihood')
    # ax2.set_xlim(ax0.get_xlim())
    # ax2.set_ylim([-0.05, 1.05])

    # # Plot likelihood
    # ax7.plot(data_generator.time_series['timesteps'], likelihood_y_a, label='action')
    # ax7.plot(data_generator.time_series['timesteps'], likelihood_y_na, label='no action')
    # ax7.set_ylabel('Likelihood y')
    # ax7.legend(loc='upper left', ncol = 2)

    # # Plot likelihood
    # ax8.plot(data_generator.time_series['timesteps'], likelihood_x_a, label='action')
    # ax8.plot(data_generator.time_series['timesteps'], likelihood_x_na, label='no action')
    # ax8.set_ylabel('Likelihood x')
    # ax8.legend(loc='upper left', ncol = 2)

    # # Plot LT
    # ax1.plot(data_generator.time_series['timesteps'], LT_mu, label='LT prediction')
    # ax1.fill_between(data_generator.time_series['timesteps'], np.array(LT_mu)-np.sqrt(LT_var), np.array(LT_mu)+np.sqrt(LT_var),color='gray', alpha=0.2)
    # ax1.set_ylabel('LT')

    # ############ Meta-AR model ############
    # ax3.plot(data_generator.time_series['timesteps'], y_pred_mus_d, label='Prediction')
    # ax3.fill_between(data_generator.time_series['timesteps'], np.array(y_pred_mus_d)-np.sqrt(y_pred_vars_d), np.array(y_pred_mus_d)+np.sqrt(y_pred_vars_d),color='gray', alpha=0.2)
    # ax3.plot(data_generator.time_series['timesteps'], LLd_mu, 'k--', label='LL prediction')
    # ax3.fill_between(data_generator.time_series['timesteps'], np.array(LLd_mu)-np.sqrt(LLd_var), np.array(LLd_mu)+np.sqrt(LLd_var),color='gray', alpha=0.2)
    # ax3.plot(data_generator.time_series['timesteps'], AR_mu, label='AR')
    # ax3.fill_between(data_generator.time_series['timesteps'], np.array(AR_mu)-np.sqrt(AR_var), np.array(AR_mu)+np.sqrt(AR_var),color='gray', alpha=0.2)
    # if anm_mag != 0:
    #     ax3.axvline(x=anm_pos, color='r', linestyle='--', label='Anomaly')
    # if len(trigger_pos) > 0:
    #     for trigger in trigger_pos:
    #         ax3.axvline(x=trigger, color='k', linestyle='--')
    # ax3.set_ylabel('LLd')

    # # Plot LT
    # ax4.plot(data_generator.time_series['timesteps'], LTd_mu, label='LT prediction')
    # ax4.fill_between(data_generator.time_series['timesteps'], np.array(LTd_mu)-np.sqrt(LTd_var), np.array(LTd_mu)+np.sqrt(LTd_var),color='gray', alpha=0.2)
    # ax4.set_ylabel('LTd')

    # # Plot AR
    # ax5.plot(data_generator.time_series['timesteps'], ARd_mu, label='AR prediction')
    # ax5.fill_between(data_generator.time_series['timesteps'], np.array(ARd_mu)-np.sqrt(ARd_var), np.array(ARd_mu)+np.sqrt(ARd_var),color='gray', alpha=0.2)
    # ax5.fill_between(data_generator.time_series['timesteps'], -np.sqrt(AR_stationary_var), np.sqrt(AR_stationary_var),color='red', alpha=0.2)
    # ax5.set_ylabel('ARd')

