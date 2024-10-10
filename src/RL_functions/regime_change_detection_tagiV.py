from pytagi.hybrid import LSTM_SSM
from pytagi import Normalizer
from tqdm import tqdm
import copy
import numpy as np
from pytagi.hybrid import process_input_ssm
import matplotlib.pyplot as plt
import csv
import pandas as pd

import random
from src.RL_functions.helpers import *
from pytagi.LSTM_KF_RL_Env import LSTM_KF_Env
from examples.data_loader import SyntheticTimeSeriesDataloader
from itertools import count
import matplotlib
from matplotlib import gridspec
from collections import namedtuple, deque
import math

from pytagi.nn import Linear, OutputUpdater, ReLU, Sequential, EvenExp

import time

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class TAGI_Net():
    def __init__(self, n_observations, n_actions):
        super(TAGI_Net, self).__init__()
        self.net = Sequential(
                    Linear(n_observations, 128),
                    ReLU(),
                    Linear(128, 128),
                    ReLU(),
                    Linear(128, n_actions * 2),
                    EvenExp()
                    )
        self.n_actions = n_actions
        self.n_observations = n_observations
    def forward(self, mu_x, var_x):
        return self.net.forward(mu_x, var_x)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class regime_change_detection_tagiV():
    def __init__(self, **kwargs):
        self.trained_BDLM = kwargs.get('trained_BDLM', None)
        self.val_datetime_values = kwargs.get('val_datetime_values', None)

        self.n_observations = kwargs.get('n_observations', None)
        self.n_actions = kwargs.get('n_actions', None)
        self.phi_AA = self.trained_BDLM.phi_AA
        self.Sigma_AA_ratio = self.trained_BDLM.Sigma_AA_ratio
        self.device = kwargs.get('device', 'cpu')

        self.LSTM_test_net = self.trained_BDLM.net_test
        self.init_mu_lstm = self.trained_BDLM.init_mu_lstm
        self.init_var_lstm = self.trained_BDLM.init_var_lstm
        self.init_z = self.trained_BDLM.init_z
        self.init_Sz = self.trained_BDLM.init_Sz
        # self.init_mu_W2b = self.trained_BDLM.init_mu_W2b
        # self.init_var_W2b = self.trained_BDLM.init_var_W2b
        self.last_seq_obs = self.trained_BDLM.last_seq_obs
        self.last_lstm_x = self.trained_BDLM.last_lstm_x

        self.train_xmean = self.trained_BDLM.train_dtl.x_mean
        self.train_xstd = self.trained_BDLM.train_dtl.x_std
        self.time_covariates = self.trained_BDLM.time_covariates
        self.input_seq_len = self.trained_BDLM.input_seq_len
        self.num_features = self.trained_BDLM.num_features
        self.output_col = self.trained_BDLM.output_col
        self.phi_AR = self.trained_BDLM.phi_AR
        self.Sigma_AR = self.trained_BDLM.Sigma_AR

        self.memory = ReplayMemory(10000)
        self.policy_net = TAGI_Net(self.n_observations, self.n_actions)
        self.target_net = TAGI_Net(self.n_observations, self.n_actions)
        self.optimal_net = TAGI_Net(self.n_observations, self.n_actions)
        policy_net_param = self.policy_net.net.get_state_dict()
        self.target_net.net.load_state_dict(policy_net_param)
        self.optimal_net.net.load_state_dict(policy_net_param)
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_f1t = []
        self.GAMMA = 0.99

    def generate_synthetic_ts(self, num_syn_ts, syn_ts_len, plot = True):
        self.syn_ts_all = []
        for j in tqdm(range(num_syn_ts)):
            # hybrid_gen = LSTM_SSM(
            #             neural_network = self.LSTM_test_net,           # LSTM
            #             baseline = 'AA + AR',
            #             z_init  = self.init_z,
            #             Sz_init = self.init_Sz,
            #             use_auto_AR = True,
            #             mu_W2b_init = self.init_mu_W2b,
            #             var_W2b_init = self.init_var_W2b,
            #             Sigma_AA_ratio = self.Sigma_AA_ratio,
            #             phi_AA = self.phi_AA,
            #         )
            hybrid_gen = LSTM_SSM(
                        neural_network = self.LSTM_test_net,           # LSTM
                        baseline = 'AA + AR_fixed',
                        z_init  = self.init_z,
                        Sz_init = self.init_Sz,
                        use_auto_AR = False,
                        # Sample phi_AR following a Gaussian with mean self.phi_AR and variance self.trained_BDLM.var_phi_AR
                        # phi_AR = np.random.normal(self.phi_AR, np.sqrt(self.trained_BDLM.var_phi_AR)),
                        # Sigma_AR = np.random.normal(self.Sigma_AR, np.sqrt(self.trained_BDLM.var_Sigma_AR)),
                        phi_AR = self.phi_AR,
                        Sigma_AR = self.Sigma_AR,
                        Sigma_AA_ratio = self.Sigma_AA_ratio,
                        phi_AA = self.phi_AA,
                    )

            syn_ts_i=copy.deepcopy(self.last_seq_obs)
            gen_datetime = np.array(self.val_datetime_values, dtype='datetime64')
            self.datetime_values_tosave = []
            for i in range(25):
                self.datetime_values_tosave.append(gen_datetime[-(25-i)])
            current_date_time = gen_datetime[-1] + np.timedelta64(7, 'D')
            self.datetime_values_tosave.append(current_date_time)

            x = copy.deepcopy(self.last_lstm_x)
            gen_mu_lstm = copy.deepcopy(self.init_mu_lstm)
            gen_var_lstm = copy.deepcopy(self.init_var_lstm)

            hybrid_gen.init_ssm_hs()
            hybrid_gen.z = copy.deepcopy(self.init_z)
            hybrid_gen.Sz = copy.deepcopy(self.init_Sz)

            for i in range(syn_ts_len):
                # remove the first two elements in x, and add two new at the end
                gen_datetime = np.append(gen_datetime, [gen_datetime[-1] + np.timedelta64(7, 'D')]).reshape(-1, 1)
                next_date = self._normalize_date(gen_datetime[-1], self.train_xmean[-1], self.train_xstd[-1], self.time_covariates)
                x[0:-2] = x[2:]
                x[-2] = gen_mu_lstm[-1].item()
                x[-1] = next_date.item()

                x_input = np.copy(x)
                mu_x_, var_x_ = process_input_ssm(
                        mu_x = x_input, mu_preds_lstm = gen_mu_lstm, var_preds_lstm = gen_var_lstm,
                        input_seq_len = self.input_seq_len, num_features = self.num_features,
                        )
                # Feed forward
                y_pred, Sy_red, z_prior, Sz_prior, m_pred, v_pred = hybrid_gen(mu_x_, var_x_)
                hybrid_gen.backward(mu_obs = np.nan, var_obs = np.nan, train_LSTM=False)

                # Sample
                Q_gen = copy.deepcopy(hybrid_gen.Q)
                # Q_gen[-2, -2] = hybrid_gen.mu_W2b_posterior
                # Q_gen[2, 2] = hybrid_gen.Sigma_AR * hybrid_gen.Sigma_AA_ratio
                Q_gen[-1, -1] = v_pred
                z_sample = z_prior.flatten() + np.random.multivariate_normal(0*z_prior.flatten(), Q_gen)
                y_sample = np.dot(hybrid_gen.F, z_sample)
                hybrid_gen.z = z_sample.reshape(-1, 1)

                obs_sample = Normalizer.unstandardize(
                    y_sample, self.train_xmean[self.output_col], self.train_xstd[self.output_col]
                )

                gen_mu_lstm.extend(m_pred)
                gen_var_lstm.extend(v_pred)
                syn_ts_i.extend(obs_sample)
                current_date_time = gen_datetime[-1] + np.timedelta64(7, 'D')
                self.datetime_values_tosave.append(current_date_time[0])

            self.syn_ts_all.append(syn_ts_i)

        if plot:
            COLORS = self._get_cmap(10)
            plt.figure(figsize=(20, 9))
            for i in range(10):
                plt.plot(self.syn_ts_all[i], color = COLORS(i))
            plt.show()

    def save_synthetic_ts(self, datetime_save_path=None, observation_save_path=None):
        with open(datetime_save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['date_time'])  # Write header
            for dt in self.datetime_values_tosave:
                writer.writerow([dt])  # Write formatted datetime string

        transposed_data = list(zip(*self.syn_ts_all))

        with open(observation_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(transposed_data)

    def load_synthetic_ts(self, datetime_save_path=None, observation_save_path=None):
        self.datetime_values = self._load_data_from_csv(datetime_save_path)
        self.syn_ts_all = self._load_data_from_csv(observation_save_path).T
        self.datetime_save_path = datetime_save_path
        self.observation_save_path = observation_save_path

    def plot_synthetic_ts(self):
        COLORS = self._get_cmap(10)
        plt.figure(figsize=(20, 9))
        for i in range(10):
            plt.plot(self.syn_ts_all[i], color = COLORS(i))
        plt.show()

    def train(self, num_episodes, step_look_back, abnormal_ts_percentage, anomaly_range,
              init_z, init_Sz, init_mu_preds_lstm, init_var_preds_lstm,
              batchsize, TAU, plot_samples=False, learning_curve_ylim = None,
              early_stopping = False, patience = 10, validation_episode_num = 0, early_stop_start = 0,
              agent_net_save_path='./saved_param/CASC_LGA007PIAP_E010_2024_07/tagi_models/agent_test', mean_R=None, std_R=None):
        self.batchsize = batchsize
        num_steps_per_episode = len(self.syn_ts_all[0])
        track_intervention_taken_times = np.zeros(num_episodes-validation_episode_num)
        optim_F1t = -1E8
        optim_episode = 0
        # Set seed for numpy random
        np.random.seed(0)
        validation_rand_samples = [np.random.random() for _ in range(validation_episode_num)]
        anm_positions_val = np.random.randint(step_look_back + self.trained_BDLM.input_seq_len, int(num_steps_per_episode/2), validation_episode_num)
        anm_magnitudes_val = np.random.uniform(anomaly_range[0], anomaly_range[1], validation_episode_num)
        print(validation_rand_samples)
        print(anm_positions_val)
        print(anm_magnitudes_val)

        np.random.seed(int(time.time() * 1000)% (2**32 - 1))
        # Estimatethe cost of intervention
        if mean_R is None or std_R is None:
            print('Estimating the R distribution...')
            mean_R, std_R = self._estimate_R_distribution(num_episodes, validation_episode_num, init_z, init_Sz, init_mu_preds_lstm, init_var_preds_lstm)
            self.mean_R = mean_R
            self.std_R = std_R
            self.cost_intervention = (self.mean_R + self.std_R)/(1-self.GAMMA) # 11.608093917361542 # self.mean_Q + self.std_Q
            print('The mean and std of R:', mean_R, std_R)
            print('=====================================')
        else:
            self.mean_R = mean_R
            self.std_R = std_R
            self.cost_intervention = (self.mean_R + self.std_R)/(1-self.GAMMA)

        for i_episode in range(num_episodes-validation_episode_num):
            anm_pos = np.random.randint(step_look_back + self.trained_BDLM.input_seq_len, int((num_steps_per_episode-step_look_back + self.trained_BDLM.input_seq_len)/2))

            sample = random.random()
            if sample < abnormal_ts_percentage:
                train_dtl = SyntheticTimeSeriesDataloader(
                    x_file=self.observation_save_path,
                    select_column=i_episode,
                    date_time_file=self.datetime_save_path,
                    add_anomaly = True,
                    anomaly_magnitude=[np.random.uniform(anomaly_range[0], anomaly_range[1]), np.random.uniform(anomaly_range[0], anomaly_range[1])],
                    anomaly_start=anm_pos,
                    x_mean=self.trained_BDLM.train_dtl.x_mean,
                    x_std=self.trained_BDLM.train_dtl.x_std,
                    output_col=self.trained_BDLM.output_col,
                    input_seq_len=self.trained_BDLM.input_seq_len,
                    output_seq_len=self.trained_BDLM.output_seq_len,
                    num_features=self.trained_BDLM.num_features,
                    stride=self.trained_BDLM.seq_stride,
                    time_covariates=self.trained_BDLM.time_covariates,
                )
                anomaly_injected = True
            else:
                train_dtl = SyntheticTimeSeriesDataloader(
                    x_file=self.observation_save_path,
                    select_column = i_episode,
                    date_time_file=self.datetime_save_path,
                    x_mean=self.trained_BDLM.train_dtl.x_mean,
                    x_std=self.trained_BDLM.train_dtl.x_std,
                    output_col=self.trained_BDLM.output_col,
                    input_seq_len=self.trained_BDLM.input_seq_len,
                    output_seq_len=self.trained_BDLM.output_seq_len,
                    num_features=self.trained_BDLM.num_features,
                    stride=self.trained_BDLM.seq_stride,
                    time_covariates=self.trained_BDLM.time_covariates,
                )
                anomaly_injected = False

            env = LSTM_KF_Env(render_mode=None, data_loader=train_dtl, step_look_back=step_look_back)

            state, info = env.reset(z=init_z, Sz=init_Sz, mu_preds_lstm = copy.deepcopy(init_mu_preds_lstm), var_preds_lstm = copy.deepcopy(init_var_preds_lstm),
                        net_test = self.LSTM_test_net, init_mu_W2b = None, init_var_W2b = None, phi_AR = self.phi_AR, Sigma_AR = self.Sigma_AR,
                        phi_AA = self.phi_AA, Sigma_AA_ratio = self.Sigma_AA_ratio)
            state = state['hidden_states']
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Nomalize the state
            # Compute the stationary standard deviation for AR
            AR_std_stationary = np.sqrt(self.trained_BDLM.Sigma_AR/(1 - self.trained_BDLM.phi_AR**2))
            LA_var_stationary = self.trained_BDLM.Sigma_AA_ratio *  self.trained_BDLM.Sigma_AR/(1 - self.trained_BDLM.phi_AA**2)
            if step_look_back == 64:
                seg_len = 8
            state = normalize_tensor_two_parts(state, 0, np.sqrt(LA_var_stationary), \
                                            0, AR_std_stationary, seg_len)

            total_reward_one_episode = 0
            dummy_steps = 0
            Q_values_all = []
            Q_var_all = []
            Q_var_epstic_all = []
            for t in count():
                action = self._select_action(state)

                observation, reward, terminated, truncated, info = env.step(action.item(), cost_intervention=self.cost_intervention)

                Q_values_t, var_Q, epist_var_Q, alea_var_Q = self._track_Qvalues(state)

                Q_values_t = Q_values_t[0].tolist()
                var_Q = var_Q[0].tolist()
                epist_var_Q = epist_var_Q[0].tolist()
                Q_values_all.append(Q_values_t)
                Q_var_all.append(var_Q)
                Q_var_epstic_all.append(epist_var_Q)

                dummy_steps += 1

                if action.item() == 1:
                    track_intervention_taken_times[i_episode] += 1
                    intervention_taken = True

                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                total_reward_one_episode += reward

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation['hidden_states'],\
                            dtype=torch.float32, device=self.device).unsqueeze(0)
                    next_state = normalize_tensor_two_parts(next_state, 0, np.sqrt(LA_var_stationary),\
                                                            0, AR_std_stationary, seg_len)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                self._infer_model()

                target_net_state_dict = self.target_net.net.get_state_dict()
                policy_net_state_dict = self.policy_net.net.get_state_dict()
                for key in policy_net_state_dict:
                    for key2 in policy_net_state_dict[key]:
                        target_net_state_dict[key][key2] = (np.asarray(policy_net_state_dict[key][key2])*TAU +
                                                            np.asarray(target_net_state_dict[key][key2])*(1-TAU)).tolist()
                self.target_net.net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_rewards.append(total_reward_one_episode)
                    if early_stopping is not True:
                        self._plot_rewards(metric = self.episode_rewards, ylim=learning_curve_ylim)
                    else:
                        self._plot_rewards(metric = self.episode_f1t, ylim=[-0.1, 1.1])
                    break

            # Fill 65 rows of [nan, nan] values in front of Q_values_all
            Q_values_all = [[np.nan, np.nan]] * 65 + Q_values_all
            Q_values_all = np.array(Q_values_all).T
            Q_var_all = [[np.nan, np.nan]] * 65 + Q_var_all
            Q_var_all = np.array(Q_var_all).T
            Q_var_epstic_all = [[np.nan, np.nan]] * 65 + Q_var_epstic_all
            Q_var_epstic_all = np.array(Q_var_epstic_all).T

            print(track_intervention_taken_times)
            print(self.episode_f1t)

            if plot_samples:
                timesteps = np.arange(0, len(info['measurement_one_episode']), 1)
                mu_hidden_states_one_episode = np.array(info['hidden_state_one_episode']['mu'])
                var_hidden_states_one_episode = np.array(info['hidden_state_one_episode']['var'])
                mu_prediction_one_episode = np.array(info['prediction_one_episode']['mu']).flatten()
                var_prediction_one_episode = np.array(info['prediction_one_episode']['var']).flatten()
                # if track_intervention_taken_times[i_episode] == 1:
                if True:
                    # Plot prediction
                    fig = plt.figure(figsize=(15, 12))
                    gs = gridspec.GridSpec(5, 1)
                    ax0 = plt.subplot(gs[0])
                    ax1 = plt.subplot(gs[2])
                    ax2 = plt.subplot(gs[3])
                    ax3 = plt.subplot(gs[4])
                    ax4 = plt.subplot(gs[1])

                    ax0.plot(timesteps, info['measurement_one_episode'], label='True')
                    # plot the standard deviation of the prediction
                    ax0.plot(timesteps, mu_prediction_one_episode, label='Predicted')
                    ax0.fill_between(timesteps, mu_prediction_one_episode - np.sqrt(var_prediction_one_episode),\
                                        mu_prediction_one_episode + np.sqrt(var_prediction_one_episode), color='gray', alpha=0.2)
                    ax0.set_title(f'Predicted vs True, epoch {i_episode+1}')
                    ax0.set_ylabel('y')
                    delta_max_min = np.max(mu_prediction_one_episode) - np.min(mu_prediction_one_episode)
                    ax0.set_ylim(np.min(mu_prediction_one_episode)-0.05*delta_max_min, np.max(mu_prediction_one_episode)+0.05*delta_max_min)
                    if anomaly_injected:
                        anomaly_pos = timesteps[anm_pos - self.trained_BDLM.input_seq_len]
                        anomaly_pos2 = timesteps[anm_pos+int((len(timesteps)-65)/2)-self.trained_BDLM.input_seq_len]
                        ax0.axvline(x=anomaly_pos, color='gray', linestyle='--')
                        ax0.axvline(x=anomaly_pos2, color='gray', linestyle='--')
                    ax0.legend()

                    ax1.plot(timesteps, mu_hidden_states_one_episode[:,2], label='LA')
                    ax1.fill_between(timesteps, mu_hidden_states_one_episode[:,2] - np.sqrt(var_hidden_states_one_episode[:,2,2]),\
                                        mu_hidden_states_one_episode[:,2] + np.sqrt(var_hidden_states_one_episode[:,2,2]), color='gray', alpha=0.2)
                    ax1.set_ylabel('LA')

                    ax2.plot(timesteps, mu_hidden_states_one_episode[:,-1], label='PD')
                    ax2.fill_between(timesteps, mu_hidden_states_one_episode[:,-1] - np.sqrt(var_hidden_states_one_episode[:,-1,-1]),\
                                        mu_hidden_states_one_episode[:,-1] + np.sqrt(var_hidden_states_one_episode[:,-1,-1]), color='gray', alpha=0.2)
                    ax2.set_ylabel('PD')

                    ax3.fill_between(timesteps, np.zeros_like(timesteps)-3*AR_std_stationary, np.zeros_like(timesteps)+3*AR_std_stationary, color='red', alpha=0.1)
                    ax3.plot(timesteps, mu_hidden_states_one_episode[:,-2], label='AR')
                    ax3.fill_between(timesteps, mu_hidden_states_one_episode[:,-2] - np.sqrt(var_hidden_states_one_episode[:,-2,-2]),\
                                        mu_hidden_states_one_episode[:,-2] + np.sqrt(var_hidden_states_one_episode[:,-2,-2]), color='gray', alpha=0.2)
                    ax3.set_ylabel('AR')

                    ax4.plot(Q_values_all[0], label='Q_value_0')
                    ax4.plot(Q_values_all[1], label='Q_value_1')
                    ax4.fill_between(timesteps, Q_values_all[0] - np.sqrt(Q_var_all[0]),\
                                        Q_values_all[0] + np.sqrt(Q_var_all[0]), color='gray', alpha=0.2)
                    ax4.fill_between(timesteps, Q_values_all[1] - np.sqrt(Q_var_all[1]),\
                                        Q_values_all[1] + np.sqrt(Q_var_all[1]), color='gray', alpha=0.2)
                    ax4.fill_between(timesteps, Q_values_all[0] - np.sqrt(Q_var_epstic_all[0]),\
                                        Q_values_all[0] + np.sqrt(Q_var_epstic_all[0]), color='green', alpha=0.2)
                    ax4.fill_between(timesteps, Q_values_all[1] - np.sqrt(Q_var_epstic_all[1]),\
                                        Q_values_all[1] + np.sqrt(Q_var_epstic_all[1]), color='green', alpha=0.2)
                    ax4.set_ylabel('Q_values')
                    ax4.set_xlim(ax0.get_xlim())
                    ax4.legend(loc='upper left')
                    plt.show()
                    # filename = f'saved_results/Qvalues_training/Qvalues_episode#{i_episode}.png'
                    # plt.savefig(filename)
                    # plt.close()

            # Early stopping
            if early_stopping:
                if i_episode >= early_stop_start:
                    FP = 0
                    FN = 0
                    TP = 0
                    lambdas_all = []
                    for i_val_episode in range(validation_episode_num):
                        anm_pos = anm_positions_val[i_val_episode]

                        sample = validation_rand_samples[i_val_episode]
                        if sample < 0.5:
                            train_dtl = SyntheticTimeSeriesDataloader(
                                x_file=self.observation_save_path,
                                select_column = num_episodes-validation_episode_num+i_val_episode,
                                date_time_file=self.datetime_save_path,
                                add_anomaly = True,
                                anomaly_magnitude = anm_magnitudes_val[i_val_episode],
                                anomaly_start=anm_pos,
                                x_mean=self.trained_BDLM.train_dtl.x_mean,
                                x_std=self.trained_BDLM.train_dtl.x_std,
                                output_col=self.trained_BDLM.output_col,
                                input_seq_len=self.trained_BDLM.input_seq_len,
                                output_seq_len=self.trained_BDLM.output_seq_len,
                                num_features=self.trained_BDLM.num_features,
                                stride=self.trained_BDLM.seq_stride,
                                time_covariates=self.trained_BDLM.time_covariates,
                            )
                            anomaly_injected = True
                        else:
                            train_dtl = SyntheticTimeSeriesDataloader(
                                x_file=self.observation_save_path,
                                select_column = num_episodes-validation_episode_num+i_val_episode,
                                date_time_file=self.datetime_save_path,
                                x_mean=self.trained_BDLM.train_dtl.x_mean,
                                x_std=self.trained_BDLM.train_dtl.x_std,
                                output_col=self.trained_BDLM.output_col,
                                input_seq_len=self.trained_BDLM.input_seq_len,
                                output_seq_len=self.trained_BDLM.output_seq_len,
                                num_features=self.trained_BDLM.num_features,
                                stride=self.trained_BDLM.seq_stride,
                                time_covariates=self.trained_BDLM.time_covariates,
                            )
                            anomaly_injected = False

                        env = LSTM_KF_Env(render_mode=None, data_loader=train_dtl, step_look_back=step_look_back)
                        state, info = env.reset(z=init_z, Sz=init_Sz, mu_preds_lstm = copy.deepcopy(init_mu_preds_lstm), var_preds_lstm = copy.deepcopy(init_var_preds_lstm),
                                    net_test = self.LSTM_test_net, init_mu_W2b = None, init_var_W2b = None, phi_AR = self.phi_AR, Sigma_AR = self.Sigma_AR,
                                    phi_AA = self.phi_AA, Sigma_AA_ratio = self.Sigma_AA_ratio)

                        # Nomalize the state
                        AR_std_stationary = np.sqrt(self.trained_BDLM.Sigma_AR/(1 - self.trained_BDLM.phi_AR**2))
                        LA_var_stationary = self.trained_BDLM.Sigma_AA_ratio *  self.trained_BDLM.Sigma_AR/(1 - self.trained_BDLM.phi_AA**2)

                        # Run agent on the i_val_episode time series
                        val_ep_steps = 0
                        for t in count():
                            state = torch.tensor(state['hidden_states'],\
                                dtype=torch.float32, device='cpu').unsqueeze(0)
                            state = normalize_tensor_two_parts(state, 0, np.sqrt(LA_var_stationary),\
                                                                0, AR_std_stationary, seg_len)

                            action = self._select_action(state, greedy=True)
                            state, _, terminated, truncated, info = env.step(action.item(), cost_intervention=self.cost_intervention)

                            done = terminated or truncated

                            if action.item() == 1:
                                if anomaly_injected:
                                    # Alarm is triggered
                                    trigger_pos = val_ep_steps + step_look_back + self.trained_BDLM.input_seq_len
                                    if trigger_pos >= anm_pos:
                                        # True alarm
                                        TP += 1
                                        lambda_i = 1 - (trigger_pos - anm_pos) / (num_steps_per_episode - anm_pos)
                                    else:
                                        # False alarm
                                        FP += 1
                                        lambda_i = 1
                                    lambdas_all.append(lambda_i)
                                else:
                                    FP += 1
                                done = True

                            if terminated:
                                # When comes to here, alarm is never triggered and the time series ends
                                if anomaly_injected:
                                    FN += 1
                                    lambda_i = 0
                                    lambdas_all.append(lambda_i)
                                state = None

                            val_ep_steps += 1

                            if done:
                                print('For validation episode:', i_val_episode)
                                print('Anomaly is injected:', anomaly_injected)
                                print(f'TP = {TP}, FP = {FP}, FN = {FN}, lambda = {lambdas_all}')
                                break

                    current_F1t = np.mean(lambdas_all)*2*TP/(2*TP+FP+FN)
                    self.episode_f1t.append(current_F1t)

                    self._test_real_data(i_episode, init_z, init_Sz, init_mu_preds_lstm, init_var_preds_lstm, current_F1t)

                    if current_F1t > optim_F1t:
                        print('------------------------------------')
                        print(f'Optimal F1t: {optim_F1t}, episode: {optim_episode}')
                        print(f'Current F1t: {current_F1t}, episode: {i_episode}')
                        print('****** Update the optimal model ******')
                        print('------------------------------------')
                        optim_F1t = current_F1t
                        optim_episode = i_episode
                        net_param_temp = self.policy_net.net.get_state_dict()
                        self.optimal_net.net.load_state_dict(net_param_temp)
                    else:
                        print('------------------------------------')
                        print(f'Optimal F1t: {optim_F1t}, episode: {optim_episode}')
                        print(f'Current F1t: {current_F1t}, episode: {i_episode}')
                        print('Optimal model remains')
                        print('------------------------------------')
                        if i_episode - optim_episode > patience:
                            print('Early stopping is triggered, training stopped, model saved at epoch:', optim_episode)
                            net_param_temp = self.optimal_net.net.get_state_dict()
                            self.policy_net.net.load_state_dict(net_param_temp)
                            # Save policy net
                            self.policy_net.net.save_csv(agent_net_save_path)
                            break
                    if i_episode == num_episodes-validation_episode_num-1:
                        print('Finished training in all episodes.')
                        # Save policy net
                        self.policy_net.net.save_csv(agent_net_save_path)


        print('Complete')
        if early_stopping is not True:
            self._plot_rewards(metric = self.episode_rewards, show_result=True, ylim=learning_curve_ylim)
        else:
            self._plot_rewards(metric = self.episode_f1t, show_result=True, ylim=[-0.1, 1.1])
        # plot a bar chart of the number of interventions taken
        plt.figure(2)
        plt.bar(np.arange(num_episodes-validation_episode_num)+1, track_intervention_taken_times)
        plt.title('Number of interventions taken')
        plt.xlabel('Episode')
        plt.ylabel('Number of interventions taken')
        plt.show()
        plt.ioff()

    def _test_real_data(self, i_episode, init_z, init_Sz, init_mu_preds_lstm, init_var_preds_lstm, validation_F1t):
        from itertools import count

        step_look_back = 64
        env = LSTM_KF_Env(render_mode=None, data_loader=self.trained_BDLM.test_dtl, \
                            ts_model=self.trained_BDLM.model, step_look_back=step_look_back)

        state, _ = env.reset(z=init_z, Sz=init_Sz, mu_preds_lstm = copy.deepcopy(init_mu_preds_lstm), var_preds_lstm = copy.deepcopy(init_var_preds_lstm),
                                net_test = self.LSTM_test_net, init_mu_W2b = None, init_var_W2b = None, phi_AR=self.trained_BDLM.phi_AR, Sigma_AR=self.trained_BDLM.Sigma_AR,
                                phi_AA = self.trained_BDLM.phi_AA, Sigma_AA_ratio = self.trained_BDLM.Sigma_AA_ratio)

        intervention_index =[]
        LA_var_stationary = self.trained_BDLM.Sigma_AA_ratio *  self.trained_BDLM.Sigma_AR/(1 - self.trained_BDLM.phi_AA**2)
        AR_std_stationary = np.sqrt(self.trained_BDLM.Sigma_AR/(1 - self.trained_BDLM.phi_AR**2))
        if step_look_back == 64:
            seg_len = 8
        RL_step_taken = 0
        Q_values_all = []
        Q_var_all = []
        Q_var_epstic_all = []
        for t in count():
            state = torch.tensor(state['hidden_states'],\
                                dtype=torch.float32, device='cpu').unsqueeze(0)
            state = normalize_tensor_two_parts(state, 0, np.sqrt(LA_var_stationary),\
                                                0, AR_std_stationary, seg_len)

            # Select action
            action = self._select_action(state, greedy=True)
            # Track Q values
            Q_values_t, Q_var_t, epist_var_Q, alea_var_Q  = self._track_Qvalues(state)
            Q_values_t = Q_values_t[0].tolist()
            Q_var_t = Q_var_t[0].tolist()
            epist_var_Q = epist_var_Q[0].tolist()
            Q_values_all.append(Q_values_t)
            Q_var_all.append(Q_var_t)
            Q_var_epstic_all.append(epist_var_Q)

            state, _, terminated, truncated, info = env.step(action.item(), cost_intervention=self.cost_intervention)

            RL_step_taken += 1

            if action.item() == 1:
                intervention_index.append(t + step_look_back + 1)

            done = terminated or truncated
            if done:
                break

        # Fill 65 rows of [nan, nan] values in front of Q_values_all
        Q_values_all = [[np.nan, np.nan]] * 65 + Q_values_all
        Q_values_all = np.array(Q_values_all).T
        Q_var_all = [[np.nan, np.nan]] * 65 + Q_var_all
        Q_var_all = np.array(Q_var_all).T
        Q_var_epstic_all = [[np.nan, np.nan]] * 65 + Q_var_epstic_all
        Q_var_epstic_all = np.array(Q_var_epstic_all).T

        # Plot prediction
        timesteps = np.arange(0, len(info['measurement_one_episode']), 1)
        mu_hidden_states_one_episode = np.array(info['hidden_state_one_episode']['mu'])
        var_hidden_states_one_episode = np.array(info['hidden_state_one_episode']['var'])
        mu_prediction_one_episode = np.array(info['prediction_one_episode']['mu']).flatten()
        var_prediction_one_episode = np.array(info['prediction_one_episode']['var']).flatten()
        from matplotlib import gridspec
        fig = plt.figure(figsize=(15, 14))
        gs = gridspec.GridSpec(6, 1)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[2])
        ax2 = plt.subplot(gs[3])
        ax3 = plt.subplot(gs[4])
        ax4 = plt.subplot(gs[5])
        ax5 = plt.subplot(gs[1])

        ax0.plot(timesteps, info['measurement_one_episode'], label='True')
        ax0.plot(timesteps, mu_prediction_one_episode , label='Predicted')
        for count, i in enumerate(intervention_index):
            if count == 0:
                ax0.axvline(x=timesteps[i], color='r', linestyle='--', label='RL triggers')
            else:
                ax0.axvline(x=timesteps[i], color='r', linestyle='--')
        ax0.plot(timesteps, mu_hidden_states_one_episode[:,0], color = 'k',alpha=0.4, label='LL')
        # ax0.fill_between(timesteps, mu_hidden_states_one_episode[:,0] - np.sqrt(var_hidden_states_one_episode[:,0,0]),\
        #                     mu_hidden_states_one_episode[:,0] + np.sqrt(var_hidden_states_one_episode[:,0,0]), color='gray', alpha=0.2)
        ax0.set_ylabel('y')
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.07,1))


        ax1.plot(timesteps, mu_hidden_states_one_episode[:,2], label='LA')
        ax1.fill_between(timesteps, mu_hidden_states_one_episode[:,2] - np.sqrt(var_hidden_states_one_episode[:,2,2]),\
                            mu_hidden_states_one_episode[:,2] + np.sqrt(var_hidden_states_one_episode[:,2,2]), color='gray', alpha=0.2)
        ax1.set_ylabel('LA')

        ax2.fill_between(timesteps, np.zeros_like(timesteps)-3*AR_std_stationary, np.zeros_like(timesteps)+3*AR_std_stationary, color='red', alpha=0.1)
        ax2.plot(timesteps, mu_hidden_states_one_episode[:,-2], label='AR')
        ax2.fill_between(timesteps, mu_hidden_states_one_episode[:,-2] - np.sqrt(var_hidden_states_one_episode[:,-2,-2]),\
                            mu_hidden_states_one_episode[:,-2] + np.sqrt(var_hidden_states_one_episode[:,-2,-2]), color='gray', alpha=0.2)
        ax2.set_ylabel('AR')
        # ax2.set_ylim(-1.1, 1.1)

        ax3.plot(timesteps, mu_hidden_states_one_episode[:,1], label='LT')
        ax3.fill_between(timesteps, mu_hidden_states_one_episode[:,1] - np.sqrt(var_hidden_states_one_episode[:,1,1]),\
                            mu_hidden_states_one_episode[:,1] + np.sqrt(var_hidden_states_one_episode[:,1,1]), color='gray', alpha=0.2)
        ax3.set_ylabel('LT')
        ax3.set_ylim(-0.1,0.1)

        ax4.plot(timesteps, mu_hidden_states_one_episode[:,-1], label='LSTM')
        ax4.fill_between(timesteps, mu_hidden_states_one_episode[:,-1] - np.sqrt(var_hidden_states_one_episode[:,-1,-1]),\
                            mu_hidden_states_one_episode[:,-1] + np.sqrt(var_hidden_states_one_episode[:,-1,-1]), color='gray', alpha=0.2)
        ax4.set_ylabel('LSTM')
        ax4.set_xlabel('Time steps')

        ax5.plot(timesteps, Q_values_all[0], label='Q(0)')
        ax5.plot(timesteps, Q_values_all[1], label='Q(1)')
        ax5.fill_between(timesteps, Q_values_all[0] - np.sqrt(Q_var_all[0]),\
                            Q_values_all[0] + np.sqrt(Q_var_all[0]), color='gray', alpha=0.2)
        ax5.fill_between(timesteps, Q_values_all[1] - np.sqrt(Q_var_all[1]),\
                            Q_values_all[1] + np.sqrt(Q_var_all[1]), color='gray', alpha=0.2)
        ax5.fill_between(timesteps, Q_values_all[0] - np.sqrt(Q_var_epstic_all[0]),\
                            Q_values_all[0] + np.sqrt(Q_var_epstic_all[0]), color='green', alpha=0.2)
        ax5.fill_between(timesteps, Q_values_all[1] - np.sqrt(Q_var_epstic_all[1]),\
                            Q_values_all[1] + np.sqrt(Q_var_epstic_all[1]), color='green', alpha=0.2)
        ax5.set_ylabel('Q values')
        # set x axis to be the same as ax0
        ax5.set_xlim(ax0.get_xlim())
        ax5.legend(loc='upper center', bbox_to_anchor=(1.07,1))

        # Print the F1t score in the title
        ax0.set_title(f'Epoch {i_episode+1}, F1t = {validation_F1t:.5f}')

        # save figure to local
        filename = f'saved_results/CASC_LGA007PIAP_E010_2024_07/update_vs_real_performance/episode#{i_episode}.png'
        plt.savefig(filename)

    def _estimate_Q_distribution(self, num_episodes, validation_episode_num, init_z, init_Sz, init_mu_preds_lstm, init_var_preds_lstm):
        Q_estimates = []
        for i_episode in tqdm(range(num_episodes-validation_episode_num)):
            from itertools import count

            train_dtl = SyntheticTimeSeriesDataloader(
                        x_file=self.observation_save_path,
                        select_column=i_episode,
                        date_time_file=self.datetime_save_path,
                        add_anomaly = False,
                        x_mean=self.trained_BDLM.train_dtl.x_mean,
                        x_std=self.trained_BDLM.train_dtl.x_std,
                        output_col=self.trained_BDLM.output_col,
                        input_seq_len=self.trained_BDLM.input_seq_len,
                        output_seq_len=self.trained_BDLM.output_seq_len,
                        num_features=self.trained_BDLM.num_features,
                        stride=self.trained_BDLM.seq_stride,
                        time_covariates=self.trained_BDLM.time_covariates,
                    )
            step_look_back = 64
            env = LSTM_KF_Env(render_mode=None, data_loader=train_dtl, step_look_back=step_look_back)
            env.reset(z=init_z, Sz=init_Sz, mu_preds_lstm = copy.deepcopy(init_mu_preds_lstm), var_preds_lstm = copy.deepcopy(init_var_preds_lstm),
                        net_test = self.LSTM_test_net, init_mu_W2b = None, init_var_W2b = None, phi_AR = self.phi_AR, Sigma_AR = self.Sigma_AR,
                        phi_AA = self.phi_AA, Sigma_AA_ratio = self.Sigma_AA_ratio)
            Q_estimate = 0
            for t in count():
                _, reward, terminated, truncated, _ = env.step(0)
                Q_estimate += self.GAMMA**(t) * reward
                done = terminated or truncated
                if done:
                    break

            Q_estimates.append(Q_estimate)

        # Compute the mean and std of Q_estimates
        mean_Q = np.mean(np.array(Q_estimates))
        std_Q = np.std(np.array(Q_estimates))

        return mean_Q, std_Q

    def _estimate_R_distribution(self, num_episodes, validation_episode_num, init_z, init_Sz, init_mu_preds_lstm, init_var_preds_lstm):
        R_estimates = []
        for i_episode in tqdm(range(num_episodes-validation_episode_num)):
            from itertools import count

            train_dtl = SyntheticTimeSeriesDataloader(
                        x_file=self.observation_save_path,
                        select_column=i_episode,
                        date_time_file=self.datetime_save_path,
                        add_anomaly = False,
                        x_mean=self.trained_BDLM.train_dtl.x_mean,
                        x_std=self.trained_BDLM.train_dtl.x_std,
                        output_col=self.trained_BDLM.output_col,
                        input_seq_len=self.trained_BDLM.input_seq_len,
                        output_seq_len=self.trained_BDLM.output_seq_len,
                        num_features=self.trained_BDLM.num_features,
                        stride=self.trained_BDLM.seq_stride,
                        time_covariates=self.trained_BDLM.time_covariates,
                    )
            step_look_back = 64
            env = LSTM_KF_Env(render_mode=None, data_loader=train_dtl, step_look_back=step_look_back)
            env.reset(z=init_z, Sz=init_Sz, mu_preds_lstm = copy.deepcopy(init_mu_preds_lstm), var_preds_lstm = copy.deepcopy(init_var_preds_lstm),
                        net_test = self.LSTM_test_net, init_mu_W2b = None, init_var_W2b = None, phi_AR = self.phi_AR, Sigma_AR = self.Sigma_AR,
                        phi_AA = self.phi_AA, Sigma_AA_ratio = self.Sigma_AA_ratio)
            for t in count():
                _, reward, terminated, truncated, _ = env.step(0)
                R_estimates.append(reward)
                done = terminated or truncated
                if done:
                    break

        # Compute the mean and std of Q_estimates
        mean_R = np.mean(np.array(R_estimates))
        std_R = np.std(np.array(R_estimates))

        return mean_R, std_R

    def _normalize_date(self, date_time_i, mean, std, time_covariates):
        for time_cov in time_covariates:
            if time_cov == 'hour_of_day':
                hour_of_day = date_time_i.astype('datetime64[h]').astype(int) % 24
                output = hour_of_day
            elif time_cov == 'day_of_week':
                day_of_week = date_time_i.astype('datetime64[D]').astype(int) % 7
                output = day_of_week
            elif time_cov == 'week_of_year':
                week_of_year = date_time_i.astype('datetime64[W]').astype(int) % 52 + 1
                output = week_of_year
            elif time_cov == 'month_of_year':
                month_of_year = date_time_i.astype('datetime64[M]').astype(int) % 12 + 1
                output = month_of_year
            elif time_cov == 'quarter_of_year':
                month_of_year = date_time_i.astype('datetime64[M]').astype(int) % 12 + 1
                quarter_of_year = (month_of_year - 1) // 3 + 1
                output = quarter_of_year
            elif time_cov == 'day_of_year':
                day_of_year = date_time_i.astype('datetime64[D]').astype(int) % 365
                output = day_of_year

        output = Normalizer.standardize(data=output, mu=mean, std=std)
        return output

    def _get_cmap(self, n, name='rainbow'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    def _load_data_from_csv(self, data_file: str, skip_row = 0, select_column = None) -> pd.DataFrame:
        """Load data from csv file"""

        data = pd.read_csv(data_file, skiprows=skip_row, delimiter=",", header=None)

        # if select_column is not given by the user, reshape the data to (-1, 1)
        if select_column is None:
            return data.values
        else:
            # if select_column is given by the user, return the selected column
            return data.values[:, select_column].reshape(-1, 1)

    def _infer_model(self):
        if len(self.memory) < self.batchsize:
            return

        self.policy_net.net.train()
        self.target_net.net.eval()
        transitions = self.memory.sample(self.batchsize)
        batch = Transition(*zip(*transitions))

        final_mask = torch.tensor(tuple(map(lambda s: s is None,
                                batch.next_state)), device=self.device, dtype=torch.bool)

        state_mean_batch = torch.cat(batch.state).numpy()
        action_batch = torch.cat(batch.action).numpy()
        reward_batch = torch.cat(batch.reward).numpy()
        next_state_mean_batch = torch.cat([s if s is not None
                                           else torch.tensor([np.ones(self.policy_net.n_observations, dtype=np.float32)])
                                           for s in batch.next_state]).numpy()

        # Add uncertainty to the state
        state_batch = {'mu': state_mean_batch, 'var': np.zeros_like(state_mean_batch)}
        next_state_batch = {'mu': next_state_mean_batch, 'var': np.zeros_like(next_state_mean_batch)}

        # Get the next state values from target net
        next_state_values_mu_f, next_state_values_var_f = self.target_net.net(next_state_batch['mu'])

        # Reshape to 2D
        next_state_values_mu_f = next_state_values_mu_f.reshape(self.batchsize, self.target_net.n_actions*2)
        next_state_values_var_f = next_state_values_var_f.reshape(self.batchsize, self.target_net.n_actions*2)

        # Along the first axis, select the first and the third columns of the 2D array next_state_values_mu
        next_state_values_mu = next_state_values_mu_f[:, [0, 2]]
        next_state_values_var = next_state_values_var_f[:, [0, 2]] + next_state_values_mu_f[:, [1, 3]]

        # Keep the maximum next state value according to the samples
        max_indices = np.argmax(next_state_values_mu, axis=1)
        next_state_values_mu = next_state_values_mu[np.arange(self.batchsize), max_indices]
        next_state_values_var = next_state_values_var[np.arange(self.batchsize), max_indices]

        # Set the next state values of final states to 0 if next state is final
        next_state_values_mu_tensor = torch.tensor(next_state_values_mu, device=self.device)
        next_state_values_var_tensor = torch.tensor(next_state_values_var, device=self.device)
        next_state_values_mu_tensor[final_mask] = 0.0
        next_state_values_var_tensor[final_mask] = 1e-4
        next_state_values_mu = next_state_values_mu_tensor.numpy()
        next_state_values_var = next_state_values_var_tensor.numpy()

        # Compute the expected Q values
        # Scale the reward so that the Q value is bounded between 0 and 1
        reward_batch = reward_batch * (1-self.GAMMA)/np.abs(self.mean_R + self.std_R)
        expected_state_values_mu = np.array((next_state_values_mu * self.GAMMA) + reward_batch)
        expected_state_values_var = np.array((next_state_values_var * self.GAMMA**2))

        # Infer the policy network using the expected Q values
        expected_state_action_values_mu_f, expected_state_action_values_var_f = self.policy_net.net(state_batch['mu'])

        # # Only change the expected Q values where actions are taken
        expected_state_action_values_mu_f = expected_state_action_values_mu_f.reshape(self.batchsize, self.policy_net.n_actions*2)
        expected_state_action_values_var_f = expected_state_action_values_var_f.reshape(self.batchsize, self.policy_net.n_actions*2)
        expected_state_action_values_mu = expected_state_action_values_mu_f[:, [0,2]]
        expected_state_action_values_var = expected_state_action_values_var_f[:, [0,2]] + expected_state_action_values_mu_f[:, [1,3]]
        expected_state_action_values_mu[np.arange(self.batchsize), action_batch.flatten()] = expected_state_values_mu
        expected_state_action_values_var[np.arange(self.batchsize), action_batch.flatten()] = expected_state_values_var
        # expected_state_action_values_mu[np.arange(self.batchsize), 1-action_batch.flatten()] = np.nan
        # expected_state_action_values_var[np.arange(self.batchsize), 1-action_batch.flatten()] = 1e8
        expected_state_action_values_mu = expected_state_action_values_mu.flatten()
        expected_state_action_values_var = expected_state_action_values_var.flatten()

        # Update output layer
        out_updater = OutputUpdater(self.policy_net.net.device)
        out_updater.update_heteros(
            output_states = self.policy_net.net.output_z_buffer,
            mu_obs = expected_state_action_values_mu,
            var_obs = expected_state_action_values_var,
            delta_states = self.policy_net.net.input_delta_z_buffer,
        )

        # Feed backward
        self.policy_net.net.backward()
        self.policy_net.net.step()

        # For numerical stability: clip the variance of the parameters to 1e-8
        policy_net_param_temp = self.policy_net.net.get_state_dict()
        for key in policy_net_param_temp:
            policy_net_param_temp[key]['var_w']=np.clip(policy_net_param_temp[key]['var_w'], 1e-8, None).tolist()
            policy_net_param_temp[key]['var_b']=np.clip(policy_net_param_temp[key]['var_b'], 1e-8, None).tolist()
            # Clip the policy_net_param_temp[key]['mu_w'] at 1e8 if it is possitive and -1e8 if it is negative
            # policy_net_param_temp[key]['mu_w']=np.sign(policy_net_param_temp[key]['mu_w'])*np.clip(np.abs(policy_net_param_temp[key]['mu_w']), 1e-8, None).tolist()
            # policy_net_param_temp[key]['mu_b']=np.sign(policy_net_param_temp[key]['mu_b'])*np.clip(np.abs(policy_net_param_temp[key]['mu_b']), 1e-8, None).tolist()
        self.policy_net.net.load_state_dict(policy_net_param_temp)

    def _plot_rewards(self, metric, show_result=False, ylim=None):
        plt.figure(3)
        durations_t = torch.tensor(metric, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(durations_t.numpy())
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def _select_action(self, state, greedy = False):
        self.policy_net.net.eval()
        state_np = state.numpy()

        state_temp = np.array(state_np)
        state_np = np.repeat(state_temp, self.batchsize, axis=0)

        ma, Sa = self.policy_net.net(state_np)
        ma = ma.reshape(self.batchsize, self.policy_net.n_actions*2)[0]
        action_mean = ma[::2]
        Sa = Sa.reshape(self.batchsize, self.policy_net.n_actions*2)[0]

        if greedy:
            action_var = Sa[::2] + ma[1::2]
            action = np.argmax(np.array(action_mean) - np.sqrt(action_var), axis=0)
        else:
            action_var = Sa[::2]

            a_sample = np.zeros_like(action_mean)
            for i in range(len(action_mean)):
                a_sample[i] = np.random.normal(action_mean[i], np.sqrt(action_var[i]))
            action = np.argmax(a_sample, axis=0)

        self.steps_done += 1
        return torch.tensor([[action]],device=self.device)

    def _track_Qvalues(self, state):
        self.policy_net.net.eval()
        state_np = state.numpy()
        Q_values_f,var_Q_val_f = self.policy_net.net(state_np)
        Q_values = Q_values_f[::2]
        var_Q_val = var_Q_val_f[::2] + Q_values_f[1::2]
        epistemic_uncertainty = var_Q_val_f[::2]
        aleatory_uncertainty = Q_values_f[1::2]
        return [Q_values], [var_Q_val], [epistemic_uncertainty], [aleatory_uncertainty]