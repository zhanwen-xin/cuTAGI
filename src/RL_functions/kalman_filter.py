from src.RL_functions.helpers import *
import numpy as np
class KalmanFilter(object):
    def __init__(self, components, time_step, hyperparameters):
        self.A, self.F, self.Q, self.R = build_matrices(components, time_step, hyperparameters)

        self.components = components
        self.time_step = time_step
        self.hyperparameters = hyperparameters

    def predict(self, x_ls, current_time_stamp = 0, LA_process_error_var = 1e-18):

        # # Set LA process error to a constant
        # self.Q[2, 2] = LA_process_error_var

        # Process error for x
        W = {'mu': np.zeros(len(x_ls['mu'])), 'var': self.Q}
        # Observation error
        V = {'mu': np.array([0]), 'var': self.R}

        # Transision model: no intervention
        x_pred = coefficient_times_gaussian(self.A, x_ls)
        x_pred = gaussian_plus_gaussian(x_pred, W)
        self.x_pred = x_pred

        # Prediction
        y_pred = coefficient_times_gaussian(self.F, x_pred)
        y_pred = gaussian_plus_gaussian(y_pred, V)
        self.y_pred = y_pred
        return y_pred, x_pred, self.A

    def update(self, yi):
        x_upt = self.obs_update(yi, self.x_pred)

        likelihood = norm.pdf(yi, loc=self.y_pred['mu'], scale=np.sqrt(self.y_pred['var']))

        # Compute the mean squared error
        mse = (yi - self.y_pred['mu'])**2

        return x_upt, likelihood, mse

    def smooth(self, x_upt, x_pred_next, smoothed_x_next, A):
        J = x_upt['var'].dot(A.T).dot(np.linalg.pinv(x_pred_next['var'], rcond=1e-12))
        mu_smooth = x_upt['mu'] + J.dot(smoothed_x_next['mu'] - x_pred_next['mu'])
        var_smooth = x_upt['var'] + J.dot(smoothed_x_next['var'] - x_pred_next['var']).dot(J.T)
        x_smooth = {'mu': mu_smooth, 'var': var_smooth}
        return x_smooth

    def obs_update(self, yi, z):
        # Use self.y_pred and self.matrices.H to update hidden state z
        # Update a Gaussian distribution using an deterministic observation
        cov_zy = z['var'] @ self.F.T
        mu_upt = z['mu'] + (cov_zy * self.y_pred['var']**(-1) * (yi - self.y_pred['mu']))
        var_upt = z['var'] - np.reshape(cov_zy, (-1, len(z['mu']))).T * self.y_pred['var']**(-1) * np.reshape(cov_zy, (-1, len(z['mu'])))
        z_upt = {'mu': mu_upt, 'var': var_upt}
        return z_upt

    def dist_update(self, z_upt, z_pred, x_t, cov_xz):
        # Use z_upt and self.matrices.self.F to update x_t
        # Update a Gaussian distribution using a Gaussian distribution
        jacobian = cov_xz @ np.linalg.inv(z_pred['var'])
        mu_upt = x_t['mu'] + jacobian @ (z_upt['mu'] - z_pred['mu'])
        var_upt = x_t['var'] + jacobian @ (z_upt['var'] - z_pred['mu']) @ jacobian.T
        x_upt = {'mu': mu_upt, 'var': var_upt}
        return x_upt