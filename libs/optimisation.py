import numpy as np
import time
import scipy as sp
from numpy.linalg import norm
from numpy.linalg import pinv
from numpy import identity as eye


def solve_wrt_d_stochastic(d_matrix, training_settings, data, x_train, y_train, n_points, task_range, param1, c_iter):
    def batch_objective(D):
        return sum([n_points[i] * norm(pinv(x_train[i] @ D @ x_train[i].T +
                                            n_points[i] * eye(n_points[i])) @
                                       y_train[i]) ** 2 for i in task_range])

    def batch_grad(D):
        return batch_grad_func(D, task_range, data)

    c_value = training_settings['c_value']

    curr_obj = batch_objective(d_matrix)

    objectives = []

    n_iter = 1
    curr_tol = 10 ** 10
    conv_tol = 10 ** -8
    inner_iter = 0

    t = time.time()
    while (inner_iter < n_iter) and (curr_tol > conv_tol):
        inner_iter = inner_iter + 1
        prev_d = d_matrix
        prev_obj = curr_obj

        c_iter = c_iter + inner_iter
        step_size = c_value / np.sqrt(c_iter)
        d_matrix = prev_d - step_size * batch_grad(prev_d)

        d_matrix = psd_trace_projection(d_matrix, 1 / param1)

        curr_obj = batch_objective(d_matrix)
        objectives.append(curr_obj)

        curr_tol = abs(curr_obj - prev_obj) / prev_obj

        if time.time() - t > 5:
            t = time.time()
            print("iter: %5d | obj: %20.18f | tol: %20.18f" % (c_iter, curr_obj, curr_tol))

    return d_matrix, c_iter


def batch_grad_func(D, task_indeces, data):
    X = [data['x_train'][i] for i in task_indeces]
    Y = [data['y_train'][i] for i in task_indeces]
    n_dims = X[0].shape[1]

    def M(D, n, t):
        return X[t] @ D @ X[t].T + n * eye(n)

    grad = np.zeros((n_dims, n_dims))
    for idx, _ in enumerate(task_indeces):
        n_points = len(Y[idx])

        Y[idx] = np.reshape(Y[idx], [1, len(Y[idx])])
        invM = sp.linalg.inv(M(D, n_points, idx))
        curr_grad = X[idx].T @ invM @ ((Y[idx].T @ Y[idx]) @ invM + invM @ (Y[idx].T @ Y[idx])) @ invM @ X[idx]

        curr_grad = -n_points * curr_grad

        grad = grad + curr_grad
    return grad


def psd_trace_projection(D, constraint):
    s, U = np.linalg.eigh(D)
    s = np.maximum(s, 0)

    if np.sum(s) < constraint:
        return U @ np.diag(s) @ U.T

    search_points = np.insert(s, 0, 0)
    low_idx = 0
    high_idx = len(search_points) - 1

    obj = lambda vec, x: np.sum(np.maximum(vec - x, 0))

    while low_idx <= high_idx:
        mid_idx = np.int(np.round((low_idx + high_idx) / 2))
        s_sum = obj(s, search_points[mid_idx])

        if s_sum == constraint:
            s = np.sort(s)
            D_proj = U @ np.diag(s) @ U.T
            return D_proj
        elif s_sum > constraint:
            low_idx = mid_idx + 1
        elif s_sum < constraint:
            high_idx = mid_idx - 1

    if s_sum > constraint:
        slope = (s_sum - obj(s, search_points[mid_idx + 1])) / (search_points[mid_idx] - search_points[mid_idx + 1])
        intercept = s_sum - slope * search_points[mid_idx]

        matching_point = (constraint - intercept) / slope
        s_sum = obj(s, matching_point)
    elif s_sum < constraint:
        slope = (s_sum - obj(s, search_points[mid_idx - 1])) / (search_points[mid_idx] - search_points[mid_idx - 1])
        intercept = s_sum - slope * search_points[mid_idx]

        matching_point = (constraint - intercept) / slope
        s_sum = obj(s, matching_point)

    s = np.maximum(s - matching_point, 0)
    s = np.sort(s)
    D_proj = U @ np.diag(s) @ U.T

    return D_proj


def solve_wrt_w(D, X, Y, W_pred, task_range):
    for _, task_idx in enumerate(task_range):
        n_points = len(Y[task_idx])
        curr_w_pred = (D @ X[task_idx].T @
                       np.linalg.solve(X[task_idx] @ D @ X[task_idx].T +
                                       n_points * eye(n_points), Y[task_idx])).ravel()
        W_pred[:, task_idx] = curr_w_pred
    return W_pred


def solve_wrt_d(D, data, x_train, y_train, n_points, task_range, param1):
    def batch_objective(D):
        return sum([n_points[i] * norm(sp.linalg.inv(x_train[i] @ D @ x_train[i].T +
                                                     n_points[i] * eye(n_points[i])) @
                                       y_train[i]) ** 2 for i in task_range])

    D = np.eye(D.shape[0])

    def batch_grad(D):
        return batch_grad_func(D, task_range, data)

    curr_obj = batch_objective(D)

    objectives = []
    n_iter = 10 ** 10
    curr_tol1 = 10 ** 10
    conv_tol_obj = 10 ** -5
    c_iter = 0

    t = time.time()
    while (c_iter < n_iter) and (curr_tol1 > conv_tol_obj):
        prev_D = D
        prev_obj = curr_obj

        step_size = 10 ** 16
        grad = batch_grad(prev_D)

        temp_D = psd_trace_projection(prev_D - step_size * grad, 1 / param1)
        temp_obj = batch_objective(temp_D)

        while temp_obj > (prev_obj + np.trace(grad.T @ (temp_D - prev_D)) +
                          1 / (2 * step_size) * norm(prev_D - temp_D, ord='fro') ** 2):
            step_size = 0.5 * step_size

            temp_D = psd_trace_projection(prev_D - step_size * grad, 1 / param1)
            temp_obj = batch_objective(temp_D)

        D = psd_trace_projection(prev_D - step_size * grad, 1 / param1)

        curr_obj = batch_objective(D)
        objectives.append(curr_obj)

        curr_tol1 = abs(curr_obj - prev_obj) / prev_obj
        c_iter = c_iter + 1

        if time.time() - t > 5:
            t = time.time()
            print("iter: %5d | obj: %12.8f | objtol: %10e | step: %5.3e" %
                  (c_iter, curr_obj, curr_tol1, step_size))

    print("iter: %5d | obj: %12.8f | objtol: %10e" %
          (c_iter, curr_obj, curr_tol1))

    return D
