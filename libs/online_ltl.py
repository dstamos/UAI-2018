import numpy as np
from libs.optimisation import solve_wrt_d_stochastic
from libs.optimisation import solve_wrt_w
from libs.utilities import performance_measurement


def online_ltl(data, data_settings, training_settings):
    param1 = training_settings['param1']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']

    n_tr_tasks = len(task_range_tr)
    all_val_perf, all_test_perf = [[] for _ in range(n_tr_tasks)], [[] for _ in range(n_tr_tasks)]
    best_param, best_train_perf, best_val_perf, best_test_perf = \
        [None] * n_tr_tasks, [None] * n_tr_tasks, [None] * n_tr_tasks, [None] * n_tr_tasks

    c_iter = 0
    D = np.eye(n_dims)

    all_D = [None] * n_tr_tasks

    W_pred = np.zeros((n_dims, n_tasks))
    for pure_task_idx, curr_task_range_tr in enumerate(task_range_tr):

        # OPTIMISATION
        x_train, y_train = [None] * n_tasks, [None] * n_tasks
        n_points = [0] * n_tasks
        for _, task_idx in enumerate([curr_task_range_tr]):
            x_train[task_idx] = data['x_train'][task_idx]
            y_train[task_idx] = data['y_train'][task_idx]
            n_points[task_idx] = len(y_train[task_idx])

        D, c_iter = solve_wrt_d_stochastic(D, training_settings, data, x_train, y_train, n_points,
                                           [curr_task_range_tr], param1, c_iter)

        #####################################################
        # VALIDATION
        W_pred = solve_wrt_w(D, data['x_train'], data['y_train'], W_pred, task_range_val)

        val_perf = performance_measurement(data_settings, data['x_val'], data['y_val'], W_pred, task_range_val)
        all_val_perf[pure_task_idx].append(val_perf)

        #####################################################
        # TEST
        all_D[pure_task_idx] = D
        if pure_task_idx >= 1:
            D_average = np.average(all_D[:pure_task_idx], axis=0)
        else:
            D_average = D

        x_train, y_train = [None] * n_tasks, [None] * n_tasks
        for _, task_idx in enumerate(task_range_test):
            x_train[task_idx] = np.concatenate((data['x_val'][task_idx], data['x_train'][task_idx]))
            y_train[task_idx] = np.concatenate((data['y_val'][task_idx], data['y_train'][task_idx]))

        W_pred = solve_wrt_w(D_average, x_train, y_train, W_pred, task_range_test)

        test_perf = performance_measurement(data_settings, data['x_test'], data['y_test'], W_pred, task_range_test)
        all_test_perf[pure_task_idx].append(test_perf)

        best_val_perf[pure_task_idx] = val_perf
        best_test_perf[pure_task_idx] = test_perf

        print('T: %3d (%3d) | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f' %
              (pure_task_idx, curr_task_range_tr, param1, val_perf, test_perf))

    results = {'param1': param1,
               'all_val_perf': best_val_perf,
               'all_test_perf': best_test_perf}

    return results, best_val_perf[-1]
