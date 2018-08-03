import numpy as np
from libs.optimisation import solve_wrt_d
from libs.optimisation import solve_wrt_w
from libs.utilities import performance_measurement


def batch_ltl(data, data_settings, training_settings):
    param1 = training_settings['param1']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']

    D = np.eye(n_dims)
    W_pred = np.zeros((n_dims, n_tasks))

    #####################################################
    # OPTIMISATION
    x_train, y_train = [None] * n_tasks, [None] * n_tasks
    n_points = [0] * n_tasks
    for _, task_idx in enumerate(task_range_tr):
        x_train[task_idx] = data['x_train'][task_idx]
        y_train[task_idx] = data['y_train'][task_idx]
        n_points[task_idx] = len(y_train[task_idx])

    D = solve_wrt_d(D, data, x_train, y_train, n_points, task_range_tr, param1)

    #####################################################
    # VALIDATION
    W_pred = solve_wrt_w(D, data['x_train'], data['y_train'], W_pred, task_range_val)

    val_perf = performance_measurement(data_settings, data['x_val'], data['y_val'], W_pred, task_range_val)

    #####################################################
    # TEST
    x_train, y_train = [None] * n_tasks, [None] * n_tasks
    for _, task_idx in enumerate(task_range_test):
        x_train[task_idx] = np.concatenate((data['x_val'][task_idx], data['x_train'][task_idx]))
        y_train[task_idx] = np.concatenate((data['y_val'][task_idx], data['y_train'][task_idx]))

    W_pred = solve_wrt_w(D, x_train, y_train, W_pred, task_range_test)
    test_perf = performance_measurement(data_settings, data['x_test'], data['y_test'], W_pred, task_range_test)

    print('lambda: %6e | val MSE: %7.5f | test MSE: %7.5f' %
          (param1, val_perf, test_perf))

    results = {'param1': param1,
               'val_perf': val_perf,
               'test_perf': test_perf}

    return results, val_perf
