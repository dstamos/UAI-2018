import pickle
import os
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split


def performance_measurement(data_settings, X, true, W, task_indeces):
    n_tasks = len(task_indeces)
    explained_variance = 0
    for _, task_idx in enumerate(task_indeces):
        pred = X[task_idx] @ W[:, task_idx]

        explained_variance = explained_variance + explained_variance_score(true[task_idx].ravel(), pred)

    performance = 100 * explained_variance / n_tasks
    return performance


def save_results(results, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']

    if not os.path.exists(foldername):
        os.makedirs(foldername)
    f = open(foldername + '/' + filename + ".pckl", 'wb')
    pickle.dump(results, f)
    pickle.dump(data_settings, f)
    pickle.dump(training_settings, f)
    f.close()


def synthetic_data_gen(data_settings):
    n_observed_tasks = data_settings['n_observed_tasks']
    n_test_tasks = 300
    n_tasks = n_test_tasks + n_observed_tasks

    data_settings['train_perc'] = 0.5
    data_settings['val_perc'] = 0.5
    data_settings['noise'] = 0.25
    # data_settings['noise'] = 0

    n_dims = data_settings['n_dims']
    n_points = data_settings['n_points']
    val_perc = data_settings['val_perc']
    noise = data_settings['noise']

    sparsity = n_dims

    fixed_sparsity =  np.random.choice(np.arange(0, n_dims), sparsity, replace=False)

    data = {}
    W_true = np.zeros((n_dims, n_tasks))
    x_train, y_train = [None]*n_tasks, [None]*n_tasks
    x_val, y_val = [None] * n_tasks, [None] * n_tasks
    x_test, y_test = [None] * n_tasks, [None] * n_tasks
    for task_idx in range(0, n_tasks):
        # generating and normalizing the data
        features = np.random.randn(n_points, n_dims)
        features = features / norm(features, axis=1, keepdims=True)

        # generating and normalizing the weight vectors
        weight_vector = np.zeros((n_dims, 1))
        weight_vector[fixed_sparsity] = np.random.randn(sparsity, 1)
        weight_vector = (weight_vector / norm(weight_vector)).ravel() * np.random.randint(1, 10)

        labels = features @ weight_vector + noise * np.random.randn(n_points)

        x_train_all, x_test[task_idx], y_train_all, y_test[task_idx] = train_test_split(features, labels,
                                                                                        test_size=100)
        test_size = int(np.floor(val_perc * len(y_train_all)))
        x_train[task_idx], x_val[task_idx], y_train[task_idx], y_val[task_idx] = train_test_split(x_train_all,
                                                                                                  y_train_all,
                                                                                                  test_size=test_size)

        W_true[:, task_idx] = weight_vector

    n_train_tasks = round(n_observed_tasks * 0.5)
    shuffled_tasks = np.random.permutation(n_observed_tasks+n_test_tasks)
    data_settings['task_range_tr'] = list(shuffled_tasks[:n_train_tasks])

    for task_idx in data_settings['task_range_tr']:
        x_temp = np.concatenate((x_val[task_idx], x_train[task_idx]))
        y_temp = np.concatenate((y_val[task_idx], y_train[task_idx]))
        x_train[task_idx] = x_temp
        y_train[task_idx] = y_temp
        x_val[task_idx] = []
        y_val[task_idx] = []

    data_settings['task_range_val'] = list(shuffled_tasks[n_train_tasks:n_observed_tasks])
    data_settings['task_range_test'] = list(shuffled_tasks[n_observed_tasks:n_observed_tasks + n_test_tasks])
    data_settings['task_range'] = list(np.arange(0, n_tasks))
    data_settings['n_tasks'] = n_tasks

    data['x_train'] = x_train
    data['y_train'] = y_train
    data['x_val'] = x_val
    data['y_val'] = y_val
    data['x_test'] = x_test
    data['y_test'] = y_test
    data['W_true'] = W_true

    return data, data_settings




