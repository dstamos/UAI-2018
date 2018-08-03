from libs.utilities import *
from libs.online_ltl import online_ltl
from libs.batch_ltl import batch_ltl


if __name__ == "__main__":

    seed = 1337
    n_points = 140
    n_dims = 40
    n_tasks = 200
    method_IDX = 1

    if method_IDX == 0:
        method = 'batch_LTL'
        c_value = np.nan
    else:
        c_value = 10 ** 8
        method = 'online_LTL'

    param1 = 10 ** -3
    print('seed: %d | lambda: %20.15f | c: %20.10f' % (seed, param1, c_value))

    data_settings = {'n_points': n_points,
                     'n_dims': n_dims,
                     'n_observed_tasks': n_tasks,
                     'dataset': 'schools',
                     'seed': seed}
    training_settings = {'param1': param1,
                         'c_value': c_value}

    data, data_settings = synthetic_data_gen(data_settings)

    training_settings['filename'] = "seed_" + str(seed) + '-c_value_' + str(c_value)

    training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                      str(n_tasks) + '-n_' + str(n_points) + '/' \
                                      + method

    if method_IDX == 0:
        training_settings['method'] = method
        results, val_perf = batch_ltl(data, data_settings, training_settings)
    else:
        training_settings['method'] = method
        results, val_perf = online_ltl(data, data_settings, training_settings)

    save_results(results, data_settings, training_settings)

    print("done")
