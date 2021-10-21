import pandas as pd
from ConfigSpace import ConfigurationSpace


def update_top_n_runs(num_models: int, top_n_runs: pd.DataFrame, run_id: int, config_space: ConfigurationSpace, errors):
    """
    updates the top n runs dataframe with the new run

    Parameters
    ----------
    num_models : int
        number of models to keep track of
    top_n_runs : pd.DataFrame
        pandas dataframe containing the top n runs of the optimization method.
    run_id : int
        run id of the new run
    config_space : ConfigurationSpace
        configuration space of the new run
    errors : np.ndarray
        errors of the new run

    Returns
    -------
    top_n_runs : pd.DataFrame
        pandas dataframe containing the top n runs of the optimization method.
    """
    # create a dataframe containing the run id, model and error of the new run
    model_performance = pd.DataFrame(data={'run_id': [run_id],
                                           'model': [config_space['algo']],
                                           'error': [errors.mean()]})

    # if the top n runs dataframe does not contain an entry for each model, add the new run to the dataframe
    if len(top_n_runs) < num_models:
        top_n_runs = pd.concat([top_n_runs, model_performance])
    # if the top n runs dataframe contains an entry for each model, check if the new run is better than the worst run
    else:
        # get the runs for each model type
        item_item_silo = top_n_runs[top_n_runs['model'] == 'ItemItem']
        user_user_silo = top_n_runs[top_n_runs['model'] == 'UserUser']
        als_bias_mf_silo = top_n_runs[top_n_runs['model'] == 'ALSBiasedMF']
        bias_silo = top_n_runs[top_n_runs['model'] == 'Bias']
        funk_svd_silo = top_n_runs[top_n_runs['model'] == 'FunkSVD']
        biased_svd_silo = top_n_runs[top_n_runs['model'] == 'BiasedSVD']

        # create a list containing the runs for each model type
        silo_list = [item_item_silo, user_user_silo, als_bias_mf_silo, bias_silo, funk_svd_silo, biased_svd_silo]

        max_len = 0
        max_index = 0

        # get the model type with the most runs
        for i, silo in enumerate(silo_list):
            if len(silo) > max_len:
                max_len = len(silo)
                max_index = i

        # get the runs for the model type with the most runs
        silo = top_n_runs[top_n_runs['model'] == config_space['regressor']]

        if len(silo) == max_len:
            # get the worst run of the model type with the most runs
            worst_performance = silo['error'].max()

            # remove the worst run of the model type with the most runs
            top_n_runs = top_n_runs[top_n_runs['model'] != config_space['regressor']]

            # add the new run to the dataframe
            silo = silo[silo['error'] < worst_performance]
            silo = pd.concat([silo, model_performance])

            # add the runs of the model type with the most runs to the dataframe
            top_n_runs = pd.concat([top_n_runs, silo])
        else:
            # get the worst run of the model type with the most runs
            max_silo = silo_list[max_index]

            # remove the worst run of the model type with the most runs
            worst_performance = max_silo['error'].max()
            top_n_runs = top_n_runs[top_n_runs['model'] != max_silo['model'].iloc[0]]
            max_silo = max_silo[max_silo['error'] < worst_performance]
            top_n_runs = pd.concat([top_n_runs, max_silo])

            # add the new run to the dataframe
            top_n_runs = pd.concat([top_n_runs, model_performance])

    return top_n_runs
