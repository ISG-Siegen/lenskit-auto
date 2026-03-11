from ConfigSpace import ConfigurationSpace
from ConfigSpace import Configuration
from ConfigSpace import Constant
from typing import List


def get_default_configurations(config_space: ConfigurationSpace) -> List[Configuration]:
    """
    returns a list of default configurations for all algorithms in the configuration space

    Parameters
    ----------
    config_space : ConfigurationSpace
        configuration space to use

    Returns
    -------
    List[Configuration]
        List of default configurations for all algorithms in the configuration space
    """

    # get all algorithms in the configuration space
    if type(config_space.get('algo')) == Constant:
        algorithms = [config_space['algo'].value]
    else:
        algorithms = config_space["algo"].choices
    # initialize a list for default configurations for each algorithm
    initial_configuration_list = []

    # get the default configuration for each algorithm and store it in the list
    for algorithm in algorithms:
        # set the default value for the algorithm hyperparameter
        config_space.get('algo').default_value = str(algorithm)
        # get the default configuration for the algorithm
        configuration = config_space.get_default_configuration()
        # add the configuration to the list
        initial_configuration_list.append(configuration)

    # return the list of default configurations
    return initial_configuration_list
