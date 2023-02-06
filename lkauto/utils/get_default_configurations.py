from ConfigSpace import ConfigurationSpace
from ConfigSpace import Configuration
from typing import List


def get_default_configurations(config_space: ConfigurationSpace) -> List[Configuration]:
    algorithms = config_space.get('algo').choices
    initial_configuration_list = []

    for algorithm in algorithms:
        config_space.get('algo').default_value = str(algorithm)
        configuration = config_space.get_default_configuration()
        initial_configuration_list.append(configuration)

    return initial_configuration_list

