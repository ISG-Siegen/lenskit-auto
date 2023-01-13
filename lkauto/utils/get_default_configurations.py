from ConfigSpace import ConfigurationSpace
from ConfigSpace import Configuration


def get_default_configurations(config_space: ConfigurationSpace) -> list[Configuration]:
    algorithms = config_space.get('regressor').choices
    initial_configuration_list = []

    for algorithm in algorithms:
        config_space.get('regressor').default_value = str(algorithm)
        configuration = config_space.get_default_configuration()
        initial_configuration_list.append(configuration)

    return initial_configuration_list

