from pathlib import Path
from ConfigSpace import ConfigurationSpace
import pandas as pd
import numpy as np
import os
import json
import pickle


class Filer:
    """Filer to handle the LensKit-Auto output

        This filer supports to structure the LensKit-Auto output in the file system

        Attributes
        ----------
        output_directory_path: path to the output directory
            path that leads to the output directory of LensKit-Auto

        Methods
        ----------
        get_output_directory_path() -> str
        get_smac_output_directory_path() -> str
        set_output_directory_path(output_directory_path: str) -> None
        save_dataframe_as_csv(self, dataframe: pd.DataFrame, output_path: str, name: str) -> None
        save_dictionary_to_json(self, dictionary: dict, output_path: str, name: str) -> None
        save_metric_scores_to_txt(self, metric_scores: np.array, output_path: str, name: str) -> None:
        get_dataframe_from_csv(self, path_to_file: str, index_column=None) -> pd.DataFrame:
        get_series_from_csv(self, path_to_file: str, index_column=None) -> pd.Series
        get_dict_from_json_file(self, path_to_file: str) -> dict
        get_numpy_array_from_txt_file(self, path_to_file: str) -> np.array
        append_dataframe_to_csv(self, dataframe: pd.DataFrame, output_path: str, name: str) -> None
        save_validataion_data(self,
                              config_space: ConfigurationSpace,
                              predictions: pd.DataFrame,
                              metric_scores: np.array,
                              output_path: str,
                              run_id: int) -> None
    """
    def __init__(self, output_directory_path='output/'):
        self.output_directory_path = output_directory_path
        Path(output_directory_path).mkdir(parents=True, exist_ok=True)

    def __check_if_folder_structure_exist(self, output_path: str) -> None:
        folder_path = os.path.join(self.output_directory_path, output_path)
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    def get_output_directory_path(self) -> str:
        return self.output_directory_path

    def get_smac_output_directory_path(self) -> str:
        return os.path.join(self.output_directory_path, 'smac')

    def set_output_directory_path(self, output_directory_path: str) -> None:
        self.output_directory_path = output_directory_path

    def save_dataframe_as_csv(self, dataframe: pd.DataFrame, output_path: str, name: str) -> None:
        self.__check_if_folder_structure_exist(output_path=output_path)
        data_path = os.path.join(self.output_directory_path, output_path, '{}.csv'.format(name))
        dataframe.to_csv(path_or_buf=data_path)

    def save_dictionary_to_json(self, dictionary: dict, output_path: str, name: str) -> None:
        self.__check_if_folder_structure_exist(output_path)
        data_path = os.path.join(self.output_directory_path, output_path, '{}.json'.format(name))

        temp_dic = {}
        for key, value in dictionary.items():
            if isinstance(value, np.bool):
                if value:
                    temp_dic[key] = True
                else:
                    temp_dic[key] = False
            else:
                temp_dic[key] = value

        with open(data_path, 'w') as f:
            json.dump(temp_dic, f)

    def save_metric_scores_to_txt(self, metric_scores: np.array, output_path: str, name: str) -> None:
        self.__check_if_folder_structure_exist(output_path=output_path)
        data_path = os.path.join(self.get_output_directory_path(), output_path, '{}.txt'.format(name))

        np.savetxt(data_path, metric_scores)

    def get_dataframe_from_csv(self, path_to_file: str, index_column=None) -> pd.DataFrame:
        if index_column is None:
            dataframe = pd.read_csv(os.path.join(self.get_output_directory_path(), path_to_file))
        else:
            dataframe = pd.read_csv(os.path.join(self.get_output_directory_path(), path_to_file),
                                    index_col=index_column)
        return dataframe

    def get_series_from_csv(self, path_to_file: str, index_column=None) -> pd.Series:
        if index_column is None:
            series = pd.read_csv(os.path.join(self.get_output_directory_path(), path_to_file))
        else:
            series = pd.read_csv(os.path.join(self.get_output_directory_path(), path_to_file),
                                 index_col=index_column)
        series = series.iloc[:, 0]
        return series

    def get_dict_from_json_file(self, path_to_file: str) -> dict:
        with open(os.path.join(self.get_output_directory_path(), path_to_file)) as f:
            temp_dic = json.load(f)

        dictionary = {}

        for key, value in temp_dic.items():
            if isinstance(value, bool):
                if value:
                    dictionary[key] = True
                else:
                    dictionary[key] = False
            else:
                dictionary[key] = value

        return dictionary

    def get_numpy_array_from_txt_file(self, path_to_file: str) -> np.array:
        return np.loadtxt(fname=os.path.join(self.get_output_directory_path(), path_to_file))

    def append_dataframe_to_csv(self, dataframe: pd.DataFrame, output_path: str, name: str) -> None:
        self.__check_if_folder_structure_exist(output_path=output_path)
        data_path = os.path.join(self.get_output_directory_path(), output_path, '{}.csv'.format(name))

        dataframe.to_csv(data_path, mode='a', header=False, index=False)

    def save_validataion_data(self,
                              config_space: ConfigurationSpace,
                              predictions: pd.DataFrame,
                              metric_scores: np.array,
                              output_path: str,
                              run_id: int) -> None:
        """method to simplify the validation data output

                The validation data output is mainly used to build ensambles of the best models.
                Therefore, predictions, error scores and configuration spaces of each run need
                to be stored to the file system

                Parameters
                ----------
                config_space : ConfigurationSpace
                    configuraiton space of run
                predictions : pd.Dataframe
                    dataframe containing raw predictions
                metric_scores : np.array
                    numpy array containing metric values
                    Depending on the metric, differnet kind of metric values can be stored
                    in the metric scores array
                output_path : str
                    path to output folder
                run_id : int
                    id of smac search iteration
            """
        output_path = os.path.join(output_path, str(run_id))
        dictionary = dict(config_space)
        print("Saving predictions:")
        print(predictions.head())
        print(predictions.columns)

        self.save_metric_scores_to_txt(metric_scores=metric_scores, output_path=output_path, name='rmse')
        self.save_dictionary_to_json(dictionary=dictionary, output_path=output_path, name='config_space')
        self.save_dataframe_as_csv(dataframe=predictions, output_path=output_path, name='predictions')

    def save_model(self, model):
        """
        Saves the model to the output directory specified by the filer

        Parameters
        ----------
        model
            The model to be saved
        """
        model_name = model.__class__.__name__
        timestamp = pd.Timestamp.now().strftime('%d.%m.%Y-%H:%M')
        file_name = "Trained_" + model_name + "_" + timestamp + '.pkl'

        output_path = Path(self.output_directory_path)

        pickle.dump(model, open(output_path / file_name, 'wb'))

    def save_incumbent(self, incumbent):
        """
        Saves the incumbent to the output directory specified by the filer

        Parameters
        ----------
        incumbent
                    The incumbent to be saved
        """
        timestamp = pd.Timestamp.now().strftime('%d.%m.%Y-%H:%M')
        file_name = "incumbent _" + timestamp + '.pkl'

        output_path = Path(self.output_directory_path)

        pickle.dump(incumbent, open(output_path / file_name, 'wb'))
