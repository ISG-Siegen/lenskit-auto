# test_filer.py
import os
import pickle
import unittest
import tempfile
import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace

from lkauto.utils.filer import Filer


class DummyModel:
    pass


class TestFiler(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = self.temp_dir.name
        self.filer = Filer(output_directory_path=self.output_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_givenPath_directoryCreated(self):
        self.assertTrue(os.path.exists(self.output_path))

    def test_set_output_directory_path_givenNewPath_pathUpdated(self):
        new_path = os.path.join(self.output_path, "new_output")

        self.filer.set_output_directory_path(new_path)

        self.assertEqual(self.filer.get_output_directory_path(), new_path)

    def test_get_smac_output_directory_path_givenDefault_expectedSmacFolder(self):
        smac_path = self.filer.get_smac_output_directory_path()

        expected = os.path.join(self.output_path, "smac")

        self.assertEqual(smac_path, expected)

    def test_save_dataframe_and_load_dataframe_expectedEquality(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        self.filer.save_dataframe_as_csv(df, "data", "test")

        loaded = self.filer.get_dataframe_from_csv("data/test.csv")

        pd.testing.assert_frame_equal(df, loaded.iloc[:, 1:])

    def test_get_dataframe_from_csv_givenIndexColumn_expectedIndexPreserved(self):
        df = pd.DataFrame({"a": [1, 2]}, index=[10, 11])

        self.filer.save_dataframe_as_csv(df, "data", "index_test")

        loaded = self.filer.get_dataframe_from_csv("data/index_test.csv", index_column=0)

        self.assertEqual(list(loaded.index), [10, 11])

    def test_save_dictionary_and_load_dictionary_expectedEquality(self):
        dictionary = {"a": 1, "b": True}

        self.filer.save_dictionary_to_json(dictionary, "config", "test")

        loaded = self.filer.get_dict_from_json_file("config/test.json")

        self.assertEqual(dictionary, loaded)

    def test_save_dictionary_givenNumpyBool_expectedConvertedCorrectly(self):
        dictionary = {"flag": np.bool_(True)}

        self.filer.save_dictionary_to_json(dictionary, "config", "bool_test")

        loaded = self.filer.get_dict_from_json_file("config/bool_test.json")

        self.assertTrue(loaded["flag"])

    def test_save_metric_scores_and_load_numpy_expectedEquality(self):
        arr = np.array([1.0, 2.0, 3.0])

        self.filer.save_metric_scores_to_txt(arr, "metrics", "scores")

        loaded = self.filer.get_numpy_array_from_txt_file("metrics/scores.txt")

        self.assertTrue(np.allclose(arr, loaded))

    def test_get_series_from_csv_givenFile_expectedSeriesReturned(self):
        df = pd.DataFrame({"a": [1, 2, 3]})

        self.filer.save_dataframe_as_csv(df, "data", "series_test")

        series = self.filer.get_series_from_csv("data/series_test.csv")

        self.assertIsInstance(series, pd.Series)
        self.assertEqual(len(series), 3)

    def test_get_series_from_csv_givenIndexColumn_expectedSeriesReturned(self):
        df = pd.DataFrame({"a": [1, 2, 3]})

        self.filer.save_dataframe_as_csv(df, "data", "series_index")

        series = self.filer.get_series_from_csv("data/series_index.csv", index_column=0)

        self.assertIsInstance(series, pd.Series)

    def test_append_dataframe_to_csv_givenTwoDataframes_rowsDoubled(self):
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [3], "b": [4]})

        self.filer.save_dataframe_as_csv(df1, "data", "append_test")
        self.filer.append_dataframe_to_csv(df2, "data", "append_test")

        loaded = pd.read_csv(os.path.join(self.output_path, "data", "append_test.csv"))

        self.assertEqual(len(loaded), 2)

    def test_append_dataframe_to_csv_givenTwoDataframes_valuesCorrect(self):
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [3], "b": [4]})

        self.filer.save_dataframe_as_csv(df1, "data", "append_values")
        self.filer.append_dataframe_to_csv(df2, "data", "append_values")

        loaded = pd.read_csv(os.path.join(self.output_path, "data", "append_values.csv"))

        self.assertEqual(list(loaded["a"]), [1, 3])

    def test_nested_folder_structure_created_whenSavingData(self):
        df = pd.DataFrame({"a": [1]})

        self.filer.save_dataframe_as_csv(df, "nested/folder/structure", "test")

        expected_path = os.path.join(
            self.output_path,
            "nested",
            "folder",
            "structure",
            "test.csv"
        )

        self.assertTrue(os.path.exists(expected_path))

    def test_save_validation_data_expectedFilesCreated(self):
        predictions = pd.DataFrame({
            "user_id": [1],
            "item_id": [2],
            "score": [0.5]
        })

        metric_scores = np.array([0.1])

        cs = ConfigurationSpace()

        self.filer.save_validataion_data(
            config_space={},
            predictions=predictions,
            metric_scores=metric_scores,
            output_path="validation",
            run_id=1
        )

        base = os.path.join(self.output_path, "validation", "1")

        self.assertTrue(os.path.exists(os.path.join(base, "rmse.txt")))
        self.assertTrue(os.path.exists(os.path.join(base, "config_space.json")))
        self.assertTrue(os.path.exists(os.path.join(base, "predictions.csv")))

    def test_save_model_givenModel_pickleCreated(self):
        model = DummyModel()

        self.filer.save_model(model)

        files = os.listdir(self.output_path)

        pkl_files = [f for f in files if f.startswith("Trained_") and f.endswith(".pkl")]

        self.assertEqual(len(pkl_files), 1)

        loaded = pickle.load(open(os.path.join(self.output_path, pkl_files[0]), "rb"))

        self.assertIsInstance(loaded, DummyModel)

    def test_save_incumbent_givenObject_pickleCreated(self):
        incumbent = {"best": True}

        self.filer.save_incumbent(incumbent)

        files = os.listdir(self.output_path)

        pkl_files = [f for f in files if f.startswith("incumbent") and f.endswith(".pkl")]

        self.assertEqual(len(pkl_files), 1)

        loaded = pickle.load(open(os.path.join(self.output_path, pkl_files[0]), "rb"))

        self.assertEqual(loaded, incumbent)


if __name__ == "__main__":
    unittest.main()
