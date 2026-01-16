import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from lkauto.explicit.explicit_evaler import ExplicitEvaler


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def rmse():
    pass

rmse.__name__ = "rmse"


class DummyConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


# ----------------------------------------------------------------------
# Global dependency patching fixture
# ----------------------------------------------------------------------

@pytest.fixture
def mocks():
    """
    Patch all external dependencies used by ExplicitEvaler,
    INCLUDING validation_split to prevent LensKit execution.
    """
    with patch("lkauto.explicit.explicit_evaler.validation_split") as validation_split, \
         patch("lkauto.explicit.explicit_evaler.get_model_from_cs") as get_model, \
         patch("lkauto.explicit.explicit_evaler.predict_pipeline") as predict_pipeline, \
         patch("lkauto.explicit.explicit_evaler.topn_pipeline") as topn_pipeline, \
         patch("lkauto.explicit.explicit_evaler.predict") as predict, \
         patch("lkauto.explicit.explicit_evaler.recommend") as recommend, \
         patch("lkauto.explicit.explicit_evaler.RunAnalysis") as run_analysis_cls, \
         patch("lkauto.explicit.explicit_evaler.update_top_n_runs") as update_top_n_runs:

        # Prevent __init__ from touching LensKit internals
        validation_split.return_value = []

        yield {
            "validation_split": validation_split,
            "get_model": get_model,
            "predict_pipeline": predict_pipeline,
            "topn_pipeline": topn_pipeline,
            "predict": predict,
            "recommend": recommend,
            "run_analysis_cls": run_analysis_cls,
            "update_top_n_runs": update_top_n_runs,
        }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_single_fold_prediction_evaluation(mocks):
    train = MagicMock()
    filer = MagicMock()

    evaler = ExplicitEvaler(
        train=train,
        optimization_metric=rmse,
        filer=filer,
        split_folds=1,
        predict_mode=True
    )

    fold = MagicMock()
    fold.train = "train_data"
    fold.test = "test_data"
    evaler.train_test_splits = [fold]

    mocks["get_model"].return_value = MagicMock()

    pipeline = MagicMock()
    pipeline.clone.return_value = pipeline
    mocks["predict_pipeline"].return_value = pipeline

    recs = MagicMock()
    recs.to_df.return_value = pd.DataFrame(
        {"user": [1], "item": [1], "prediction": [4.0]}
    )
    mocks["predict"].return_value = recs

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.return_value = \
        pd.DataFrame({"mean": [1.5]}, index=["rmse"])
    mocks["run_analysis_cls"].return_value = run_analysis

    mocks["update_top_n_runs"].return_value = pd.DataFrame()

    error, best_model = evaler.evaluate(DummyConfig(algo="Algo"))

    assert error == pytest.approx(1.5)
    assert best_model is pipeline
    pipeline.train.assert_called_once_with(data="train_data")
    filer.save_validataion_data.assert_called_once()


def test_best_model_selected_across_folds(mocks):
    evaler = ExplicitEvaler(
        train=MagicMock(),
        optimization_metric=rmse,
        filer=MagicMock(),
        split_folds=2,
        predict_mode=True
    )

    fold1 = MagicMock(train="train1", test="test1")
    fold2 = MagicMock(train="train2", test="test2")
    evaler.train_test_splits = [fold1, fold2]

    mocks["get_model"].return_value = MagicMock()

    pipeline1 = MagicMock()
    pipeline2 = MagicMock()
    pipeline1.clone.return_value = pipeline1
    pipeline2.clone.return_value = pipeline2

    mocks["predict_pipeline"].side_effect = [pipeline1, pipeline2]

    mocks["predict"].return_value = MagicMock(to_df=lambda: pd.DataFrame())

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.side_effect = [
        pd.DataFrame({"mean": [2.0]}, index=["rmse"]),
        pd.DataFrame({"mean": [1.0]}, index=["rmse"]),
    ]
    mocks["run_analysis_cls"].return_value = run_analysis

    mocks["update_top_n_runs"].return_value = pd.DataFrame()

    error, best_model = evaler.evaluate(DummyConfig(algo="Algo"))

    expected_error = (2.0 + 1.0) / 2
    assert error == pytest.approx(expected_error)
    assert best_model is pipeline2


def test_maximize_metric_returns_inverted_score(mocks):
    evaler = ExplicitEvaler(
        train=MagicMock(),
        optimization_metric=rmse,
        filer=MagicMock(),
        minimize_error_metric_val=False
    )

    fold = MagicMock()
    fold.train = "train_data"
    fold.test = "test_data"
    evaler.train_test_splits = [fold]

    mocks["get_model"].return_value = MagicMock()

    pipeline = MagicMock()
    pipeline.clone.return_value = pipeline
    mocks["predict_pipeline"].return_value = pipeline

    recs = MagicMock()
    recs.to_df.return_value = pd.DataFrame()
    mocks["predict"].return_value = recs

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.return_value = \
        pd.DataFrame({"mean": [0.25]}, index=["rmse"])
    mocks["run_analysis_cls"].return_value = run_analysis

    mocks["update_top_n_runs"].return_value = pd.DataFrame()

    score, _ = evaler.evaluate(DummyConfig(algo="Algo"))

    # Since minimize_error_metric_val=False, returned value should be 1 - error
    assert score == pytest.approx(1 - 0.25)


def test_recommendation_mode_uses_topn_pipeline(mocks):
    evaler = ExplicitEvaler(
        train=MagicMock(),
        optimization_metric=rmse,
        filer=MagicMock(),
        predict_mode=False
    )

    fold = MagicMock()
    fold.train = "train_data"
    fold.test = "test_data"
    evaler.train_test_splits = [fold]

    mocks["get_model"].return_value = MagicMock()

    pipeline = MagicMock()
    pipeline.clone.return_value = pipeline
    mocks["topn_pipeline"].return_value = pipeline

    recs = MagicMock()
    recs.to_df.return_value = pd.DataFrame()
    mocks["recommend"].return_value = recs

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.return_value = \
        pd.DataFrame({"mean": [0.5]}, index=["rmse"])
    mocks["run_analysis_cls"].return_value = run_analysis

    mocks["update_top_n_runs"].return_value = pd.DataFrame()

    error, best_model = evaler.evaluate(DummyConfig(algo="Algo"))

    assert error == pytest.approx(0.5)
    assert best_model is pipeline


def test_external_validation_dataset(mocks):
    # Provide explicit validation data (no splitting)
    train = MagicMock()
    validation = MagicMock()
    filer = MagicMock()

    evaler = ExplicitEvaler(
        train=train,
        optimization_metric=rmse,
        filer=filer,
        validation=validation,
        split_folds=1,
        predict_mode=True
    )

    mocks["get_model"].return_value = MagicMock()

    pipeline = MagicMock()
    pipeline.clone.return_value = pipeline
    mocks["predict_pipeline"].return_value = pipeline

    recs = MagicMock()
    recs.to_df.return_value = pd.DataFrame()
    mocks["predict"].return_value = recs

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.return_value = \
        pd.DataFrame({"mean": [1.2]}, index=["rmse"])
    mocks["run_analysis_cls"].return_value = run_analysis

    mocks["update_top_n_runs"].return_value = pd.DataFrame()

    error, best_model = evaler.evaluate(DummyConfig(algo="Algo"))

    assert error == pytest.approx(1.2)
    assert best_model is pipeline


def test_top_n_runs_is_updated(mocks):
    train = MagicMock()
    filer = MagicMock()

    evaler = ExplicitEvaler(
        train=train,
        optimization_metric=rmse,
        filer=filer,
        split_folds=1,
        predict_mode=True
    )

    fold = MagicMock()
    fold.train = "train_data"
    fold.test = "test_data"
    evaler.train_test_splits = [fold]

    mocks["get_model"].return_value = MagicMock()

    pipeline = MagicMock()
    pipeline.clone.return_value = pipeline
    mocks["predict_pipeline"].return_value = pipeline

    recs = MagicMock()
    recs.to_df.return_value = pd.DataFrame()
    mocks["predict"].return_value = recs

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.return_value = \
        pd.DataFrame({"mean": [1.5]}, index=["rmse"])
    mocks["run_analysis_cls"].return_value = run_analysis

    df_top_n = pd.DataFrame()
    mocks["update_top_n_runs"].return_value = df_top_n

    error, best_model = evaler.evaluate(DummyConfig(algo="Algo"))

    mocks["update_top_n_runs"].assert_called_once()
    assert evaler.top_n_runs is df_top_n
