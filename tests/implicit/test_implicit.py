import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from lkauto.implicit.implicit_evaler import ImplicitEvaler


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def ndcg():
    pass


ndcg.__name__ = "ndcg"


class DummyConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


# ----------------------------------------------------------------------
# Global dependency patching fixture
# ----------------------------------------------------------------------

@pytest.fixture
def mocks():
    """
    Patch all external dependencies used by ImplicitEvaler,
    INCLUDING validation_split to prevent LensKit execution.
    """
    with patch("lkauto.implicit.implicit_evaler.validation_split") as validation_split, \
         patch("lkauto.implicit.implicit_evaler.get_model_from_cs") as get_model, \
         patch("lkauto.implicit.implicit_evaler.predict_pipeline") as predict_pipeline, \
         patch("lkauto.implicit.implicit_evaler.topn_pipeline") as topn_pipeline, \
         patch("lkauto.implicit.implicit_evaler.predict") as predict, \
         patch("lkauto.implicit.implicit_evaler.recommend") as recommend, \
         patch("lkauto.implicit.implicit_evaler.RunAnalysis") as run_analysis_cls:

        validation_split.return_value = []

        yield {
            "validation_split": validation_split,
            "get_model": get_model,
            "predict_pipeline": predict_pipeline,
            "topn_pipeline": topn_pipeline,
            "predict": predict,
            "recommend": recommend,
            "run_analysis_cls": run_analysis_cls,
        }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_single_fold_prediction_evaluation(mocks):
    train = MagicMock()
    filer = MagicMock()

    evaler = ImplicitEvaler(
        train=train,
        optimization_metric=ndcg,
        filer=filer,
        split_folds=1,
        predict_mode=True
    )

    fold = MagicMock()
    fold.train = "train_data"
    fold.test = "test_data"
    evaler.val_fold_indices = [fold]

    mocks["get_model"].return_value = MagicMock()

    pipeline = MagicMock()
    pipeline.clone.return_value = pipeline
    mocks["predict_pipeline"].return_value = pipeline

    recs = MagicMock()
    recs.to_df.return_value = pd.DataFrame(
        {"user": [1], "item": [42], "score": [0.9]}
    )
    mocks["predict"].return_value = recs

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.return_value = \
        pd.DataFrame({"mean": [0.8]}, index=["ndcg"])
    mocks["run_analysis_cls"].return_value = run_analysis

    error, best_model = evaler.evaluate(DummyConfig(algo="Algo"))

    # ndcg - 1
    assert error == pytest.approx(-0.2)
    assert best_model is pipeline
    pipeline.train.assert_called_once_with("train_data")
    filer.save_validataion_data.assert_called_once()


def test_best_model_selected_across_folds(mocks):
    evaler = ImplicitEvaler(
        train=MagicMock(),
        optimization_metric=ndcg,
        filer=MagicMock(),
        split_folds=2,
        predict_mode=True
    )

    fold1 = MagicMock(train="train1", test="test1")
    fold2 = MagicMock(train="train2", test="test2")
    evaler.val_fold_indices = [fold1, fold2]

    mocks["get_model"].return_value = MagicMock()

    pipeline1 = MagicMock()
    pipeline2 = MagicMock()
    pipeline1.clone.return_value = pipeline1
    pipeline2.clone.return_value = pipeline2

    mocks["predict_pipeline"].side_effect = [pipeline1, pipeline2]
    mocks["predict"].return_value = MagicMock(to_df=lambda: pd.DataFrame())

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.side_effect = [
        pd.DataFrame({"mean": [0.4]}, index=["ndcg"]),
        pd.DataFrame({"mean": [0.7]}, index=["ndcg"]),
    ]
    mocks["run_analysis_cls"].return_value = run_analysis

    error, best_model = evaler.evaluate(DummyConfig(algo="Algo"))

    # best fold is second → 0.7 - 1
    assert error == pytest.approx(-0.3)
    assert best_model is pipeline2


def test_maximize_metric_returns_inverted_score(mocks):
    evaler = ImplicitEvaler(
        train=MagicMock(),
        optimization_metric=ndcg,
        filer=MagicMock(),
        minimize_error_metric_val=False
    )

    fold = MagicMock(train="train_data", test="test_data")
    evaler.val_fold_indices = [fold]

    mocks["get_model"].return_value = MagicMock()

    pipeline = MagicMock()
    pipeline.clone.return_value = pipeline
    mocks["predict_pipeline"].return_value = pipeline

    mocks["predict"].return_value = MagicMock(to_df=lambda: pd.DataFrame())

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.return_value = \
        pd.DataFrame({"mean": [0.6]}, index=["ndcg"])
    mocks["run_analysis_cls"].return_value = run_analysis

    score, _ = evaler.evaluate(DummyConfig(algo="Algo"))

    # original error = 0.6 - 1 = -0.4 → inverted
    assert score == pytest.approx(1 - (-0.4))


def test_recommendation_mode_uses_topn_pipeline(mocks):
    evaler = ImplicitEvaler(
        train=MagicMock(),
        optimization_metric=ndcg,
        filer=MagicMock(),
        predict_mode=False
    )

    fold = MagicMock(train="train_data", test="test_data")
    evaler.val_fold_indices = [fold]

    mocks["get_model"].return_value = MagicMock()

    pipeline = MagicMock()
    pipeline.clone.return_value = pipeline
    mocks["topn_pipeline"].return_value = pipeline

    recs = MagicMock()
    recs.to_df.return_value = pd.DataFrame()
    mocks["recommend"].return_value = recs

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.return_value = \
        pd.DataFrame({"mean": [0.5]}, index=["ndcg"])
    mocks["run_analysis_cls"].return_value = run_analysis

    error, best_model = evaler.evaluate(DummyConfig(algo="Algo"))

    assert error == pytest.approx(-0.5)
    assert best_model is pipeline
    mocks["recommend"].assert_called_once()


def test_external_validation_dataset(mocks):
    train = MagicMock()
    validation = MagicMock()
    filer = MagicMock()

    evaler = ImplicitEvaler(
        train=train,
        optimization_metric=ndcg,
        filer=filer,
        validation=validation,
        split_folds=1,
        predict_mode=True
    )

    mocks["get_model"].return_value = MagicMock()

    pipeline = MagicMock()
    pipeline.clone.return_value = pipeline
    mocks["predict_pipeline"].return_value = pipeline

    mocks["predict"].return_value = MagicMock(to_df=lambda: pd.DataFrame())

    run_analysis = MagicMock()
    run_analysis.measure.return_value.list_summary.return_value = \
        pd.DataFrame({"mean": [0.9]}, index=["ndcg"])
    mocks["run_analysis_cls"].return_value = run_analysis

    error, best_model = evaler.evaluate(DummyConfig(algo="Algo"))

    assert error == pytest.approx(-0.1)
    assert best_model is pipeline
