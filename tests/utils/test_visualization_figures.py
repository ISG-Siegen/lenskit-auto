import pytest
import os

from pathlib import Path
from lkauto.utils.visualizer import Visualizer


class TestSaveVisualizationFigures():

    def test_save_visualization_figures(self):
        current_dir = Path(__file__).resolve().parent
        run_path = current_dir / "visualization_test_dir" / "test_data" / "0"
        visualizer = Visualizer()

        visualizer.save_visualization_figures(run_path=run_path)

        assert (Path(run_path) / "visualization").exists()

        figures_path = Path(run_path) / "visualization"

        assert os.path.exists(figures_path / "ablation_paths_0.png")
        assert os.path.exists(figures_path / "ablation_paths_1.png")
        assert os.path.exists(figures_path / "config_cube.png")
        assert os.path.exists(figures_path / "cost_configuration_footprint_performance.png")
        assert os.path.exists(figures_path / "cost_configuration_footprint_coverage.png")
        assert os.path.exists(figures_path / "cost_over_time.png")
        assert os.path.exists(figures_path / "pareto_front.png")
        assert os.path.exists(figures_path / "partial_dependencies.png")

        for filename in os.listdir(figures_path):
            file_path = os.path.join(figures_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(figures_path)

