import unittest
import os

from pathlib import Path
from lkauto.utils.filer import Filer


class MyTestCase(unittest.TestCase):

    def test_save_visualization_figures(self):
        run_path = Path("visualization_test_dir/test_data/0")
        filer = Filer()

        filer.save_visualization_figures(run_path=run_path)

        self.assertTrue((Path(run_path) / "visualization").exists())

        figures_path = Path(run_path) / "visualization"

        self.assertTrue(os.path.exists(figures_path / "ablation_paths_0.png"))
        self.assertTrue(os.path.exists(figures_path / "ablation_paths_1.png"))
        self.assertTrue(os.path.exists(figures_path / "config_cube.png"))
        self.assertTrue(os.path.exists(figures_path / "cost_configuration_footprint_performance.png"))
        self.assertTrue(os.path.exists(figures_path / "cost_configuration_footprint_coverage.png"))
        self.assertTrue(os.path.exists(figures_path / "cost_over_time.png"))
        self.assertTrue(os.path.exists(figures_path / "pareto_front.png"))
        self.assertTrue(os.path.exists(figures_path / "partial_dependencies.png"))

        for filename in os.listdir(figures_path):
            file_path = os.path.join(figures_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(figures_path)

