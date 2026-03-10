import os
from pathlib import Path
from datetime import datetime

from deepcave.plugins.hyperparameter.ablation_paths import AblationPaths
from deepcave.plugins.hyperparameter.configuration_cube import ConfigurationCube
from deepcave.plugins.hyperparameter.pdp import PartialDependencies
from deepcave.plugins.objective.cost_over_time import CostOverTime
from deepcave.plugins.objective.pareto_front import ParetoFront
from deepcave.plugins.summary.footprint import FootPrint
from deepcave.runs.converters.smac3v2 import SMAC3v2Run


class Visualizer:

    def save_visualization_figures(self, run_path):
        run = SMAC3v2Run.from_path(Path(run_path))
        objective_id_cost = run.get_objective_ids()[0]
        objective_id_time = run.get_objective_ids()[1]
        budget_id = run.get_budget_ids()[-1]

        if not os.path.exists(Path(run_path) / 'visualization'):
            os.mkdir(Path(run_path) / "visualization")
        figure_save_path = Path(Path(run_path) / "visualization")

        hp_names = [hp.name for hp in run.configspace.values()]
        print(f"configurations: \n{hp_names}")

        time_start = datetime.now()
        self.save_fig_partial_dependencies(run, objective_id_cost, budget_id, figure_save_path)
        self.save_fig_cost_over_time(run, objective_id_cost, budget_id, figure_save_path)
        self.save_fig_config_cube(run, objective_id_cost, budget_id, figure_save_path)
        self.save_fig_pareto_front(run, objective_id_cost, objective_id_time, budget_id, figure_save_path)
        self.save_fig_ablation_paths(run, objective_id_cost, objective_id_time, budget_id, figure_save_path)
        self.save_configuration_footprint(run, objective_id_cost, budget_id, figure_save_path)

        stop_time = datetime.now()

        print(f"Generating plots took {stop_time - time_start}")

    def save_fig_partial_dependencies(self, run, cost_id, budget_id, save_path):
        plugin = PartialDependencies()
        inputs = plugin.generate_inputs(
            hyperparameter_name_1="algo",
            hyperparameter_name_2=None,
            objective_id=cost_id,
            budget_id=budget_id,
            show_confidence=False,
            show_ice=True,
        )
        outputs = plugin.generate_outputs(run, inputs)

        figure = plugin.load_outputs(run, inputs, outputs)
        figure.write_image(save_path / "partial_dependencies.png")

    def save_fig_cost_over_time(self, run, cost_id, budget_id, save_path):
        plugin = CostOverTime()
        inputs = plugin.generate_inputs(
            objective_id=cost_id,
            budget_id=budget_id,
            xaxis="Time",
            show_runs=True,
            show_groups=True
        )
        outputs = plugin.generate_outputs(run, inputs)

        if not hasattr(run, "__iter__"):
            run = [run]
        figure = plugin.load_outputs(run, inputs, outputs)
        figure.write_image(save_path / "cost_over_time.png")

    def save_fig_config_cube(self, run, cost_id, budget_id, save_path):
        plugin = ConfigurationCube()
        inputs = plugin.generate_inputs(objective_id=cost_id,
                                        budget_id=budget_id,
                                        hyperparameter_names=["algo"],
                                        n_configs=20)

        outputs = plugin.generate_outputs(run, inputs)
        figure = plugin.load_outputs(run, inputs, outputs)
        figure.write_image(save_path / "config_cube.png")

    def save_fig_pareto_front(self, run, cost_id, time_id, budget_id, save_path):
        plugin = ParetoFront()
        inputs = plugin.generate_inputs(objective_id_1=cost_id,
                                        objective_id_2=time_id,
                                        show_all=True,
                                        show_runs=True,
                                        show_error=False,
                                        show_groups=True,
                                        budget_id=budget_id)
        outputs = plugin.generate_outputs(run, inputs)

        if not hasattr(run, "__iter__"):
            run = [run]
        figure = plugin.load_outputs(run, inputs, outputs)
        figure.write_image(save_path / "pareto_front.png")

    def save_fig_ablation_paths(self, run, cost_id, time_id, budget_id, save_path):
        plugin = AblationPaths()
        inputs = plugin.generate_inputs(n_hps=19,
                                        show_confidence=True,
                                        budget_id=budget_id,
                                        objective_id1=cost_id,
                                        objective_id2=time_id,
                                        n_trees=100)
        outputs = plugin.generate_outputs(run, inputs)
        figures = plugin.load_outputs(run, inputs, outputs)
        for idx, figure in enumerate(figures):
            figure.write_image(save_path / f"ablation_paths_{idx}.png")

    def save_configuration_footprint(self, run, cost_id, budget_id, save_path):
        plugin = FootPrint()
        inputs = plugin.generate_inputs(objective_id=cost_id,
                                        budget_id=budget_id,
                                        details=10,
                                        show_supports=True,
                                        show_borders=True)
        outputs = plugin.generate_outputs(run, inputs)
        figures = plugin.load_outputs(run, inputs, outputs)
        for idx, figure in enumerate(figures):
            if idx == 0:
                figure.write_image(save_path / "cost_configuration_footprint_performance.png")
            if idx == 1:
                figure.write_image(save_path / "cost_configuration_footprint_coverage.png")
