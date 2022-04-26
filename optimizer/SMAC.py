import numpy as np
from lenskit.metrics.predict import rmse
from lenskit.crossfold import sample_rows
from lenskit.algorithms.basic import Fallback, Bias
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.als import BiasedMF as ALSBiasedMF
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.tf import BiasedMF as TFBiasedMF, BPR, IntegratedBiasMF
from lenskit.algorithms.svd import BiasedSVD
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from lenskit.optimizer.missing_config_parameters import biasedMFAlgorithm_add_to_config_space, \
    bprAlgorithm_add_to_config_space, \
    integratedBiasMF_add_to_config_space


class AutoLensKit:

    def __init__(self):
        self.train = None
        self.validation_split = None
        self.x_validation_split = None
        self.y_validation_split = None
        self.cs = ConfigurationSpace()
        regressor = CategoricalHyperparameter('regressor', ['ItemItemKNN',
                                                            'Bias',
                                                            'FunkSVD',
                                                            'UserUserKNN',
                                                            'BiasedSVD',
                                                            'TFBiasedMF',
                                                            'BPR',
                                                            'IntegratedBiasMF',
                                                            'ALS'])
        self.cs.add_hyperparameter(regressor)
        self.__get_configuration_space()

    def __get_configuration_space(self):
        ItemItem.add_to_config_space(self.cs)
        Bias.add_to_config_space(self.cs)
        FunkSVD.add_to_config_space(self.cs)
        UserUser.add_to_config_space(self.cs)
        ALSBiasedMF.add_to_config_space(self.cs)
        BiasedSVD.add_to_config_space(self.cs)
        biasedMFAlgorithm_add_to_config_space(self.cs)
        bprAlgorithm_add_to_config_space(self.cs)
        integratedBiasMF_add_to_config_space(self.cs)

    def __get_model_from_cs(self, config_space):
        model = None

        current_model = config_space.get('regressor')
        print(current_model)

        if current_model == 'ItemItemKNN':
            model = ItemItem(nnbrs=config_space['Nnbrs'],
                             min_nbrs=config_space['min_nbrs'],
                             min_sim=config_space['min_sim'],
                             center=config_space['center'],
                             aggregate=config_space['aggregate'])

        if current_model == 'Bias':
            model = Bias(items=config_space['items'],
                         users=config_space['users'])

        if current_model == 'FunkSVD':
            model = FunkSVD(features=100,
                            lrate=config_space['lrate'],
                            reg=config_space['reg'],
                            damping=config_space['damping'])

        if current_model == 'UserUserKNN':
            model = UserUser(nnbrs=config_space['user_user_Nnbrs'],
                             min_nbrs=config_space['user_user_min_nbrs'],
                             min_sim=config_space['user_user_min_sim'],
                             center=config_space['user_user_center'],
                             aggregate=config_space['user_user_aggregate'])

        if current_model == 'ALS':
            model = ALSBiasedMF(features=100,
                                reg=config_space['als_regularization'],
                                damping=config_space['als_damping'],
                                method=config_space['als_method'])

        if current_model == 'BiasedSVD':
            model = BiasedSVD(100,
                              damping=config_space['bias_svd_damping'],
                              algorithm=config_space['bias_svd_algorithm'])

        if current_model == 'TFBiasedMF':  # not working
            model = TFBiasedMF(100,
                               damping=config_space['biased_mf_damping'],
                               reg=config_space['biased_mf_reg'],
                               batch_size=config_space['biased_mf_batch_size'])

        if current_model == 'BPR':
            model = BPR(100,
                        reg=config_space['bpr_reg'],
                        neg_weight=config_space['bpr_neg_weight'],
                        batch_size=config_space['bpr_batch_size'])

        if current_model == 'IntegratedBiasMF':  # Funktioniert nicht
            model = IntegratedBiasMF(100,
                                     reg=config_space['integrated_bias_mf_reg'],
                                     bias_reg=config_space['integrated_bias_mf_bias_reg'],
                                     batch_size=config_space['integrated_bias_mf_batch_size'])

        return model

    def __evaluate(self, configuration_space):
        model = self.__get_model_from_cs(configuration_space)
        model.fit(self.train)
        predictions = model.predict(self.x_validation_split)
        root_mean_square_error = rmse(predictions, self.y_validation_split, missing='ignore')
        print(root_mean_square_error)
        return root_mean_square_error

    def find_best_configuration(self, train, time_limit_in_sec=14400):
        self.train, self.validation_split = sample_rows(train, None, 15000)
        self.x_validation_split = self.validation_split.copy()
        self.x_validation_split.drop('rating', inplace=True, axis=1)
        self.y_validation_split = self.validation_split[['rating']].iloc[:, 0]

        scenario = Scenario({
            'run_obj': 'quality',
            'wallclock_limit': time_limit_in_sec,
            'cs': self.cs,
            'deterministic': False
        })

        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(42),
                        tae_runner=self.__evaluate)
        try:
            smac.optimize()
        finally:
            incumbent = smac.solver.incumbent

        print('best config found:')
        print(incumbent)

        model = self.__get_model_from_cs(incumbent)
        return model
