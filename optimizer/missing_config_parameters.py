from ConfigSpace.conditions import InCondition, OrConjunction
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace

def biasedMFAlgorithm_add_to_config_space(cs: ConfigurationSpace):
    """
           adds parameters to ConfigurationSpace
    """
    try:
        regressor = cs["regressor"]
        if 'TFBiasedMF' not in regressor.choices:
            return

        biased_mf_damping = UniformFloatHyperparameter('biased_mf_damping', lower=0.0, upper=20, default_value=5)
        biased_mf_reg = UniformFloatHyperparameter('biased_mf_reg', lower=0.0, upper=0.99, default_value=0.02)
        biased_mf_batch_size = UniformIntegerHyperparameter('biased_mf_batch_size', lower=100, upper=1000000,
                                                            default_value=10000)

        cs.add_hyperparameters([biased_mf_damping, biased_mf_reg, biased_mf_batch_size])

        cs.add_condition(InCondition(biased_mf_damping, regressor, values=['TFBiasedMF']))
        cs.add_condition(InCondition(biased_mf_reg, regressor, values=['TFBiasedMF']))
        cs.add_condition(InCondition(biased_mf_batch_size, regressor, values=['TFBiasedMF']))

    except:
        return

def bprAlgorithm_add_to_config_space(cs: ConfigurationSpace):
    """
           adds parameters to ConfigurationSpace
    """
    try:
        regressor = cs["regressor"]
        if 'BPR' not in regressor.choices:
            return

        bpr_reg = UniformFloatHyperparameter('bpr_reg', lower=0.0, upper=0.99, default_value=0.02)
        bpr_neg_weight = CategoricalHyperparameter('bpr_neg_weight', choices=[True, False])
        bpr_batch_size = UniformIntegerHyperparameter('bpr_batch_size', lower=100, upper=1000000,
                                                      default_value=10000)

        cs.add_hyperparameters([bpr_reg, bpr_neg_weight, bpr_batch_size])

        cs.add_condition(InCondition(bpr_reg, regressor, values=['BPR']))
        cs.add_condition(InCondition(bpr_neg_weight, regressor, values=['BPR']))
        cs.add_condition(InCondition(bpr_batch_size, regressor, values=['BPR']))

    except:
        return

def integratedBiasMF_add_to_config_space(cs: ConfigurationSpace):
    """
           adds parameters to ConfigurationSpace
    """
    try:
        regressor = cs["regressor"]
        if 'BPR' not in regressor.choices:
            return

        integrated_bias_mf_bias_reg = UniformFloatHyperparameter('integrated_bias_mf_bias_reg', lower=0.0, upper=0.99,
                                                                 default_value=0.2)
        integrated_bias_mf_reg = UniformFloatHyperparameter('integrated_bias_mf_reg', lower=0.0, upper=0.99,
                                                            default_value=0.02)
        integrated_bias_mf_batch_size = UniformIntegerHyperparameter('integrated_bias_mf_batch_size', lower=100,
                                                                     upper=1000000,
                                                                     default_value=10000)

        cs.add_hyperparameters([integrated_bias_mf_bias_reg, integrated_bias_mf_reg, integrated_bias_mf_batch_size])

        cs.add_condition(InCondition(integrated_bias_mf_bias_reg, regressor, values=['IntegratedBiasMF']))
        cs.add_condition(InCondition(integrated_bias_mf_reg, regressor, values=['IntegratedBiasMF']))
        cs.add_condition(InCondition(integrated_bias_mf_batch_size, regressor, values=['IntegratedBiasMF']))

    except:
        return
