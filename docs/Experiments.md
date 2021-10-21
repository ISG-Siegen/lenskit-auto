# Experiments

## Default Configuration Spaces
LensKit-Auto provides default configuration spaces for all of LensKit's algorithms. 
All default configurations are based on the original algorithm papers documented by the 
[LensKit library](https://lkpy.readthedocs.io/en/stable/algorithms.html).

In cases, where the original papers do not provide information on the configuration space, we evaluated the default 
configvalues in extensive experiments. 

LensKit rating prediciont algorithms are evaluated on [70 different datasets](https://github.com/ISG-Siegen/recsys-dataloader). 
We took the best performing algorithm hyperparameter configuration for each dataset and used them as indicators for the hyperparameter range. 
We rounded the lower and upper bound of the configurations to the closest number to the power of ten or two. 

LensKit Top-N recommendation algorithms are evaluated on [14 different datasets](https://recpack.froomle.ai/recpack.datasets.html)
Again, we took the best performing algorithm hyperparameter configuration for each dataset and used them as indicators for the hyperparameter range.
Then we rounded the lower and upper bound of the configurations to the closest number to the power of ten or two. 

