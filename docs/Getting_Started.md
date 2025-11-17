# LensKit-Auto

LensKit-Auto is built as a wrapper around the Python [LensKit](https://lkpy.readthedocs.io/en/stable/)
recommender-system library. It automates algorithm selection and hyper parameter optimization and can build ensemble
models based on LensKit models.

## Install

Lenskit-auto can be installed on both Linux and Windows:

### Pip Install:

```bash
pip install lenskit-auto
```

### Conda Install:

```bash
conda install -c conda-forge lenskit-auto
```

## Getting Started

### Standard use-case

In the standard use-case you just need to call a single function to get the best performing model for your dataset. It
is either

```python
from lkauto.lkauto import get_best_recommender_model

get_best_recommender_model(train=train_split)
```

for the recommendation use-case or

```python
from lkauto.lkauto import get_best_prediction_model

get_best_prediction_model(train=train_split)
```

for the prediction use-case

### Examples and Advanced Use-Cases

LensKit-Auto allows three application scenarios:

Note: All application scenarios apply to Top-N ranking prediction and rating prediction use-cases.

* **Scenario 1:** LensKit-Auto performs combined algorithm selection and hyperparameter optimization for a given dataset.
* **Scenario 2:** LensKit-Auto performs hyperparameter optimization on a single algorithm for a given dataset.
* **Scenario 3:** LensKit-Auto performs combined algorithm selection and hyperparameter optimization for a specified set of algorithms and/or different hyperparameter ranges for the provided dataset.

In order to take advantage of LensKit-Auto, a developer needs to read in a dataset.

```python
from lenskit.data import load_movielens

ml100k = load_movielens('path_to_file')
```

Furthermore, it is suggested, that we take advantage of the Filer to control the LensKit-Auto output

```python
from lkauto.utils.filer import Filer

filer = Filer('output/')
```

### Top-N ranking prediction

First, we need to split the data in a train and test split to evaluate our model. The train-test splits can be performed based on data rows or user data. For the Top-N ranking prediction example we are splitting the data based on user data.

```python
from lenskit.splitting import crossfold_users, SampleN
from lenskit.batch import recommend
from lenskit.metrics import RunAnalysis, NDCG
from lkauto.lkauto import get_best_recommender_model

# User based data-split
for split in crossfold_users(ml100k, 1, SampleN(5)):
    train_split = split.train
    test_split = split.test

    # INSERT SCENARIO CODE HERE
    # See Scenario 1, 2, and 3 sections below
   
    # recommend
    recs = recommend(model, users=tuple(test_split.keys()), n=5, n_jobs=1)

    # initialize analysis
    analysis = RunAnalysis()
    # add ndcg metric
    analysis.add_metric(NDCG())
    # evaluate recommendations against the test interactions
    scores = analysis.measure(recs, test_split)
```

### Rating Prediction

First, we need to split the data in a train and test split to evaluate our model. The train-test splits can be performed based on data rows or user data. For the rating prediction example we are splitting the data based on the data rows. The Top-N ranking predicion example showcases the data-split based on user data.

```python
from lenskit.splitting import sample_records
from lenskit.metrics import RunAnalysis, RMSE
from lkauto.lkauto import get_best_prediction_model
from lkauto.utils.pred_and_rec_functions import predict

# record-based split: hold out 25k random interactions
split = sample_records(ml100k, size=25_000)
train_split = split.train
test_split = split.test

# INSERT SCENARIO CODE HERE
# See Scenario 1, 2, and 3 sections below

# generate rating predictions for the held-out interactions
# here we need to use the wrapper predict() and not the lenskit.batch.predict()
predictions = predict(model, test_split)

# initialize analysis
analysis = RunAnalysis()
# add rmse metric
analysis.add_metric(RMSE())
# evaluate recommendations against the test interactions
scores = analysis.measure(predictions, test_split)
```
### Application Scenarios

The following scenarios can be used in both Top-N ranking prediction and rating prediction use cases. Simply insert the scenario code where indicated by `# INSERT SCENARIO CODE HERE` in the examples above.

#### Scenario 1: Fully Automated Model Selection & Hyperparameter Optimization

Scenario 1 describes the fully automated combined algorithm selection and hyperparameter optimization (CASH problem).
This scenario is recommended for inexperienced developers who have no or little experience in model selection.

LensKit-Auto performs the combined algorithm selection and hyperparameter optimization with a single function call.

```python
model, config = get_best_recommender_model(train=train_split, filer=filer)
```

Note: As described above, the *get_best_recommender_model()* is used for Top-N ranking prediction. If you want to find a predictor instead of a recommender, replace the function call with *get_best_prediction_model()*

The *get_best_recommender_model()* or *get_best_prediction_model()* function call will return the best performing model, with tuned hyperparameters and a configuration dictionary that contains all information about the model. In the Scenario 1 use-case the model is chosen out of all LensKit algorithms with hyperparameters within the LensKit-Auto default hyperparameter range. 


We can use the model in the exact same way like a regular LensKit model. (nicht mehr true sondern:
For rating prediction with ensemble models, use the *predict()* wrapper function from *lkauto.utils.pred_and_rec_functions* instead of *lenskit.batch.predict()* directly (as shown in the Rating Prediction example above).
)

#### Scenario 2: Single-Algorithm Hyperparameter Optimization

In Scenario 2 we are going to perform hyperparameter optimization on a single algorithm. First we need to define our custom configuration space with just a single algorithm included.

```python
from ConfigSpace import Constant
from lkauto.algorithms.item_knn import ItemItem

# initialize ItemItem ConfigurationSpace
cs = ItemItem.get_default_configspace()
# add algorithm name as a constant (in this case ItemItem algorithm)
cs.add([Constant("algo", "ItemItem")])
# set a random seed for reproducible results
cs.seed(42)

# Provide the ItemItem ConfigurationSpace to the get_best_recommender_model function. 
model, config = get_best_recommender_model(train=train_split, filer=filer, cs=cs)
```

Note: As described above, the *get_best_recommender_model()* is used for Top-N ranking prediction. If you want to find a predictor instead of a recommender, replace the function call with *get_best_prediction_model()*

The *get_best_recommender_model()* or *get_best_prediction_model()* function call will return the best performing ItemItem model. Besides the model, the *get_best_recommender_model()* function returns a configuration dictionary with all information about the model.

#### Scenario 3: Custom Search Space Model Selection & Hyperparameter Optimization

Scenario 3 describes the automated combined algorithm selection and hyperparameter optimization of a custom configuration space. A developer that wants to use Scenario 3 needs to provide hyperparameter ranges for the hyperparameter optimization process.

First, a parent-ConfigurationSpace needs to be initialized. All algorithm names need to be added to the parent-ConfigurationSpace categorical *algo* hyperparameter.

```python
from ConfigSpace import ConfigurationSpace, Categorical

# initialize parent ConfigurationSpace
parent_cs = ConfigurationSpace()
# set a random seed for reproducible results
parent_cs.seed(42)
# add algorithm names as a constant
parent_cs.add([Categorical("algo", ["ItemItem", "UserUser"])])
```

Afterward, we need to build the *ItemItem* and *UserUser* sub-ConfigurationSpace.

We can use the default sub-ConfigurationSpace from LensKit-Auto and add it to the parent-ConfigurationSpace:

```python
from lkauto.algorithms.item_knn import ItemItem

# get default ItemItem ConfigurationSpace
item_item_cs = ItemItem.get_default_configspace()

# Add sub-ConfigurationSpace to parent-ConfigurationSpace
parent_cs.add_configuration_space(
    prefix="ItemItem",
    delimiter=":",
    configuration_space=item_item_cs,
    parent_hyperparameter={"parent": parent_cs["algo"], "value": "ItemItem"},
)
```

Or we can build our own ConfigurationSpace for a specific algorithm.

```python
from ConfigSpace import ConfigurationSpace
from ConfigSpace import Integer, Float, Constant

# first we initialize hyperparameter objects for all hyperparameters that we want to optimize
nnbrs = Constant('nnbrs', 1000)
min_nbrs = Integer('min_nbrs', bounds=(1, 50), default=1)
min_sim = Float('min_sim', bounds=(0, 0.1), default=0)

# Then, we initialize the sub-ConfigurationSpace and add the hyperparameters to it
user_user_cs = ConfigurationSpace()
user_user_cs.add([nnbrs, min_nbrs, min_sim])

# Last, we add the user_user_cs to the parent-ConfigurationSpace 

parent_cs.add_configuration_space(
    prefix="UserUser",
    delimiter=":",
    configuration_space=user_user_cs,
    parent_hyperparameter={"parent": parent_cs["algo"], "value": "UserUser"},
)
```

After creating the parent-ConfigurationSpace, we can use it in the same way like Scenario 2

```python
# Provide the parent-ConfigurationSpace to the get_best_recommender_model function. 
model, config = get_best_recommender_model(train=train_split, filer=filer, cs=parent_cs)
```

Note: As described above, the *get_best_recommender_model()* is used for Top-N ranking prediction. If you want to find a predictor instead of a recommender, replace the function call with *get_best_prediction_model()*

