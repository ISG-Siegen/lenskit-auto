# LensKit-Auto

## Install

TODO

## How To Use

### Standard use-case

In the standard use-case you just need to call a single function to get the best performing model for your dataset.
It is either 

    find_best_implicit_configuration(train=train_split)

for the recommendation use-case or

    find_best_implicit_configuration(train=train_split)

for the prediction use-case

### Examples and Advanced Use-Cases

LensKit-Auto allows three application scenarios:

Note: All application scenarios apply to Top-N ranking prediction and rating prediction use-cases.

* **Scenario 1:** LensKit-Auto performs combined algorithm selection and hyperparameter optimization for a given
  dataset.
* **Scenario 2:** LensKit-Auto performs hyperparameter optimization on a single algorithm for a given dataset.
* **Scenario 3:** LensKit-Auto performs combined algorithm selection and hyperparameter optimization for a specified set
  of algorithms and/or different hyperparameter ranges for the provided dataset.

In order to take advantage of LensKit-Auto, a developer needs to read in a dataset.

    from lenskit.datasets import ML100K
    
    ml100k = ML100K('ml-100k') 
    ratings = ml100k.ratings
    ml100k.name = 'ml_100k'

Furthermore, it is suggested, that we take advantage of the Filer to control the LensKit-Auto output

    from lkauto.utils.filer import Filer
    
    filer = Filer('output/')

### Top-N ranking prediction

First, we need to split the data in a train and test split to evaluate our model. The train-test splits can be performed
based on data rows or user data. For the rating prediction example we are splitting the data based on user data.

    # User based data-split
    for i, tp in enumerate(xf.partition_users(mlsmall, 1, xf.SampleN(5))):
        train_split = tp.train.copy()
        test_split = tp.test.copy()

    # Fixme: INSERT SCENARIO CODE HERE

    # fit
    model.fit(train)
    # recommend
    recs = batch.recommend(algo=model, users=test['user'].unique(), n=5)

    # initialize RecListAnalysis
    rla = topn.RecListAnalysis()
    # add precision metric
    rla.add_metric(topn.ndcg)

    # compute scores
    scores = rla.compute(recs, test, include_missing=True)

#### Rating Prediction

First, we need to split the data in a train and test split to evaluate our model. The train-test splits can be performed
based on data rows or user data. For the rating prediction example we are splitting the data based on the data rows. The
Top-N ranking predicion example showcases the data-split based on user data.

    train_split, test_split = sample_rows(mlsmall, None, 25000)

    # Fixme: INSERT SCENARIO CODE HERE

    model.fit(train)
    predictions = model.predict(x_test)
    root_mean_square_error = rmse(predictions, y_test, missing='error')

#### Scenario 1

Scenario 1 describes the fully automated combined algorithm selection and hyperparameter optimization (CASH problem).
This scenario is recommended for inexperienced developers who have no or little experience in model selection.

LensKit-Auto performs the combined algorithm selection and hyperparameter optimization with a single function call.

    model, config = find_best_implicit_configuration(train=train_split, filer=filer)

The *find_best_implicit_configuration()* function call will return the best performing model, with tuned hyperparameters
and a configuration dictionary that contains all information about the model. In the Scenario 1 use-case the model is
chosen out of all LensKit algorithms with hyperparameters within the LensKit-Auto default  
hyperparameter range. We can use the model in the exact same way like a regular LensKit model. 

#### Scenario 2

In Senario 2 we are going to perform hyperparameter optimization on a single algorithm. First we need to define our
custom configuration space with just a single algorithm included.

    from ConfigSpace import Constant
    from lkauto.algorithms.item_knn import ItemItem
    
    # initialize ItemItem ConfigurationSpace
    cs = ItemItem.get_default_configspace()
    cs.add_hyperparameters([Constant("algo", "ItemItem")])
    # set a random seed for reproducible results
    cs.seed(42)

    # Provide the ItemItem ConfigurationSpace to the find_best_implicit_configuraiton function. 
    model, config = find_best_implicit_configuration(train=train_split, filer=filer, cs=cs)

The *find_best_implicit_configuration()* function call will return the best performing ItemItem model. Besides the
model, the *find_best_implicit_configuration()* function returns a configuration dictionary with all information about
the model.

#### Scenario 3

Scenario 3 describes the automated combined algorithm selection and hyperparameter optimization of a custom
configuration space. A developer that wants to use Scenario 3 needs to provide hyperparameter ranges for the
hyperparameter optimization process.

First, a parent-ConfigurationSpace needs to be initialized. All algorithm names need to be added to the
parent-ConfigurationSpace categorical *algo* hyperparameter.

    from ConfigSpace import ConfigurationSpace, Categorical
    
    # initialize ItemItem ConfigurationSpace
    parent_cs = ConfigurationSpace()
    # set a random seed for reproducible results
    parent_cs.seed(42)
    # add algorithm names as a constant
    parent_cs.add_hyperparameters([Categorical("algo", ["ItemItem", "UserUser"])])

Afterward, we need to build the *ItemItem* and *UserUser* sub-ConfigurationSpace.

We can use the default sub-ConfigurationSpace from LensKit-Auto and add it to the parent-ConfigurationSpace:

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

Or we can build our own ConfigurationSpace for a specific algorithm.

    from ConfigSpace import ConfigurationSpace
    from ConfigSpace import Integer, Float

    # first we initialize hyperparameter objects for all hyperparameters that we want to optimize
    min_nbrs = Integer('min_nbrs', bounds=(1, 50), default=1)
    min_sim = Float('min_sim', bounds=(0, 0.1), default=0)

    # Then, we initialize the sub-ConfigurationSpace and add the hyperparameters to it
    user_user_cs = ConfigurationSpace()
    user_user_cs.add_hyperparameters([min_nbrs, min_sim])

    # Last, we add the user_user_cs to the parent-ConfigurationSpace 

     parent_cs.add_configuration_space(
            prefix="UserUser",
            delimiter=":",
            configuration_space=user_user_cs,
            parent_hyperparameter={"parent": parent_cs["algo"], "value": "UserUser"},
        )

After creating the parent-ConfigurationSpace, we can use it in the same way like Scenario 2

    # Provide the parent-ConfigurationSpace to the find_best_implicit_configuraiton function. 
    model, config = find_best_implicit_configuration(train=train_split, filer=filer, cs=parent_cs)

