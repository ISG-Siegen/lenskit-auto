# LensKit-Auto

## Installing

Once LensKit-Auto is public, a conda installation will be provided

For now, we can install all dependencies using the provided environment.yml by calling

    conda env create -f environment.yml

## Examples

LensKit-Auto allows three application scenarios: 

* **Scenario 1:** LensKit-Auto performs combined algorithm selection and hyperparameter optimization for a provided dataset. 
* **Scenario 2:** LensKit-Auto performs combined algorithm selection and hyperparameter optimization for a specified 
  set of algorithms and/or different value ranges for hyperparameters for the provided dataset.
* **Scenario 3:** LensKit-Auto performs hyperparameter optimization on a single algorithm for a given dataset.

All application scenarios apply to Top-N ranking prediction and rating prediction use-cases.

In order to take advantage of LensKit-Auto, a developer needs to read in a dataset.
```python
mlsmall = MovieLens('data/ml-latest-small').ratings
mlsmall.name = 'ml_small'
```

Furthermore, it is suggested, that a developer takes advantage of the Filer to control the LensKit-Auto output

```python    
filer = Filer('output/')
```

### Top-N ranking prediction

First, we need to split the data in a train and test split to evaluate our model.
The train-test splits can be performed based on data rows or user data. For the rating prediction example we are
splitting the data based on user data. 

```python
# User based data-split
for i, tp in enumerate(xf.partition_users(mlsmall, 1, xf.SampleN(5))):
    train = tp.train.copy()
    test = tp.test.copy()
```

#### Scenario 1

Scenario 1 describes the fully automated combined algorithm selection and hyperparameter optimization. This 
application scenario is recommended for inexperienced developers who have no ixerience in setting up a 
configuration space. 


We perform combined algorithm selection and hyperparameter optimization with LensKit-Auto

```python
model, config = find_best_implicit_configuration(train, filer=filer)
```

The developer will get a model and a configuration dictionary in return. The default search time is four hours. 
Afterward, the developer can handle the returned model as any other LensKit Recommender model.

```python
# fit
model.fit(train)
# recommend
recs = batch.recommend(algo=model, users=test['user'].unique(), n=5)

# initialize RecListAnalysis
rla = topn.RecListAnalysis()
# add precision metric
rla.add_metric(topn.precision)

# compute scores
scores = rla.compute(recs, test, include_missing=True)
```

#### Scenario 2

Scenario 2 describes the automated combined algorithm selection and hyperparameter optimization of a custom 
configuration space. A developer that wants to use application Scenario 2 has to be able to define their own 
custom configuration space in order to provide it as an input parameter. Apart from defining a custom 
configuration space, the usage of LensKit-Auto is the same.

First, a configuration space needs to be initialized. All included algorithms need to be added to the regressor 
list of the configuration space. 

```python
# initialize ConfigurationSpace
cs = ConfigurationSpace()
# define ItemItem and UserUser as set of algorithms
regressor = CategoricalHyperparameter('regressor', ['ItemItem', 'UserUser'])
# add the ItemItem and UserUser set to the ConfigurationSpace cs
cs.add_hyperparameter(regressor)
```

Afterward, we need to add the hyperparameters for each algorithm to the configuration space.
We can use the pre defined configuration space of LensKit (Showcased on the ItemItem case):

```python
# Get a list of all ItemItem parameters of the provided ItemItem.get_default_configspace_hyperparameters() function
hyperparameter_list = ItemItem.get_default_configspace_hyperparameters()
# add all hyperparameters of hyperparameterlist to the ConfigurationSpace
cs.add_hyperparameters(hyperparameter_list)
# Set the condition that all ItemItem hyperparameters are bound to the ItemItem algorithm
for hyperparameter in hyperparameter_list:
    cs.add_condition(InCondition(hyperparameter, regressor, values=['ItemItem']))
```

Or we can define the configuration space by ourselves (Showcased on the UserUser case): 

```python
# Other than the ItemItem case, we can define the default_configspace_parameters by hand.
user_user_min_nbrs = UniformIntegerHyperparameter('user_user_min_nbrs', lower=1, upper=50, default_value=1)
user_user_min_sim = UniformFloatHyperparameter('user_user_min_sim', lower=0, upper=0.1, default_value=0)

# add the UserUser hyperparameters to the ConfigSpace cs
cs.add_hyperparameters([user_user_min_sim, user_user_min_nbrs])

# Set the condition that all UserUser hyperparameters are bound to the UserUser algorithm
cs.add_condition(InCondition(user_user_min_sim, regressor, values=['UserUser']))
cs.add_condition(InCondition(user_user_min_nbrs, regressor, values=['UserUser']))
```

After setting up the custom configuration space, we can perform combined algorithm selection and hyperparameter 
optimization on our custom configuraiton space using LensKit. This is done in the same way as Scenario 1. We just
have to provide our custom configuration space as an additional parameter. 

```python
# search for the best explicit configuration on the defined subset of algorithms
model, config = find_best_impllicit_configuration(train, cs, filer=filer)
```

Once we got the best model returned, we can handle it like any other Recommender model.

```python
# fit
model.fit(train)
# recommend
recs = batch.recommend(algo=model, users=test['user'].unique(), n=5)

# initialize RecListAnalysis
rla = topn.RecListAnalysis()
# add precision metric
rla.add_metric(topn.precision)

# compute scores
scores = rla.compute(recs, test, include_missing=True)
```

#### Scenario 3

In Senario 3 we are going to perform hyperparameter optimization on a single algorithm. First we need to define
our custom configuration space with just a single algorithm included.

```python
# initialize ConfigurationSpace
cs = ConfigurationSpace()
# define ItemItem as only regressor
regressor = CategoricalHyperparameter('regressor', ['ItemItem'])
# add regressor to ConfigurationSpace
cs.add_hyperparameter(regressor)
# Get a list of all ItemItem parameters of the provided ItemItem.get_default_configspace_hyperparameters() function
hyperparameter_list = ItemItem.get_default_configspace_hyperparameters()
# add all hyperparameters of hyperparameterlist to the ConfigurationSpace
cs.add_hyperparameters(hyperparameter_list)
# Set the condition that all ItemItem hyperparameters are bound to the ItemItem algorithm
for hyperparameter in hyperparameter_list:
    cs.add_condition(InCondition(hyperparameter, regressor, values=['ItemItem']))
```

Then we can use LensKit-Auto like we did in Scenario 2.

```python
# optimize the ItemItem algorithms' parameters for the given dataset with LensKit-Auto
model, config = find_best_implicit_configuration(train, cs, filer=filer)
```

As in all other Scenarios, we can handle the returned optimized model as any other LensKit Predictor.
```python
# fit
model.fit(train)
# recommend
recs = batch.recommend(algo=model, users=test['user'].unique(), n=5)

# initialize RecListAnalysis
rla = topn.RecListAnalysis()
# add precision metric
rla.add_metric(topn.precision)

# compute scores
scores = rla.compute(recs, test, include_missing=True)
```

### Rating Prediction

First, we need to split the data in a train and test split to evaluate our model.
The train-test splits can be performed based on data rows or user data. For the rating prediction example we are
splitting the data based on the data rows. The Top-N ranking predicion example showcases the data-split based 
on user data. 

```
train, test = sample_rows(mlsmall, None, 25000)
x_test = test.copy()
x_test.drop('rating', inplace=True, axis=1)
y_test = test[['rating']].iloc[:, 0]
```

#### Scenario 1

Scenario 1 describes the fully automated combined algorithm selection and hyperparameter optimization. This 
application scenario is recommended for inexperienced developers who have no ixerience in setting up a 
configuration space. 


We perform combined algorithm selection and hyperparameter optimization with LensKit-Auto
```python
model, config = find_best_explicit_configuration(train, filer=filer)
```

The developer will get a model and a configuration dictionary in return. The default search time is four hours. 
Afterward, the developer can handle the returned model as any other LensKit Predictor model.

```python
model.fit(train)
predictions = model.predict(x_test)
root_mean_square_error = rmse(predictions, y_test)
```

#### Scenario 2

Scenario 2 describes the automated combined algorithm selection and hyperparameter optimization of a custom 
configuration space. A developer that wants to use application scenario 2 has to be able to define their own 
custom configuration space in order to provide it as an input parameter. Apart from defining a custom 
configuration space, the usage of LensKit-Auto is the same.

First, a configuration space needs to be initialized. All included algorithms need to be added to the regressor 
list of the configuration space. 

```python
# initialize ConfigurationSpace
cs = ConfigurationSpace()
# define ItemItem and UserUser as set of algorithms
regressor = CategoricalHyperparameter('regressor', ['ItemItem', 'UserUser'])
# add the ItemItem and UserUser set to the ConfigurationSpace cs
cs.add_hyperparameter(regressor)
```

Afterward, we need to add the hyperparameters for each algorithm to the configuration space.
We can use the pre defined configuration space of LensKit (Showcased on the ItemItem case):

```python
# Get a list of all ItemItem parameters of the provided ItemItem.get_default_configspace_hyperparameters() function
hyperparameter_list = ItemItem.get_default_configspace_hyperparameters()
# add all hyperparameters of hyperparameterlist to the ConfigurationSpace
cs.add_hyperparameters(hyperparameter_list)
# Set the condition that all ItemItem hyperparameters are bound to the ItemItem algorithm
for hyperparameter in hyperparameter_list:
    cs.add_condition(InCondition(hyperparameter, regressor, values=['ItemItem']))
```

Or we can define the configuration space by ourselves (Showcased on the UserUser case): 

```python
# Other than the ItemItem case, we can define the default_configspace_parameters by hand.
user_user_min_nbrs = UniformIntegerHyperparameter('user_user_min_nbrs', lower=1, upper=50, default_value=1)
user_user_min_sim = UniformFloatHyperparameter('user_user_min_sim', lower=0, upper=0.1, default_value=0)

# add the UserUser hyperparameters to the ConfigSpace cs
cs.add_hyperparameters([user_user_min_sim, user_user_min_nbrs])

# Set the condition that all UserUser hyperparameters are bound to the UserUser algorithm
cs.add_condition(InCondition(user_user_min_sim, regressor, values=['UserUser']))
cs.add_condition(InCondition(user_user_min_nbrs, regressor, values=['UserUser']))
```

After setting up the custom configuration space, we can perform combined algorithm selection and hyperparameter 
optimization on our custom configuraiton space using LensKit. This is done in the same way as Scenario 1. We just
have to provide our custom configuration space as an additional parameter. 

```python
# search for the best explicit configuration on the defined subset of algorithms
model, config = find_best_explicit_configuration(train, cs, filer=filer)
```

Once we got the best model returned, we can handle it like any other Predictor model.

```python
model.fit(train)
predictions = model.predict(x_test)
root_mean_square_error = rmse(predictions, y_test)
```

#### Scenario 3

In Senario 3 we are going to perform hyperparameter optimization on a single algorithm. First we need to define
our custom configuration space with just a single algorithm included.

```python
 # initialize ConfigurationSpace
cs = ConfigurationSpace()
# define ItemItem as only regressor
regressor = CategoricalHyperparameter('regressor', ['ItemItem'])
# add regressor to ConfigurationSpace
cs.add_hyperparameter(regressor)
# Get a list of all ItemItem parameters of the provided ItemItem.get_default_configspace_hyperparameters() function
hyperparameter_list = ItemItem.get_default_configspace_hyperparameters()
# add all hyperparameters of hyperparameterlist to the ConfigurationSpace
cs.add_hyperparameters(hyperparameter_list)
# Set the condition that all ItemItem hyperparameters are bound to the ItemItem algorithm
for hyperparameter in hyperparameter_list:
    cs.add_condition(InCondition(hyperparameter, regressor, values=['ItemItem']))
```

Then we can use LensKit-Auto like we did in Scenario 2.

```python
# optimize the ItemItem algorithms' parameters for the given dataset with LensKit-Auto
model, config = find_best_explicit_configuration(train, cs, filer=filer)
```

As in all other Scenarios, we can handle the returned optimized model as any other LensKit Predictor.

```python    
model.fit(train)
predictions = model.predict(x_test)
root_mean_square_error = rmse(predictions, y_test)
```
