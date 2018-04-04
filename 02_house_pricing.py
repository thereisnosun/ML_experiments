import os
import tarfile
from six.moves import urllib
import hashlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import CategoricalEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
#transformer to handle pandas DataFrame in skilearn
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def plot_housing(housing):
    housing.hist(bins=50, figsize=(20,15))
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100,
                 label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.show()


def plot_scatter(housing):
    scatter_matrix(housing[attributes], figsize=(12,8))
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.show()


def expriment_test_set(housing):
    housing_with_id = housing.reset_index()
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    # print (train_set)


def housing_corr_experiment(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    corr_matrix = housing.corr();
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def find_statification(housing):
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_test_set, strat_train_set


def category_to_num(housing):
    housing_cat = housing["ocean_proximity"]
    housing_cat_encoded, housing_categories = housing_cat.factorize()
    encoder = OneHotEncoder();
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
    print(housing_cat_1hot.toarray())


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def compute_scores(model, housing_prepared, housing_labels):
    scores = cross_val_score(model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    display_scores(tree_rmse_scores)


def create_pipelines(housing_num):
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_caller', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense"))
    ])
    return num_pipeline, cat_pipeline


def compute_mse(housing_prepared, regression_model):
    housing_prediction = regression_model.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_prediction)
    lin_rmse = np.sqrt(lin_mse)
    print ("MSE is -", lin_rmse)


def hyper_param_search(housing_prepared, housing_labels):
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]
    forest_reg = RandomForestRegressor();
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(housing_prepared, housing_labels)
    print(grid_search.best_params_)
    return grid_search


def SVMRegression(housing_prepared, housing_labels):
    param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]
    svm_reg = SVR()
    grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring="neg_mean_squared_error", verbose=2, n_jobs=4)
    grid_search.fit(housing_prepared, housing_labels)

if __name__ == '__main__':

    #fetch_housing_data();
    housing = load_housing_data()

    #stratification
    strat_test_set, strat_train_set = find_statification(housing)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    median = housing["total_bedrooms"].median() # option 3
    housing["total_bedrooms"].fillna(median, inplace=True)

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

    #filling missing values
    imputer = Imputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis = 1)
    imputer.fit(housing_num)
    print (imputer.statistics_)

    #feature regularization
    num_pipeline, cat_pipeline = create_pipelines(housing_num)
    #union pipelines into one entity
    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)
    ])
    housing_prepared = full_pipeline.fit_transform(housing)

    #training the model
    lin_reg = LinearRegression();
    lin_reg.fit(housing_prepared, housing_labels)
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_dapa_prepared = full_pipeline.transform(some_data)
    print("Prediction - ", lin_reg.predict(some_dapa_prepared))
    print("Labels:", list(some_labels))

    #computing the error
    compute_mse(housing_prepared, lin_reg);

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)
    compute_mse(housing_prepared, tree_reg)

    #cross-validation
    compute_scores(tree_reg, housing_prepared, housing_labels)
    compute_scores(lin_reg, housing_prepared, housing_labels)

    forest_reg = RandomForestRegressor();
    forest_reg.fit(housing_prepared, housing_labels)
    compute_scores(forest_reg, housing_prepared, housing_labels)

    #hyperparameters search
    grid_search = hyper_param_search(housing_prepared, housing_labels)

    #evaluating the final model
    final_model = grid_search.best_estimator_
    X_test = strat_test_set.drop("median_house_value", axis=1)
    Y_test = strat_test_set['median_house_value'].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(Y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print('Final mse {0}. Final rmse {1}'.format(final_mse, final_rmse))

    SVMRegression(housing_prepared, housing_labels)
