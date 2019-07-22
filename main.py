import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA

from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def score_data_set_rf(X, y, test):
    rf = RandomForestRegressor()
    n_estimators = [6, 60, 600, 1000]
    min_samples_leaf = [2, 5, 10]
    max_features = [0.1, 0.5, 0.9]
    rf_random_grid = {'n_estimators': n_estimators,
                      'min_samples_leaf': min_samples_leaf,
                      'max_features': max_features}
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_random_grid, random_state=42, n_jobs=-1)
    rf_random.fit(X, y)
    print(rf_random.best_params_)
    model = rf_random.best_estimator_
    df = pd.DataFrame(model.predict(test), columns=['SalePrice'])
    df.index += 1461
    df.to_csv("predicted.csv")

    score = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', fit_params=None)
    return -1 * score.mean()


def score_data_set_dt(X, y):
    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', fit_params=None)
    return -1 * score.mean()


train = pd.read_csv("./train.csv", index_col=0)
test = pd.read_csv("./test.csv", index_col=0)


imputed_train = DataFrameImputer().fit_transform(train)
imputed_test = DataFrameImputer().fit_transform(test)

for i in imputed_train.columns:
    if imputed_train[i].dtype not in (int, float):
        imputed_train[i] = imputed_train[i].astype(str)
        temp = imputed_train[i].unique()
        temp_dict = {temp[i]: i for i in range(len(temp))}
        imputed_test[i] = imputed_test[i].map(temp_dict)
        imputed_train[i] = imputed_train[i].map(temp_dict)

imputed_train_predictors = imputed_train.drop(['SalePrice'], axis=1)
imputed_train_target = imputed_train.SalePrice

imputed_test_predictors = imputed_test
#import matplotlib.pyplot as plt
#plt.matshow(imputed_train.corr())
#plt.savefig('correlation.pdf')

print(score_data_set_rf(imputed_train_predictors, imputed_train_target, imputed_test))
#print(score_data_set_dt(imputed_train_predictors, imputed_train_target))
pca = PCA()
pca.fit(imputed_train_predictors)
pcaed_imputed_train_predictors = pca.transform(imputed_train_predictors)
pcaed_imputed_test_predictors = pca.transform(imputed_test_predictors)

print(score_data_set_rf(pcaed_imputed_train_predictors, imputed_train_target))
#print(score_data_set_dt(pcaed_imputed_train_predictors, imputed_train_target))
