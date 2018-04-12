import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew

private_train = pd.read_csv('data/private_train.csv')
private_test = pd.read_csv('data/private_test.csv')
private_data_dict = pd.read_csv('data/private_data_dict.csv')

print("The train data size before dropping Id feature is : {} ".format(private_train.shape))
print("The test data size before dropping Id feature is : {} ".format(private_test.shape))

private_train_idx = private_train['index']
private_test_idx = private_test['index']

# Now drop the 'index' column since it is unnecessary for  the prediction process.
private_train.drop("index", axis = 1, inplace = True)
private_test.drop("index", axis = 1, inplace = True)

# Check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(private_train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(private_test.shape))

##### Feature Engineering: Pre-process features #####

# sLength = len(private_train['address'])
# private_train['street'] = np.empty(sLength)
# private_train['street'][:] = np.nan
# private_train.to_csv('data/private_train_street.csv',sep=',')

# for i, j in enumerate(private_train['address']):
#     if i % 100 == 0:
#         print('Processing ' + str(i) + ' / ' + str(sLength) + ' ( '+ str(float(i)*100/sLength) +'%)')
        
#     strs = j.lower().strip(' ').split(' ')
#     street = ''
#     for mystr in strs:
#         flag = False
#         if(mystr.isalpha()):
#             street += (' ' + mystr)
#             flag = True
#         else:
#             if(flag):
#                 break
#         # print(mystr, mystr.isalpha())
#     private_train['street'][i] = street
#     print(j, street)
    
# private_train.to_csv('data/private_train_street.csv',sep=',')

private_train_street = pd.read_csv('data/private_test_street.csv')
private_train['street'] = private_train_street['street']
private_train["street"] = private_train["street"].fillna(' other')

# sLength = len(private_test['address'])
# private_test['street'] = np.empty(sLength)
# private_test['street'][:] = np.nan
# private_test.to_csv('data/private_test_street.csv',sep=',')

# for i, j in enumerate(private_test['address']):
#     if i % 100 == 0:
#         print('Processing ' + str(i) + ' / ' + str(sLength) + ' ( '+ str(float(i)*100/sLength) +'%)')
        
#     strs = j.lower().strip(' ').split(' ')
#     street = ''
#     for mystr in strs:
#         flag = False
#         if(mystr.isalpha()):
#             street += (' ' + mystr)
#             flag = True
#         else:
#             if(flag):
#                 break
#         # print(mystr, mystr.isalpha())
#     private_test['street'][i] = street
#     print(j, street)
    
# private_test.to_csv('data/private_test_street.csv',sep=',')

private_test_street = pd.read_csv('data/private_test_street.csv')
private_test['street'] = private_test_street['street']
private_train["street"] = private_train["street"].fillna(' other')

# sLength = len(private_train['completion_date'])
# private_train['new_completion_date'] = np.empty(sLength)
# private_train['new_completion_date'][:] = np.nan
# private_train.to_csv('data/private_train_completion_date.csv',sep=',')

# for i, j in enumerate(private_train['completion_date']):
#     if i % 100 == 0:
#         print('Processing ' + str(i) + ' / ' + str(sLength) + ' ( '+ str(float(i)*100/sLength) +'%)')    
#     tenure_list = j.strip().split('/')
#     private_train['new_completion_date'][i] = tenure_list[-1]
    
# private_train.to_csv('data/private_train_completion_date.csv',sep=',')

private_train_completion_date = pd.read_csv('data/private_train_completion_date.csv')
private_train['new_completion_date'] = private_train_completion_date['new_completion_date']

# sLength = len(private_test['completion_date'])
# private_test['new_completion_date'] = np.empty(sLength)
# private_test['new_completion_date'][:] = np.nan
# private_test.to_csv('data/private_test_completion_date.csv',sep=',')

# for i, j in enumerate(private_test['completion_date']):
#     if i % 100 == 0:
#         print('Processing ' + str(i) + ' / ' + str(sLength) + ' ( '+ str(float(i)*100/sLength) +'%)')    
#     tenure_list = j.strip().split('/')
#     private_test['new_completion_date'][i] = tenure_list[-1]
    
# private_test.to_csv('data/private_test_completion_date.csv',sep=',')

private_test_completion_date = pd.read_csv('data/private_test_completion_date.csv')
private_test['new_completion_date'] = private_test_completion_date['new_completion_date']

# sLength = len(private_train['tenure'])
# private_train['remaining_year'] = np.empty(sLength)
# private_train['remaining_year'][:] = np.nan
# private_train.to_csv('data/private_train_remaining_year.csv',sep=',')

# for i, j in enumerate(private_train['tenure']):
#     if i % 100 == 0:
#         print('Processing ' + str(i) + ' / ' + str(sLength) + ' ( '+ str(float(i)*100/sLength) +'%)')    
#     tenure_list = j.strip().split(' ')
#     if len(tenure_list) < 4:
#         continue
#     max_year = tenure_list[0]
#     from_year = tenure_list[-1].split('/')[-1]
#     contract_year = private_train['month'][i].split('-')[0]
#     remaining_year = int(max_year) - int(contract_year) + int(from_year)
#     private_train['remaining_year'][i] = remaining_year
#     print('contract_year:', contract_year, ', max_year:', max_year, ', from_year:', from_year, ', remaining year:', remaining_year)
    
# private_train.to_csv('data/private_train_remaining_year.csv',sep=',')

private_train_remaining_year = pd.read_csv('data/private_train_remaining_year.csv')
private_train['remaining_year'] = private_train_remaining_year['remaining_year']

# sLength = len(private_test['tenure'])
# private_test['remaining_year'] = np.empty(sLength)
# private_test['remaining_year'][:] = np.nan
# private_test.to_csv('data/private_test_remaining_year.csv',sep=',')

# for i, j in enumerate(private_test['tenure']):
#     if i % 100 == 0:
#         print('Processing ' + str(i) + ' / ' + str(sLength) + ' ( '+ str(float(i)*100/sLength) +'%)')    
#     tenure_list = j.strip().split(' ')
#     if len(tenure_list) < 4:
#         continue
#     max_year = tenure_list[0]
#     from_year = tenure_list[-1].split('/')[-1]
#     contract_year = private_test['month'][i].split('-')[0]
#     remaining_year = int(max_year) - int(contract_year) + int(from_year)
#     private_test['remaining_year'][i] = remaining_year
#     print('contract_year:', contract_year, ', max_year:', max_year, ', from_year:', from_year, ', remaining year:', remaining_year)
    
# private_test.to_csv('data/private_test_remaining_year.csv',sep=',')

private_test_remaining_year = pd.read_csv('data/private_test_remaining_year.csv')
private_test['remaining_year'] = private_test_remaining_year['remaining_year']

# sLength = len(private_train['month'])
# private_train['new_month'] = np.empty(sLength)
# private_train['new_month'][:] = np.nan
# private_train.to_csv('data/private_train_month.csv',sep=',')

# for i, j in enumerate(private_train['month']):
#     if i % 100 == 0:
#         print('Processing ' + str(i) + ' / ' + str(sLength) + ' ( '+ str(float(i)*100/sLength) +'%)')    
#     tenure_list = j.strip().split('-')
#     if len(tenure_list) > 2:    
#         continue
#     private_train['new_month'][i] = int(tenure_list[0]) + int(tenure_list[1])
#     print('Year Month:', j, ', New Month:', private_train['new_month'][i])
    
# private_train.to_csv('data/private_train_month.csv',sep=',')

private_train_month = pd.read_csv('data/private_train_month.csv')
private_train['new_month'] = private_train_month['new_month']

# sLength = len(private_test['month'])
# private_test['new_month'] = np.empty(sLength)
# private_test['new_month'][:] = np.nan
# private_test.to_csv('data/private_test_month.csv',sep=',')

# for i, j in enumerate(private_test['month']):
#     if i % 100 == 0:
#         print('Processing ' + str(i) + ' / ' + str(sLength) + ' ( '+ str(float(i)*100/sLength) +'%)')    
#     tenure_list = j.strip().split('-')
#     if len(tenure_list) > 2:    
#         continue
#     private_test['new_month'][i] = (int(tenure_list[0]) - 1995)*12 + int(tenure_list[1])
#     print('Year Month:', j, ', New Month:', private_test['new_month'][i])
    
# private_test.to_csv('data/private_test_month.csv',sep=',')

private_test_month = pd.read_csv('data/private_test_month.csv')
private_test['new_month'] = private_test_month['new_month']

##### END of Feature Engineering: Pre-process features #####

import pickle

# private_dict = {}
# for i in private_train.keys():
#     private_dict[i] = sorted(set(private_train[i]))
# f = open('data/private_dict.pckl', 'wb')
# pickle.dump(private_dict, f)
# f.close()

with open('data/private_dict.pckl', 'rb') as f:
    private_dict = pickle.load(f)


# scatter plot floor_area_sqm/resale_price
# var = 'floor_area_sqm'
# # hdb_train_var = [int(i) for i in hdb_dict[var]]
# data = pd.concat([private_train['price'], private_train[var]], axis=1)
# data.plot.scatter(x=var, y='price');


# Deleting outliers
private_train = private_train.drop(private_train[(private_train['floor_area_sqm']>6000) & (private_train['price'] < 50000000)].index)
private_train = private_train.drop(private_train[(private_train['floor_area_sqm']<4000) & (private_train['price'] > 100000000)].index)


# scatter plot floor_area_sqm/resale_price
# var = 'floor_area_sqm'
# # hdb_train_var = [int(i) for i in hdb_dict[var]]
# data = pd.concat([private_train['price'], private_train[var]], axis=1)
# data.plot.scatter(x=var, y='price');


# scatter plot floor_area_sqm/resale_price
var = 'remaining_year'
# hdb_train[var] = pd.to_numeric(hdb_train[var])
# hdb_train_var = [int(i) for i in hdb_dict[var]]
data = pd.concat([private_train['price'], private_train[var]], axis=1)
data.plot.scatter(x=var, y='price');


# Data distribution and Q-Q Plot
# sns.distplot(private_train['price'] , fit=norm);
#
# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(private_train['price'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
#
# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(private_train['price'], plot=plt)
# plt.show()


# Perform Log-Transformation
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
private_train["price"] = np.log1p(private_train["price"])

sns.distplot(private_train['price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(private_train['price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(private_train['price'], plot=plt)
plt.show()


### Correlation map to see how features are correlated with SalePrice ###
# corrmat = private_train.corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.9, square=True)


# Getting all_data

ntrain = private_train.shape[0]
ntest = private_test.shape[0]
y_train = private_train.price.values
all_data = pd.concat((private_train, private_test)).reset_index(drop=True)
all_data.drop(['price'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

# Missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data

# Missing data by feature
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

# Check Missing data again
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data

# Select attributes for training purpose

for i in ['address', 'completion_date', 'contract_date', 'latitude', 'longitude', 'month', 'postal_code', 'postal_district', 'postal_sector', 'remaining_year', 'tenure', 'unit_num', 'street']:
    all_data = all_data.drop([i], axis=1)

with open('data/all_data_private_no_street_before_dummy.pckl', 'wb') as f:
    pickle.dump(all_data, f, protocol=2)

# Transform numerical values into categorical values
all_data = pd.get_dummies(all_data)
print(all_data.shape)

with open('data/all_data_private_no_street.pckl', 'wb') as f:
    pickle.dump(all_data, f, protocol=2)


train = all_data[:ntrain]
test = all_data[ntrain:]
print('Trainibng set shape: ', train.shape)
print('Test set shape: ', test.shape)

print('all data shape: ', all_data.shape)
print('y_train shape: ', y_train.shape)
print('Trainibng set shape: ', train.shape)
print('Test set shape: ', test.shape)

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# Validation function
n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

# Lasso
print('Processing lasso...')
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# Elastic net
print('Processing ENet...')
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# Gradient Boosting
print('Processing Gradient Boosting...')
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# XGBoost
print('Processing XGBoost...')
model_xgb = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.0468,
                             learning_rate=0.05, max_depth=10,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.7, silent=1,
                             random_state=7, eta=0.1)

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# LightGBM
print('Processing lgb...')
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# stacked Avg models
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


print('Processing Stacked Averaged Models...')
stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost), meta_model=lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))

'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train, (stacked_pred * 0.6 + xgb_pred * 0.2 + lgb_pred * 0.2)))

ensemble = stacked_pred * 0.6 + xgb_pred * 0.2 + lgb_pred * 0.2

sub = pd.DataFrame()
sub['index'] = private_test_idx
sub['price'] = ensemble
sub.to_csv('submission_private_tuned_stacked.csv', index=False)

# For debug purpose
import pdb;
pdb.set_trace()
