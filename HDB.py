import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew


hdb_train = pd.read_csv('data/hdb_train.csv')
hdb_test = pd.read_csv('data/hdb_test.csv')
hdb_data_dict = pd.read_csv('data/hdb_data_dict.csv')

hdb_train.head(5)

# Check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(hdb_train.shape))
print("The test data size before dropping Id feature is : {} ".format(hdb_test.shape))

# Save the 'Id' column
hdb_train_idx = hdb_train['index']
hdb_test_idx = hdb_test['index']

# Now drop the  'index' colum since it's unnecessary for  the prediction process.
hdb_train.drop("index", axis = 1, inplace = True)
hdb_test.drop("index", axis = 1, inplace = True)

# Check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(hdb_train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(hdb_test.shape))

'''
# scatter plot floor_area_sqm/resale_price
var = 'floor_area_sqm'
# hdb_train_var = [int(i) for i in hdb_dict[var]]
data = pd.concat([hdb_train['resale_price'], hdb_train[var]], axis=1)
data.plot.scatter(x=var, y='resale_price', ylim=(0,1200000));
'''

# Deleting outliers
hdb_train = hdb_train.drop(hdb_train[(hdb_train['floor_area_sqm']>200)].index)
hdb_train = hdb_train.drop(hdb_train[(hdb_train['floor_area_sqm']>170) & (hdb_train['resale_price']<200000)].index)


'''
# scatter plot floor_area_sqm/resale_price after Deleting outliers
var = 'floor_area_sqm'
# hdb_train_var = [int(i) for i in hdb_dict[var]]
data = pd.concat([hdb_train['resale_price'], hdb_train[var]], axis=1)
data.plot.scatter(x=var, y='resale_price', ylim=(0,1200000));
'''


'''
### Plot the distribution and Q-Q plots ###

sns.distplot(hdb_train['resale_price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(hdb_train['resale_price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(hdb_train['resale_price'], plot=plt)
plt.show()
'''

# Load the pre-processed new attribute 'new_month' = {1, ... , N}
hdb_train_month = pd.read_csv('data/hdb_train_month.csv')
hdb_train['new_month'] = hdb_train_month['new_month']

hdb_test_month = pd.read_csv('data/hdb_test_month.csv')
hdb_test['new_month'] = hdb_test_month['new_month']


# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
hdb_train['resale_price'] = np.log1p(hdb_train['resale_price'])

'''
#Check the new distribution 
sns.distplot(hdb_train['resale_price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(hdb_train['resale_price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(hdb_train['resale_price'], plot=plt)
plt.show()

'''

'''
# Correlation map to see how features are correlated with SalePrice
corrmat = hdb_train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
'''

# Drop the dependent variable and form all_data for training purpose
ntrain = hdb_train.shape[0]
ntest = hdb_test.shape[0]
y_train = hdb_train.resale_price.values
all_data = pd.concat((hdb_train, hdb_test)).reset_index(drop=True)
all_data.drop(['resale_price'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

# Select features for training
for i in ['longitude', 'latitude', 'block', 'floor', 'month', 'postal_code', 'storey_range', 'street_name']:
    all_data = all_data.drop([i], axis=1)

# Transform numerical values into categorical values
all_data = pd.get_dummies(all_data)
print(all_data.shape)

train = all_data[:ntrain]
test = all_data[ntrain:]
print('Trainibng set shape: ', train.shape)
print('Test set shape: ', test.shape)

### Start to train! ###

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# Check models on training data set
# Lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# Elastic net
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# Gradient Boosting
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.0468, 
                             learning_rate=0.05, max_depth=10, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.7, silent=1,
                             random_state = 7, eta=0.1)
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# Light GBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# Stacked Averaged Models
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
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

# Construct RMSLE measure
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# Fit the stacked avg model
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

# Fit the XGBoost model
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

# Fit the Light GBM model
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))

'''RMSE on the entire Train data when averaging'''
print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.6 +
               xgb_train_pred*0.2 + lgb_train_pred*0.2 ))

# Ensembled methods
ensemble = stacked_pred*0.6 + xgb_pred*0.2 + lgb_pred*0.2

# Submission
sub = pd.DataFrame()
sub['index'] = hdb_test_idx
sub['price'] = np.round(ensemble)
sub.to_csv('submission_hdb_stacked_new.csv',index=False)

# For debug purpose
import pdb
pdb.set_trace()
