# Resource: https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/

import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
#import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import random
import math

def func1(x):
    total=0
    for i in range(len(x)):
        total += abs(x[i]**2 - 7*x[i] + 12)
    return total

def randomizeBits(size):
    return np.random.randint(2, size=size)

def getCardinality(threshold, accumulation):
    for i in range(0, len(accumulation)):
        if (accumulation[i] > threshold):
            return i;
    return len(accumulation) - 1;

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

lasso = make_pipeline(RobustScaler(), Lasso(alpha =10, random_state=1, max_iter = 7000))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=1, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
'''model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)'''
forest_model = RandomForestRegressor()
#models = [lasso, ENet, KRR, GBoost, model_xgb, model_lgb, forest_model]
models = [lasso, ENet, KRR, GBoost, model_xgb, forest_model]

class MetaModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model

        #placeholder
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

n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, train_result.reshape(-1), scoring="neg_mean_squared_log_error", cv = kf))
    return(rmse)


class Particle:
    def __init__(self, models):
        self.num_dimensions = len(models)
        self.position = randomizeBits(len(models))
        self.velocity = np.zeros((2, self.num_dimensions))
        self.pbest_pos = []
        self.pbest_error = -1
        self.curr_error = -1
        self.models = models

        for i in range(0, self.num_dimensions):
            self.velocity[0, i] = random.uniform(-1, 1)

    def calculate_cost(self):
        break_boolean = False
        for i in range(0, self.num_dimensions):
            if(self.position[i] != 0):
                break_boolean = True
        if(break_boolean == False):
            return

        instance_meta_model = models[0]
        instance_base_models = []

        for i in range(0, self.num_dimensions):
            if(self.position[i] == 1):
                instance_base_models.append(models[i])

        meta_model = MetaModel(base_models = instance_base_models, meta_model = instance_meta_model)
        meta_model.fit(train, train_result.reshape(-1))

        score = rmsle_cv(meta_model)
        self.curr_error = score.mean()

        print("Error: " + str(self.curr_error))
        print("Position: " + str(self.position))

        if(self.curr_error < self.pbest_error or self.pbest_error == -1):
            self.pbest_pos = self.position
            self.pbest_error = self.curr_error

    def updateVelocity(self, inertia, cog_const, soc_const, gbest_pos):
        for i in range(0, self.num_dimensions):
            r1=random.random()
            r2=random.random()

            card_count_soc = card_count_cog = card_count_self = 0

            # Get Encoding of Cardinality
            for j in self.pbest_pos:
                card_count_cog += int(j)
            for j in gbest_pos:
                card_count_soc += int(j)
            for j in self.position:
                card_count_self += int(j)

            # Encode Cardinality
            L_cog = np.zeros((2, self.num_dimensions))
            L_soc = np.zeros((2, self.num_dimensions))
            L_self = np.zeros((2, self.num_dimensions))
            #print(card_count_cog, card_count_soc, card_count_self)
            L_cog[0, card_count_cog - 1] = 1
            L_soc[0, card_count_soc - 1] = 1
            L_self[0, card_count_self - 1] = 1

            # Include Selection Likelihood
            for a in range(0, self.num_dimensions):
                if(self.pbest_pos[a] == 1 and self.position[a] == 0):
                    L_cog[1, a] = 1;
                if(gbest_pos[a] == 1 and self.position[a] == 0):
                    L_soc[1, a] = 1;

            vel_cognitive = cog_const * r1 * L_cog
            vel_social = soc_const * r2 * L_soc
            vel_self = 0.2 * L_self
            self.velocity = inertia * self.velocity + vel_cognitive + vel_social + vel_self

    def updatePosition(self):
        # Create Accumulation Matrix
        accumulation = np.zeros((self.num_dimensions))
        for i in range(0, self.num_dimensions):
            if (i == 0):
                accumulation[i] = self.velocity[0, i]
            else:
                accumulation[i] = self.velocity[0, i] + accumulation[i-1]

        # Get Random Threshold
        random_thresh = random.random() * accumulation[-1]

        # Calculate Cardinality
        cardinality = getCardinality(random_thresh, accumulation)

        # Fill in updated position with maximum selection likelihoods
        temp_selection = np.copy(self.velocity[1, :])
        print("Velocity: " + str(self.velocity))
        updated_position = np.zeros((self.num_dimensions))
        for i in range(cardinality):
            max_index = np.argmax(temp_selection)
            #print("Max Index is: " + str(max_index))
            updated_position[max_index] = 1
            temp_selection[max_index] = float('-inf')

        self.position = updated_position


class PSO:
    def __init__(self, num_particles, epoch, continue_flag = False, pso=""):
        self.num_dimensions = len(models)
        self.gbest_pos = []
        self.gbest_error = -1
        self.inertia = 0.8
        self.cog_const = 1
        self.soc_const = 2
        self.epoch = epoch
        self.num_particles = num_particles
        self.epsilon = 1e6
        self.meta_weights = np.full((1, self.num_dimensions), epislon)

        if(continue_flag):
            self = pickle.loads(pso)
        else:
            self.swarm = []
            for i in range(0, num_particles):
                self.swarm.append(Particle(models))

    def run(self):
        print("")
        for timestep in range(self.epoch):
            print("Timestep: %d" % timestep)
            for i in range(0, self.num_particles):
                print(self.swarm[i].position)
                self.swarm[i].calculate_cost()

                if self.swarm[i].curr_error < self.gbest_error or self.gbest_error == -1:
                    self.gbest_pos = list(self.swarm[i].position)
                    self.gbest_error = float(self.swarm[i].curr_error)

                self.swarm[i].updateVelocity(self.inertia, self.cog_const, self.soc_const, self.gbest_pos)
                self.swarm[i].updatePosition()

        # Revisit Later
        print("---------------------------------")
        print("Final:")
        print("Gbest Position: " + str(self.gbest_pos))
        print("Gbest Error: " + str(self.gbest_error))

# Check Later
if __name__ == "__PSO__":
    main()

pso = PSO(num_particles = 10, epoch = 2)
