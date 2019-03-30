# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set a few plotting defaults
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.rcParams['patch.edgecolor'] = 'k'

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
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
warnings.simplefilter(action='ignore', category=Warning)

import random
import math
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from tqdm import tqdm_notebook

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def f1_wrapper(y, y_pred, **kwargs):
    y = np.round(y)
    y_pred = np.round(y_pred)
    return f1_score(y, y_pred, **kwargs)


# Custom scorer for cross validation
scorer = make_scorer(f1_wrapper, greater_is_better=True, average = 'macro')

pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

mapping = {"yes": 1, "no": 0}

# Apply same operation to both train and test
for df in [train, test]:
    # Fill in the values with the correct mapping
    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)

# Add null Target column to test
test['Target'] = np.nan
data = train.append(test, ignore_index = True)

# Groupby the household and figure out the number of unique values
all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]

# Iterate through each household
for household in not_equal.index:
    # Find the correct label (for the head of household)
    true_target = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'])

    # Set the correct label for all members in the household
    train.loc[train['idhogar'] == household, 'Target'] = true_target

data['v18q1'] = data['v18q1'].fillna(0)

# Fill in households that own the house with 0 rent payment
data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0

# Create missing rent payment column
data['v2a1-missing'] = data['v2a1'].isnull()

data['v2a1-missing'].value_counts()

# If individual is over 19 or younger than 7 and missing years behind, set it to 0
data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0

# Add a flag for those between 7 and 19 with a missing value
data['rez_esc-missing'] = data['rez_esc'].isnull()
data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5

id_ = ['Id', 'idhogar', 'Target']
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3',
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7',
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5',
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10',
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3',
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8',
            'instlevel9', 'mobilephone', 'rez_esc-missing']

ind_ordered = ['rez_esc', 'escolari', 'age']

hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo',
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother',
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo',
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1',
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4',
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4',
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3',
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2',
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']

sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe',
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']

# Remove squared variables
data = data.drop(columns = sqr_)
data.shape

heads = data.loc[data['parentesco1'] == 1, :]
heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
heads.shape

# Create correlation matrix
corr_matrix = heads.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop

heads = heads.drop(columns = ['tamhog', 'hogar_total', 'r4t3'])

heads['hhsize-diff'] = heads['tamviv'] - heads['hhsize']

elec = []

# Assign values
for i, row in heads.iterrows():
    if row['noelec'] == 1:
        elec.append(0)
    elif row['coopele'] == 1:
        elec.append(1)
    elif row['public'] == 1:
        elec.append(2)
    elif row['planpri'] == 1:
        elec.append(3)
    else:
        elec.append(np.nan)

# Record the new variable and missing flag
heads['elec'] = elec
heads['elec-missing'] = heads['elec'].isnull()

# Remove the electricity columns
# heads = heads.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])

heads = heads.drop(columns = 'area2')

heads.groupby('area1')['Target'].value_counts(normalize = True)

# Wall ordinal variable
heads['walls'] = np.argmax(np.array(heads[['epared1', 'epared2', 'epared3']]),
                           axis = 1)
# Roof ordinal variable
heads['roof'] = np.argmax(np.array(heads[['etecho1', 'etecho2', 'etecho3']]),
                           axis = 1)
heads = heads.drop(columns = ['etecho1', 'etecho2', 'etecho3'])

# Floor ordinal variable
heads['floor'] = np.argmax(np.array(heads[['eviv1', 'eviv2', 'eviv3']]),
                           axis = 1)

# Create new feature
heads['walls+roof+floor'] = heads['walls'] + heads['roof'] + heads['floor']

counts = pd.DataFrame(heads.groupby(['walls+roof+floor'])['Target'].value_counts(normalize = True)).rename(columns = {'Target': 'Normalized Count'}).reset_index()

# No toilet, no electricity, no floor, no water service, no ceiling
heads['warning'] = 1 * (heads['sanitario1'] +
                         (heads['elec'] == 0) +
                         heads['pisonotiene'] +
                         heads['abastaguano'] +
                         (heads['cielorazo'] == 0))

heads['phones-per-capita'] = heads['qmobilephone'] / heads['tamviv']
heads['tablets-per-capita'] = heads['v18q1'] / heads['tamviv']
heads['rooms-per-capita'] = heads['rooms'] / heads['tamviv']
heads['rent-per-capita'] = heads['v2a1'] / heads['tamviv']

variables = ['Target', 'dependency', 'warning', 'walls+roof+floor', 'meaneduc',
             'floor', 'r4m1', 'overcrowding']
ind = data[id_ + ind_bool + ind_ordered]

# Create correlation matrix
corr_matrix = ind.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]
ind = ind.drop(columns = 'male')

ind[[c for c in ind if c.startswith('instl')]].head()

ind['inst'] = np.argmax(np.array(ind[[c for c in ind if c.startswith('instl')]]), axis = 1)

# Drop the education columns
# ind = ind.drop(columns = [c for c in ind if c.startswith('instlevel')])
ind.shape

ind['escolari/age'] = ind['escolari'] / ind['age']

ind['inst/age'] = ind['inst'] / ind['age']
ind['tech'] = ind['v18q'] + ind['mobilephone']

# Define custom function
range_ = lambda x: x.max() - x.min()
range_.__name__ = 'range_'

# Group and aggregate
ind_agg = ind.drop(columns = 'Target').groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std', range_])

# Rename the columns
new_col = []
for c in ind_agg.columns.levels[0]:
    for stat in ind_agg.columns.levels[1]:
        new_col.append(f'{c}-{stat}')

ind_agg.columns = new_col

# Create correlation matrix
corr_matrix = ind_agg.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

print(f'There are {len(to_drop)} correlated columns to remove.')

ind_agg = ind_agg.drop(columns = to_drop)
ind_feats = list(ind_agg.columns)

# Merge on the household id
final = heads.merge(ind_agg, on = 'idhogar', how = 'left')

print('Final features shape: ', final.shape)

corrs = final.corr()['Target']
corrs.sort_values().dropna().tail()

head_gender = ind.loc[ind['parentesco1'] == 1, ['idhogar', 'female']]
final = final.merge(head_gender, on = 'idhogar', how = 'left').rename(columns = {'female': 'female-head'})
final.groupby('female-head')['Target'].value_counts(normalize=True)
final.groupby('female-head')['meaneduc'].agg(['mean', 'count'])

# Labels for training
train_labels = np.array(list(final[final['Target'].notnull()]['Target'].astype(np.uint8)))

# Extract the training data
train_set = final[final['Target'].notnull()].drop(columns = ['Id', 'idhogar', 'Target'])
test_set = final[final['Target'].isnull()].drop(columns = ['Id', 'idhogar', 'Target'])

# Submission base which is used for making submissions to the competition
submission_base = test[['Id', 'idhogar']].copy()

features = list(train_set.columns)

pipeline = Pipeline([('imputer', Imputer(strategy = 'median')),
                      ('scaler', MinMaxScaler())])

# Fit and transform training data
train_set = pipeline.fit_transform(train_set)
test_set = pipeline.transform(test_set)

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

lasso = make_pipeline(RobustScaler(), Lasso(alpha =10, random_state=1, max_iter = 14000))
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
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
forest_model = RandomForestRegressor()
svr = svm.SVR()
Linearsvr = svm.LinearSVR()
Nusvr = svm.NuSVR()
KNN = KNeighborsRegressor(n_neighbors=5)
theil = TheilSenRegressor(random_state=0)
logistic = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
gnb = GaussianNB()
models = [lasso, ENet, KRR, GBoost, svr, Linearsvr, Nusvr, KNN, logistic, gnb, forest_model]
#models = [lasso, ENet, KRR, GBoost, model_xgb, model_lgb, forest_model]
#models = [lasso, ENet, KRR, GBoost, model_xgb, forest_model]

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
def cv_model(model):
    """Perform 10 fold cross validation of a model"""

    cv_scores = cross_val_score(model, train_set, train_labels, cv = n_folds, scoring=scorer, n_jobs = -1)
    print(f'5 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')

    return cv_scores

		class Particle:
		    def __init__(self, models):
		        self.num_dimensions = len(models)
		        self.meta_index = np.zeros((self.num_dimensions))
		        self.meta_index[0] = np.random.randint(self.num_dimensions-1)
		        #if(self.meta_index[0] in [1, 2, 5]):
		            #self.meta_index[0] = 0
		        self.position = np.vstack((randomizeBits(len(models)), self.meta_index))
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
		            if(self.position[0][i] != 0):
		                break_boolean = True
		        if(break_boolean == False):
		            return

		        instance_meta_model = models[int(self.position[1][0])]
		        instance_base_models = []

		        for i in range(0, self.num_dimensions):
		            if(self.position[0][i] == 1):
		                instance_base_models.append(models[i])

		        meta_model = MetaModel(base_models = instance_base_models, meta_model = instance_meta_model)
		        #meta_model.fit(train, train_result.reshape(-1))

		        score = cv_model(meta_model)
		        self.curr_error = score.mean()

		        print("Error: " + str(self.curr_error))
		        print("Position: " + str(self.position))

		        if(self.curr_error > self.pbest_error or self.pbest_error == -1):
		            self.pbest_pos = self.position
		            self.pbest_error = self.curr_error

		    def updateVelocity(self, inertia, cog_const, soc_const, gbest_pos):
		        for i in range(0, self.num_dimensions):
		            r1=random.random()
		            r2=random.random()

		            card_count_soc = card_count_cog = card_count_self = 0

		            # Get Encoding of Cardinality
		            for j in self.pbest_pos[0]:
		                card_count_cog += int(j)
		            for j in gbest_pos[0]:
		                card_count_soc += int(j)
		            for j in self.position[0]:
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
		            #print(self.pbest_pos)
		            #print(self.position)
		            for a in range(0, self.num_dimensions):
		                if(self.pbest_pos[0][a] == 1 and self.position[0][a] == 0):
		                    L_cog[1, a] = 1;
		                if(gbest_pos[0][a] == 1 and self.position[0][a] == 0):
		                    L_soc[1, a] = 1;

		            vel_cognitive = cog_const * r1 * L_cog
		            vel_social = soc_const * r2 * L_soc
		            vel_self = 0.2 * L_self
		            self.velocity = inertia * self.velocity + vel_cognitive + vel_social + vel_self

		    def updatePosition(self, meta_choice_array):
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
		        updated_position = np.zeros((2, self.num_dimensions))
		        for i in range(cardinality):
		            max_index = np.argmax(temp_selection)
		            #print("Max Index is: " + str(max_index))
		            updated_position[0][max_index] = 1
		            temp_selection[max_index] = float('-inf')

		        print(meta_choice_array)

		        x = np.random.choice(range(0, self.num_dimensions), 1, p= meta_choice_array)
		        print("Initial: " + str(x))
		        #if(x in [1, 2, 5]):
		           # x = 0
		        print("Later: " + str(x))

		        self.position = updated_position
		        self.position[1][0] = x

		class PSO:
		    def __init__(self, num_particles, epoch, continue_flag = False, pso = b'a'):
		        self.num_dimensions = len(models)
		        self.gbest_pos = []
		        self.gbest_error = -1
		        self.inertia = 0.8
		        self.cog_const = 1
		        self.soc_const = 2
		        self.epoch = epoch
		        self.num_particles = num_particles
		        self.epsilon = 1e-6
		        self.meta_weights = np.full((self.num_dimensions), self.epsilon)
		        self.meta_size = np.zeros((self.num_dimensions))
		        self.real_meta_array = np.full((self.num_dimensions), self.epsilon)
		        self.global_history = []

		        if(continue_flag):
		            self.swarm = pickle.loads(pso)
		        else:
		            self.swarm = []
		            for i in range(0, num_particles):
		                self.swarm.append(Particle(models))

		    def run(self):
		        print("")
		        for timestep in range(self.epoch):
		            print("Timestep: %d" % timestep)

		            # Update Meta Array
		            for i in range(0, self.num_dimensions):
		                if(self.meta_size[i] != 0):
		                    self.real_meta_array[i] = self.meta_weights[i] / self.meta_size[i]
		            self.real_meta_array = softmax(self.real_meta_array)

		            for i in range(0, self.num_particles):
		                print(self.swarm[i].position)
		                self.swarm[i].calculate_cost()

		                meta_ind = int(self.swarm[i].position[1][0])
		                if(self.meta_size[meta_ind] == 0):
		                    self.meta_weights[meta_ind] = 0
		                self.meta_weights[meta_ind] += self.swarm[i].curr_error
		                self.meta_size[meta_ind] += 1

		                if self.swarm[i].curr_error > self.gbest_error or self.gbest_error == -1:
		                    self.gbest_pos = list(self.swarm[i].position)
		                    self.gbest_error = float(self.swarm[i].curr_error)

		                self.swarm[i].updateVelocity(self.inertia, self.cog_const, self.soc_const, self.gbest_pos)
		                self.swarm[i].updatePosition(self.real_meta_array)
		                print("Pickled: " + str(timestep) + " " + str(i))
		                print(pickle.dumps(self))

		            self.global_history.append(self.gbest_error)

		        # Revisit Later
		        print("---------------------------------")
		        print("Final:")
		        print("Gbest Position: " + str(self.gbest_pos))
		        print("Gbest Error: " + str(self.gbest_error))
		        

pso = PSO(num_particles = 50, epoch = 2)
pso.run()
