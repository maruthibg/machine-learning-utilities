from feature_selector import FeatureSelector
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv(r'car_sales.csv',encoding='iso8859_15')
train_labels = train['SalePrice']
train = train.drop(columns = ['SalePrice'])

print('Processing training data with columns - %s'%(len(train.columns)))
fs = FeatureSelector(data = train, labels = train_labels)

def drop_columns_with_lessthreshold(fs, train):
    fs.identify_missing(missing_threshold=0.6)
    missing_features = fs.ops['missing']
    train.drop(missing_features, axis = 1, inplace=True)
    print('After drop_columns_with_lessthreshold columns left - %s'%(len(train.columns)))

def drop_columns_single_unique(fs, train):
    fs.identify_single_unique()
    single_unique = fs.ops['single_unique']
    train.drop(single_unique, axis = 1, inplace=True)
    print('After drop_columns_single_unique columns left - %s'%(len(train.columns)))

def drop_highly_corelated_features(fs, train) :
    fs.identify_collinear(correlation_threshold=0.9)
    correlated_features = fs.ops['collinear']
    train.drop(correlated_features, axis = 1, inplace=True)
    print('After drop_highly_corelated_features columns left - %s'%(len(train.columns)))

def update_values(train, columns, value, criterions=[]):
    def update(row, col, value, criterions):
        if not (type(row[col]) is float or type(row[col]) is int):
            # Few filters to better filter the individual column / row data
            for c in criterions:
                if c in str(row[col]):
                    return value
        return row[col]
    for col in columns:
        # Select columns that should be numeric
        # Convert the data type to float
        train[col] = train.apply(lambda r:update(r, col, value, criterions), axis=1)

def convert_object_to_float(train):
    # Special case
    cols = ['ListPrice', 'Options']
    for col in cols:
        try:
            train[col] = pd.to_numeric(train[col], errors='coerce')
        except:
            pass
    #
    object_columns = [i for i in train.select_dtypes(include='object').columns]
    for col in object_columns:
        try:
            train[col] = train[col].fillna(0)
            train[col] = train[col].astype(float)
        except:
            pass
    
def order(train):
    object_columns = [i for i in train.select_dtypes(include='object').columns]
    nonobject_columns = [i for i in train.select_dtypes(exclude='object').columns]
    columns_ordered = object_columns + nonobject_columns
    train = train[columns_ordered]
    return train, object_columns, len(object_columns)

def reupdate_values(train, columns):
    for col in columns:
        train[col] = np.where(train[col] == 0, 'NA', train[col])


def missing_data(X, object_columns_len):
    #Taking care of Missing Data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(X[:, object_columns_len:])
    X[:, object_columns_len:] = imputer.transform(X[:, object_columns_len:])
    return X

def categorical_encoder(X, object_columns_len):
    #Encoding Text Data & handling categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X = LabelEncoder()
    for i in range(object_columns_len):
        X[:,i]=labelencoder_X.fit_transform(X[:,i])
    for i in range(object_columns_len):
        onehotencoder = OneHotEncoder(categorical_features=[i])
        X = onehotencoder.fit_transform(X).toarray()
    return X


# Full swing action
drop_columns_with_lessthreshold(fs, train)
drop_columns_single_unique(fs, train)
drop_highly_corelated_features(fs, train)
drop_manually(train)
update_values(train, train.columns, value=0, criterions=[':', '.0'])
convert_object_to_float(train)
train, object_columns, object_columns_len = order(train)
reupdate_values(train, object_columns)


# Combine data
dataset = pd.concat([train, train_labels], axis=1)
dataset['SalePrice']=dataset['SalePrice'].fillna(dataset['SalePrice'].mean())

X_initial = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_post_missing = missing_data(X_initial, object_columns_len)
X_post_encoding = categorical_encoder(X_post_missing, object_columns_len)


# Avoiding the dummy variable trap
#X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_post_encoding, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    
    # Return the performance metric
    return model_mae

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr_mae = fit_and_evaluate(lr)
print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)


from sklearn.svm import SVR
svm = SVR(C = 1000, gamma = 0.1)
svm_mae = fit_and_evaluate(svm)
print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' % svm_mae)

from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(random_state=60)
random_forest_mae = fit_and_evaluate(random_forest)
print('Random Forest Regression Performance on the test set: MAE = %0.4f' % random_forest_mae)

from sklearn.ensemble import GradientBoostingRegressor
gradient_boosted = GradientBoostingRegressor(random_state=60)
gradient_boosted_mae = fit_and_evaluate(gradient_boosted)
print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)
print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)

def plot_model_mae():
    plt.style.use('fivethirtyeight')
    # Dataframe to hold the results
    model_comparison = pd.DataFrame({'model': ['Linear Regression', 'Support Vector Machine',
                                           'Random Forest', 'Gradient Boosted',
                                            'K-Nearest Neighbors'],
                                     'mae': [lr_mae, svm_mae, random_forest_mae, 
                                         gradient_boosted_mae, knn_mae]})
    # Horizontal bar chart of test mae
    model_comparison.sort_values('mae', ascending = False).plot(x = 'model', y = 'mae', kind = 'barh',
                              color = 'red', edgecolor = 'black')
    # Plot formatting
    plt.ylabel(''); plt.yticks(size = 14); plt.xlabel('Mean Absolute Error'); plt.xticks(size = 14)
    plt.title('Model Comparison on Test MAE', size = 20)
    
plot_model_mae()




"""
#Hyperparameter Tuning with Random Search and Cross Validation¶
# Loss function to be optimized
loss = ['ls', 'lad', 'huber']
# Number of trees used in the boosting process
n_estimators = [100, 500, 900, 1100, 1500]
# Maximum depth of each tree
max_depth = [2, 3, 5, 10, 15]
# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]
# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]
# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]
# Define the grid of hyperparameters to search
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

# Create the model to use for hyperparameter tuning
model = GradientBoostingRegressor(random_state = 42)

# Set up the random search with 4-fold cross validation
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)
random_cv.fit(X_train, y_train)

"""

"""
GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='huber', max_depth=3,
             max_features='auto', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=2, min_samples_split=4,
             min_weight_fraction_leaf=0.0, n_estimators=500,
             presort='auto', random_state=42, subsample=1.0, verbose=0,
             warm_start=False)
"""
