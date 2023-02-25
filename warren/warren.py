#import libraries
import math, numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split, KFold #python3 -m pip install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_absolute_error
from xgboost import XGBRegressor #python3 -m pip install xgboost
from google.cloud import storage #pip install google-cloud-storage, pip install --upgrade google-cloud-storage, pip install --upgrade google-api-python-client

def warren(request):
    warnings.filterwarnings('ignore')

    #these should be call parameters of the cloud function
    exec_type = 'poc'
    window = 250
    pred_type = 'max'
    ticker_type = 'all'

    #inputs
    #exec_type, window, pred_type, ticker_type = 'poc', 250, 'max', 'all'
    data_filename = 'data_%s' %exec_type + '.csv'
    predict_filename = 'predict_%s' %exec_type + '.csv'

    test_size = 750 #750, put 250 to include PYPL
    max_n_splits = 3 #3
    n_jobs = 4

    n_estimators = [25, 50, 100] #0921, before we had 20, 50, 100 
    max_depth = [1] #1#1-10  [1,2,3,4,5,6,7,8,9,10]
    learning_rate = [0.1] #0.05-0.5 0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
    min_child_weight = [1] #1,1-120               #sim to min_leaf in GBM but weights instead of counts
    subsample = [0.5] #0.5-1            #observations   [0.5,0.6,0.7,0.8,0.9,1]
    colsample_bytree = [1] #0.5-1            #predictors
    gamma = [0,5] #0-5
    reg_alpha = [0] #0-1000

    #load data
    storage_client = storage.Client()
    bucket = storage_client.bucket('raw-data_bucket')

    blob = bucket.blob(data_filename)
    with blob.open("r") as f:
        data = pd.read_csv(f)

    blob = bucket.blob(predict_filename)
    with blob.open("r") as f:
        predict = pd.read_csv(f)

    ticker_list = predict['ticker']

    if ticker_type != 'all':
        predict_filter = predict[(predict['ticker'] == 'SPY') | (predict['type'] == ticker_type)]
    else:
        predict_filter = predict

    #prediction loop
    for ticker in ticker_list:
        try:
            1/(ticker in list(predict_filter['ticker']))
            data['vrat_'+ticker] = data[::-1]['Volume_'+ticker].rolling(14).mean()/data[::-1]['Volume_'+ticker].rolling(28).mean()
            data['drat_'+ticker] = data['Dividends_'+ticker]/data['Adj Close_'+ticker]
            
            #x: feature transformation
            data_ticker = data[['date_num', 'Adj Close_'+ticker, 'vrat_'+ticker, 'drat_'+ticker, 'Stock Splits_'+ticker]]
            data_ticker['adj2_'+ticker] = data_ticker['Adj Close_'+ticker].pct_change(math.floor(window/2))
            data_ticker['adj5_'+ticker] = data_ticker['Adj Close_'+ticker].pct_change(math.floor(window/5))
            data_ticker['adj10_'+ticker] = data_ticker['Adj Close_'+ticker].pct_change(math.floor(window/10))
            data_ticker = data_ticker.fillna(axis=0,method='ffill').dropna(axis=0, how='any')

            #y: predict
            if pred_type == 'max':
                #predict = maximum expected return in rolling window
                data_ticker['predict'] = (data_ticker[::-1]['Adj Close_'+ticker].rolling(window).max()[::-1]-data_ticker['Adj Close_'+ticker])/data_ticker['Adj Close_'+ticker]
            else:
                #predict = maximum expected return in rolling window
                data_ticker['predict'] = (data_ticker[::-1]['Adj Close_'+ticker].shift(window)[::-1]-data_ticker['Adj Close_'+ticker])/data_ticker['Adj Close_'+ticker]

            #normalize predict
            min_predict = data_ticker['predict'].min(axis=0)
            max_predict = data_ticker['predict'].max(axis=0)
            data_ticker['predict'] = (data_ticker['predict'] - min_predict) / (max_predict - min_predict)

            #split in data_train and data_predict
            data_train = data_ticker[:-window+1]
            data_train.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values
            data_predict = data_ticker[-window+1:]

            #split in X and y to train
            X = data_train.copy()
            X.pop('predict')
            y = data_train['predict']

            #split in X and y to predict
            X_predict = data_predict.copy()
            X_predict.pop('predict')

            #define xgboost model
            train_model = XGBRegressor(objective = 'reg:squarederror')

            param_grid = dict(n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            min_child_weight=min_child_weight,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            gamma=gamma,
                            reg_alpha=reg_alpha)
        
            #perform walk forward validation
            n_splits = min(math.floor(len(data_train)/test_size),max_n_splits)
            my_cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size).split(X)
            grid = GridSearchCV(estimator=train_model, param_grid=param_grid, cv=my_cv, scoring='neg_mean_absolute_error', verbose=False, n_jobs=n_jobs)

            grid_result = grid.fit(X, y)
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            #print('cross validation complete.')
            #print('best validation MAE: %f using %s' % (-grid_result.best_score_, grid_result.best_params_))

            #set up prediction model
            predict_model = XGBRegressor(objective='reg:squarederror',
                                        n_estimators=grid_result.best_params_['n_estimators'],
                                        max_depth=grid_result.best_params_['max_depth'],
                                        learning_rate=grid_result.best_params_['learning_rate'],
                                        min_child_weight=grid_result.best_params_['min_child_weight'],
                                        subsample=grid_result.best_params_['subsample'],
                                        colsample_bytree=grid_result.best_params_['colsample_bytree'],
                                        gamma=grid_result.best_params_['gamma'],
                                        reg_alpha=grid_result.best_params_['reg_alpha'])

            result = predict_model.fit(X, y, eval_metric=['mae'], verbose = False)
            
            #predict last window
            y_predict = predict_model.predict(X_predict)
            y_predict_denormalized = (max_predict - min_predict)*y_predict + min_predict

            #load results into predict
            predict.loc[predict['ticker'] == ticker,'mae'] = -grid_result.best_score_
            predict.loc[predict['ticker'] == ticker,'predict'] = y_predict_denormalized[-1]
        except: #force values when data length is too small to cross validate 
            predict.loc[predict['ticker'] == ticker,'mae'] = -1
            predict.loc[predict['ticker'] == ticker,'predict'] = -1

    #+MMP 19-02-23
    last_date = data['date'].max()
    predict['predict_date'] = last_date
    #-MMP 19-02-23

    blob = bucket.blob(predict_filename)
    with blob.open("w") as f:
        predict.to_csv(f, index=False)

    return f'Success!'
