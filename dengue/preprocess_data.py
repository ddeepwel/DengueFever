"""
@author: daviddeepwell
"""

import pandas as pd

def preprocess_data(data_path, labels_path=None):
    # load the provided data
    train_features = pd.read_csv(\
        data_path, index_col=[0])
    if labels_path:
        train_labels = pd.read_csv(\
            labels_path, index_col=[0])
    
    # separate San Juan and Iquitos
    sj_data = train_features.loc['sj']
    iq_data = train_features.loc['iq']
    if labels_path:
        sj_labels = train_labels.loc['sj']
        iq_labels = train_labels.loc['iq']
    
    # Set index to the week start date
    sj_data.set_index(pd.to_datetime(sj_data['week_start_date'].values), inplace=True)
    iq_data.set_index(pd.to_datetime(iq_data['week_start_date'].values), inplace=True)

    # drop unneeded columns
    sj_data.drop(columns=['year','week_start_date'], inplace=True)
    iq_data.drop(columns=['year','week_start_date'], inplace=True)
    
    # add case counts to table
    if labels_path:
        sj_data['Counts'] = sj_labels['total_cases'].values
        iq_data['Counts'] = iq_labels['total_cases'].values

    # fill missing values
    #df_orig.fillna(method='ffill', inplace=True) # use forward fill
    sj_data.interpolate(method='linear', inplace=True) # use linear interpolation
    iq_data.interpolate(method='linear', inplace=True) # use linear interpolation
    
    return sj_data, iq_data