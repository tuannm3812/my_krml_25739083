import pandas as pd

def pop_target(df, target_col):
    """Extract target variable from dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Name of the target variable

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    pd.Series
        Subsetted Pandas dataframe containing the target
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)

    return df_copy, target

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """
    import numpy as np

    if X_train is not None:
      np.save(f'{path}X_train', X_train)
    if X_val is not None:
      np.save(f'{path}X_val',   X_val)
    if X_test is not None:
      np.save(f'{path}X_test',  X_test)
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
    if y_val is not None:
      np.save(f'{path}y_val',   y_val)
    if y_test is not None:
      np.save(f'{path}y_test',  y_test)

def load_sets(path='../data/processed/'):
    """Load the different locally save sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    import numpy as np
    import os.path

    X_train = np.load(f'{path}X_train.npy', allow_pickle=True) if os.path.isfile(f'{path}X_train.npy') else None
    X_val   = np.load(f'{path}X_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}X_val.npy')   else None
    X_test  = np.load(f'{path}X_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy', allow_pickle=True) if os.path.isfile(f'{path}y_train.npy') else None
    y_val   = np.load(f'{path}y_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}y_val.npy')   else None
    y_test  = np.load(f'{path}y_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}y_test.npy')  else None

    return X_train, y_train, X_val, y_val, X_test, y_test

def subset_x_y(target, features, start_index:int, end_index:int):
    """Keep only the rows for X and y (optional) sets from the specified indexes

    Parameters
    ----------
    target : pd.DataFrame
        Dataframe containing the target
    features : pd.DataFrame
        Dataframe containing all features
    features : int
        Index of the starting observation
    features : int
        Index of the ending observation

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing the target
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    """

    return features[start_index:end_index], target[start_index:end_index]

def split_sets_by_time(df, target_col, test_ratio=0.2):
    """Split sets by indexes for an ordered dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(df_copy) / 5)

    X_train, y_train = subset_x_y(target=target, features=df_copy, start_index=0, end_index=-cutoff*2)
    X_val, y_val     = subset_x_y(target=target, features=df_copy, start_index=-cutoff*2, end_index=-cutoff)
    X_test, y_test   = subset_x_y(target=target, features=df_copy, start_index=-cutoff, end_index=len(df_copy))

    return X_train, y_train, X_val, y_val, X_test, y_test

def split_sets_random(features, target, test_ratio=0.2):
    """Split sets randomly

    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe
    target : pd.Series
        Target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    from sklearn.model_selection import train_test_split

    val_ratio = test_ratio / (1 - test_ratio)
    X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8)

    return X_train, y_train, X_val, y_val, X_test, y_test

def convert_to_date(df, cols:list):
    """Convert specified columns from a Pandas dataframe into datetime

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : list
        List of columns to be converted

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with converted columns
    """
    import pandas as pd

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df

def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.metrics import mean_absolute_error as mae

    print(f"RMSE {set_name}: {rmse(y_actuals, y_preds)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")

