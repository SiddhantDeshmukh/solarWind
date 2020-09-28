import pandas as pd



def get_df(filepath, filetype='infer', index=None):
  if filetype == 'infer':
    filetype = filepath.split('.')[-1]

  # Read in file as DataFrame
  if filetype == 'csv':
    df = pd.read_csv(filepath)
  elif filetype == 'pickle' or filetype == 'pkl':
    df = pd.read_pickle(filepath)
  else:
    print(f"Error, filetype {filetype} not understood.")
    return -1

  # Set index
  if index:
    df = df.set_index(index)
    df.sort_index(ascending=True, inplace=True)

  return df

def concat_dfs(df1, df2, how='inner'):
  # Assumes DataFrames have the same index
  concatenated_df = pd.merge(df1, df2, how=how)

  return concatenated_df

def truncate_year_range(df, year_range):
  # Assumes 'df' indexed by Date
  return df[df['DecimalDate'].between(year_range[0], year_range[1])].copy()
  
def get_correlation(df, columns):
    # Pearson correlation coefficient
    corr_df = df[columns].copy()
    
    return corr_df.corr(method='pearson')
