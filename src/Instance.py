# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 12:56:51 2024

@author: MSI-NB
"""
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from decimal import Decimal, getcontext, InvalidOperation

def normalize_column(column):
    # Set the precision for Decimal calculations (adjust as needed)
    getcontext().prec = 28

    def safe_decimal_conversion(value):
        """Safely convert a value to Decimal, returning None if conversion fails."""
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None

    # Convert each value in the column to Decimal, filtering out invalid values
    decimal_column = column.apply(safe_decimal_conversion)

    # Drop rows where conversion failed (i.e., None values)
    decimal_column = decimal_column.dropna()

    # Calculate the minimum and maximum values in Decimal format
    if decimal_column.empty:
        # If the column contains no valid data, return an empty column or handle this as needed
        return pd.Series([0.0] * len(column))
    
    min_val = min(decimal_column)
    max_val = max(decimal_column)

    # Avoid division by zero
    if max_val == min_val:
        # Return a column of zeros (or handle as needed)
        return pd.Series([0.0] * len(column))
    
    # Normalize the column using Decimal arithmetic
    normalized_column = (decimal_column - min_val) / (max_val - min_val)

    # Convert the normalized Decimal values back to float, reindexing to match original column
    return pd.Series([float(f"{x:.{16}f}") for x in normalized_column], index=decimal_column.index).reindex(column.index, fill_value=0.0)


class Instance:
    def __init__(self, path, online = False):
        self.path = path
        self.online = online
    def read(self):
        if False == self.online:
            with open(self.path, 'r') as file:
                # Read the number of data points, features, and labels from the first three lines
                I = int(file.readline().strip())
                J = int(file.readline().strip())
                K = int(file.readline().strip())

                # Initialize lists to hold the feature data and labels
                data = []
                labels = []

                # Read feature data
                for _ in range(I):
                    features = [float(file.readline().strip()) for _ in range(J)]
                    data.append(features)

                # Read label data
                labels = [int(file.readline().strip()) for _ in range(I)]

            # Create DataFrame for features
            df = pd.DataFrame(data, columns=[f'f_{i+1}' for i in range(J)])
            # Add labels to the DataFrame
            df['label'] = labels
            self.df = df
        else:
            data = fetch_ucirepo(name=self.path)
            target_name = data['data']['targets'].columns[0]
            # Normalize each numeric column in the DataFrame
            data_feats = data['data']['features']
            for col in data_feats.columns:
                if data_feats[col].dtype == 'object':
                    data_feats.loc[:, col], _ = pd.factorize(data_feats[col])
            numeric_cols = data_feats.select_dtypes(include=['float64', 'int64'])  # selecting numeric columns
            normalized_data_feats = numeric_cols.apply(normalize_column)
            #
            df = pd.concat([normalized_data_feats, data['data']['targets']], axis=1)
            if df[df.columns[-1]].dtype == 'object':
                df[df.columns[-1]],_ =  pd.factorize(df[df.columns[-1]])
            self.df = df.rename(columns={target_name: 'label'})
    def get_ith_partition(self, i):
        if False == self.online:
            # Find the last occurrence of '/' or '\\'
            last_slash = max(self.path.rfind('/'), self.path.rfind('\\'))

            # Slice the string up to and including the last '/'
            if last_slash != -1:
                path = self.path[:last_slash + 1]  # Include the slash
            else:
                path = self.path  # Return the original path if no separator is found

            path = path + 'partition'+str(i)+'.txt'

            with open(path, 'r') as file:
                # Read the sizes of the train, validation, and test sets
                I_train = int(file.readline().strip())
                I_validation = int(file.readline().strip())
                I_test = int(file.readline().strip())

                # Read the indices for each set
                train_indices = [int(file.readline().strip()) for _ in range(I_train)]
                validation_indices = [int(file.readline().strip()) for _ in range(I_validation)]
                test_indices = [int(file.readline().strip()) for _ in range(I_test)]

                # Create the DataFrame partitions
            train_df = self.df.iloc[train_indices]
            validation_df = self.df.iloc[validation_indices]
            test_df = self.df.iloc[test_indices]

            return train_df, validation_df, test_df
        else:
            # First split: 50% for training, 50% for temp (which will be split into validation and test)
            train_df, temp_df = train_test_split(self.df, test_size=0.5, random_state=i)

            # Second split of the temp_df into validation and test datasets (each 25% of the original)
            validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=i)
            return train_df, validation_df, test_df