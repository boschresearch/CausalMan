# Code for the CausalMan: A Digital-Twin for Large-Scale Causality
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Util module that contains functions that operate on data and dataframes.


import os
from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import Optional
from col_masking import mask_columns


def merge_strings(df: DataFrame, colNames: list, suffix: str, separator: str):
    col_merged = ""
    for ind, col in enumerate(colNames):
        col_section_ID = f"{df[col]}{suffix}"
        if ind == 0:
            col_merged = df[col_section_ID]  # .astype(str)
        else:
            col_merged = f"{col_merged}{separator}{df[col_section_ID]}"
    return col_merged


def clean_df(df, vars):
    for col in vars:
        if col in df.columns:
            df[col] = df[col].map(to_scalar)
            try:
                df[col] = df[col].astype(float)
            except Exception:
                pass  # If conversion fails, leave as is
        else:
            print(f"Warning: {col} not found in DataFrame columns")
    return df

def generate_interventional_table(
    intervention_dict: dict[str, float], data_df: DataFrame
) -> DataFrame:
    """This function generates an interventional table, which contains
    a one-hot encoding for every sample, indicating on which node the
    intervention is active.

    Args:
        intervention_dict (dict[str, float]): Dictionary containing the interventions.
        pathdata_df (DataFrame): Dataframe containing the sampled data.

    Returns:
        DataFrame: Dataframe containing the interventional table.
    """
    # Create dataframe with same columns as the interventional dictionary
    col_names = [str(col) for col in list(intervention_dict.keys())]
    return DataFrame(1, index=range(data_df.shape[0]), columns=col_names)


def extract_df_ending(df: DataFrame, col_suffix: str):
    new_df = [col for col in df.columns if col.endswith(col_suffix)]
    return new_df


def prepare_and_store_data(
    data_df: DataFrame,
    write_csv: bool = True,
    save_path: os.PathLike = "",
    prefix: str = "Test1_",
    time_start: float = 0,
    observable_nodes: Optional[list] = None,
    all_nodes: Optional[list] = None,
    interventions_dict: dict = None
):
    # assume mean cycle time of 20s.
    # FIXME: Include time calculation in classes
    """
    if col_time_firstStep in df.columns:
        time_series_initial = df[col_time_firstStep].cumsum() -
        df[col_time_firstStep] + time_start
    else:
        time_series_initial = pd.Series(np.arange(start=0,
        step=20, stop=sample_size*20)) + time_start
    """

    # Convert the interventions dict to a string do(X=1, Y=2)
    if interventions_dict is not None:
        for key, value in interventions_dict.items():
            if "sympy.stats" in str(value):
                interventions_dict[key] = interventions_dict[key].replace("sympy.stats.", "")
                interventions_dict[key] = interventions_dict[key].replace(f"\"{key}\",", "")
                interventions_dict[key] = interventions_dict[key].replace(" ", "")
        interventions_str = ", ".join([f"{key}={value}" for key, value in interventions_dict.items()])
        prefix_filename_save = f"{prefix}_{interventions_str}"
    else:
        prefix_filename_save = prefix

    # Write the raw data to CSV
    if write_csv:
        data_df.to_csv(
            os.path.join(save_path, f"{prefix_filename_save}_allColumns_raw.csv")
        )

    # recalculate the timeStamp to create a consecutive list
    # FIXME: Do this directly as part of the Step baseline structure class
    # FIXME: This will fail if the naming convention for the time headers
    #  changes (e.g. initial_time instead of time_initial)
    col_time = [col for col in data_df.columns if col.endswith("_time")]

    fully_observable_df = data_df.copy()
    max_time = 0
    for ind, time in enumerate(col_time):
        max_time = max(max_time, fully_observable_df[time].max())
        fully_observable_df[time] = fully_observable_df[time] + time_start

    # Prepare data
    fully_observable_df.to_csv(
        os.path.join(save_path, f"{prefix_filename_save}_fullyObservable_causal_unnormalized.csv"),
        index=False,
    )

    if observable_nodes is not None:
        masked_proj_df = mask_columns(observable_nodes, fully_observable_df)
        masked_proj_df.to_csv(
            os.path.join(save_path, f"{prefix_filename_save}_partiallyObservable_causal_masked_unnormalized.csv"),
            index=False,)
        
    # Remove those columns that are not variables in the causal dag, such as partID
    if all_nodes is not None:
        masked_df = mask_columns(all_nodes, fully_observable_df)
        masked_df.to_csv(
            os.path.join(save_path, f"{prefix_filename_save}_fullyObservable_causal_masked_unnormalized.csv"),
            index=False,)


    return 


def to_scalar(x):
    """Convert mixed/obj-dtype cell values to a Python float or np.nan.
    Handles SymPy numeric types, numpy scalars, lists/arrays/Series, and numeric strings.
    Always returns a float or np.nan and never raises.
    """
    try:
        if x is None:
            return np.nan
        # SymPy numbers
        try:
            import sympy as _sp
            if isinstance(x, _sp.Basic):
                return float(_sp.N(x))
        except Exception:
            pass
        # Native numeric types
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x) if np.isfinite(x) else np.nan
        # Strings that might represent numbers
        if isinstance(x, str):
            try:
                return float(x)
            except Exception:
                return np.nan
        # Sequences / arrays / Series: coerce then pick first finite element
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(x)
            try:
                arrf = arr.astype(float).ravel()
            except Exception:
                return np.nan
            if arrf.size == 0:
                return np.nan
            for v in arrf:
                if np.isfinite(v):
                    return float(v)
            return np.nan
        # Fallback: try float conversion
        return float(x)
    except Exception:
        return np.nan
