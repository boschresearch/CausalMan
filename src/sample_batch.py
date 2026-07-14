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

import os
import pickle
from typing import Any
from warnings import simplefilter

import pandas as pd
from line_structure.line_structure import line_structure
from utils.data import (generate_interventional_table, merge_strings)
from utils.graph import (read_all_csv_files_from_simulation,
                         sample_CausalGraph, to_graphml)
import re
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


def sample_batch(
        simulation,
    save_path: str,
    fileName_prefix: str,
    random_state_seed: int,
    write_merged_csv_after_each_iteration: bool,
    debug_mode: bool,
    index_subbatch_start: int,
    batch_info_df: pd.DataFrame,
    map_dict_batch_ID: dict,
    ind_subbatch: int,
    subbatch_idx: int,
    subbatch: int,
    save_causal_graph: bool,
    intervention_dict: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, Any, pd.DataFrame]:

    print(f"Batch: {subbatch}")

    # Create Debug and batch Paths
    debug_path = os.path.join(save_path, "DEBUG")
    batch_path = os.path.join(save_path, "batch_data", f"batch_{subbatch}")

    def remove_numeric_suffix(input_string: str) -> str:
        # Use regex to remove the suffix in the form of _number
        return re.sub(r'_\d+$', '', input_string)
    choice = remove_numeric_suffix(simulation)
    simulation_objects_path = os.path.join(os.path.dirname(__file__), "dataset_objects", choice, "batch_data", f"batch_{subbatch}")
    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(batch_path, exist_ok=True)
    os.makedirs(simulation_objects_path, exist_ok=True)

    if debug_mode:
        simplefilter(action="once", category=pd.errors.PerformanceWarning)
        simplefilter(action="once", category=pd.errors.SettingWithCopyWarning)

    # Extract batch information from the general batch_info_df
    batch_df = batch_info_df.loc[batch_info_df["subbatch_ID_unique"] == subbatch, :]
    line_ID_num = batch_df["line_ID_num"].values[0]
    subbatch_ID_num = batch_df.index.values[0]
    sample_size = int(batch_df["batch_size_run"].values[0])

    # Convert interventional string to sympy dictionary
    if intervention_dict != {}:
        print(intervention_dict)
    
    if ind_subbatch == 0:
        line_ID_num_last = line_ID_num
    ind_subbatch += 1

    # Extract product type ID
    map_dict_batch_ID.update(
        {subbatch_ID_num: batch_df["subbatch_ID_unique"].values[0]}
    )

    if sample_size == 0:
        raise ValueError(f"Sample size for sub-batch {subbatch_ID_num} is 0.")

    filepath = os.path.join(simulation_objects_path, f"production_line_object.pkl")
    with open(filepath, "rb") as f:
        production_line: line_structure = pickle.load(f)

    # Load subbatch dataframes
    filename = os.path.join(simulation_objects_path, "df_paths_subbatch.pkl")
    df_paths_subbatch = pd.read_pickle(filename)
    filename = os.path.join(simulation_objects_path, "df_timeStamps.pkl")
    df_timeStamps = pd.read_pickle(filename)

    # get the unique path combinations
    colPsectionID = [
        col for col in df_paths_subbatch.columns if col.endswith("_pSection_ID")
    ]

    df_paths_subbatch["path_unique_ID"] = df_paths_subbatch.agg(
        lambda x: merge_strings(
            x, colPsectionID, suffix="_section_ID", separator="___"
        ),
        axis=1,
    )

    # batchdata_df_list contains a list of dataframes, each containing the sampled data for a specific path
    pathdata_df_list = []
    path_df_list = []

    if subbatch_idx >= index_subbatch_start:
        # Iterate over the unique paths
        for path_idx, path in enumerate(df_paths_subbatch["path_unique_ID"].unique()):


            # Extract how many samples followed this specific path
            df_path = df_paths_subbatch.loc[
                df_paths_subbatch["path_unique_ID"] == path, :
            ]
            pSection_ID_dict = {
                df_path[pSection_ID].values[0]: df_path[
                    f"{df_path[pSection_ID].values[0]}_section_ID"
                ].values[0]
                for pSection_ID in colPsectionID
            }
            if debug_mode:
                print(pSection_ID_dict)

            # The DAG Level 2 contains edges that are causal, but on which
            #  we cannot perform ancestral sampling due to the missing
            #  structural equations.
            # This happens mostly because those edges are related to
            #  conditional functions which are not defined in the dag, but
            #  are instead hardcoded into the simulator code.

            # Load the graph models
            filepath = os.path.join(simulation_objects_path, f"dag_level_2_{path_idx}.pkl")
            with open(filepath, "rb") as f:
                dag_level_2 = pickle.load(f)
            filepath = os.path.join(simulation_objects_path, f"dag_level_1_{path_idx}.pkl")
            with open(filepath, "rb") as f:
                dag_level_1 = pickle.load(f)
            filepath = os.path.join(simulation_objects_path, f"production_line_{path_idx}.pkl")
            with open(filepath, "rb") as f:
                production_line: line_structure = pickle.load(f)
            

            if save_causal_graph:
                observed_nodes_list = production_line.merge_observed_nodes()
                save_graph_data(
                    fileName_prefix, batch_path, dag_level_2, observed_nodes_list
                )

            # Sample all the data for this path
            pathdata_df = sample_CausalGraph(
                dag_level_1,
                sample_size=df_path.shape[0],
                random_state=random_state_seed
                + 10000
                + 30 * subbatch_idx
                + 300 * path_idx,
                interventions=intervention_dict,
            )

            pathdata_df_list.append(pathdata_df)
            path_df_list.append(df_path)

            # Once we merge all the dataframes for each individual path, we
            #  will get the dataframe for the whole batch/subbatch

        batchdata_df = pd.concat(pathdata_df_list)
        path_df = pd.concat(path_df_list)

        # now reorder according to the old order
        batchdata_df["partID"] = path_df["partID"].values
        batchdata_df = batchdata_df.sort_values(by="partID").reset_index(drop=True)

        # now replace the time_stamps with the old sampled values to
        # match the path structure
        col_timeStamps = {}
        for col in batchdata_df.columns:
            for col_timestamp in df_timeStamps.columns:
                if col.endswith(col_timestamp):
                    col_timeStamps[col] = col_timestamp

        for col, col_timestamp in col_timeStamps.items():
            batchdata_df[col] = df_timeStamps[col_timestamp].values

        # Generate interventional table
        if intervention_dict is not None:
            int_table_df = generate_interventional_table(
                intervention_dict, batchdata_df
            )


        # create an overview on the csv files created so far
        # initialize the overview file for this batch
        if write_merged_csv_after_each_iteration:
            _ = read_all_csv_files_from_simulation(
                save_path,
                fileName_prefix,
                random_state_seed,
                suffix="_allColumns_raw",
                write_to_csv=True,
            )

    return batchdata_df, df_path, production_line, dag_level_2, int_table_df


def save_graph_data(fileName_prefix, batch_path, dag_level_2, observed_nodes_list):
    # Save as pickle
    filepath = os.path.join(batch_path, f"batch_graph.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(dag_level_2, f)
    # Save as graphml
    to_graphml(dag_level_2, fileName_prefix, batch_path)
    # Save the list of observable nodes on the batch_path

    filepath = os.path.join(batch_path, f"observed_nodes_list.txt")
    with open(filepath, "w") as f:
        for item in observed_nodes_list:
            f.write("%s\n" % item)
