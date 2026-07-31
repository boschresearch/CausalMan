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

from concurrent.futures import ProcessPoolExecutor

import tqdm
from pandas import DataFrame

from sample_batch import sample_batch  


def run_sampling_sequential(
    simulation,
    fileName_prefix: str,
    save_path: str,
    random_state_seed: int,
    write_merged_csv_after_each_iteration: bool,
    debug_mode: bool,
    index_subbatch_start,
    batch_info_df: DataFrame,
    ind_subbatch,
    save_causal_graph: bool = False,
    intervention_dict: dict = {},
):
    path_df_list: list[DataFrame] = []
    batchdata_df_list: list[DataFrame] = []
    map_dict_batch_ID: dict = {}
    interventional_table_list: list[DataFrame] = []

    for subbatch_idx, subbatch in enumerate(
        batch_info_df["subbatch_ID_unique"].unique()
    ):
        batchdata_df, path_df, production_line, dag_level_2, int_table = sample_batch(
            simulation,
            save_path,
            fileName_prefix,
            random_state_seed,
            write_merged_csv_after_each_iteration,
            debug_mode,
            index_subbatch_start,
            batch_info_df,
            map_dict_batch_ID,
            ind_subbatch,
            subbatch_idx,
            subbatch,
            save_causal_graph=save_causal_graph,
            intervention_dict=intervention_dict,
        )
        batchdata_df_list.append(batchdata_df)
        path_df_list.append(path_df)
        interventional_table_list.append(int_table)

    return (
        production_line,
        dag_level_2,
        batchdata_df_list,
        path_df_list,
        interventional_table_list,
    )


def parallel_sample_batch(args):
    """
    Wrapper function to unpack arguments and call the actual function.
    This is necessary because the executor's map function expects a function
    taking a single argument.
    """
    res = sample_batch(*args)
    return res


def run_sampling_parallel(
    simulation,
    fileName_prefix,
    save_path,
    random_state_seed,
    write_merged_csv_after_each_iteration,
    debug_mode,
    index_subbatch_start,
    batch_info_df,
    ind_subbatch,
    max_workers: int = 1,
    save_causal_graph: bool = False,
    intervention_dict={},
):
    path_df_list: list[DataFrame] = []
    batchdata_df_list: list[DataFrame] = []
    map_dict_batch_ID: dict = {}
    interventional_table_list: list[DataFrame] = []

    # Prepare arguments for each function call

    args_list = [
        (
            simulation,
            save_path,
            fileName_prefix,
            random_state_seed,
            write_merged_csv_after_each_iteration,
            debug_mode,
            index_subbatch_start,
            batch_info_df,
            map_dict_batch_ID,
            ind_subbatch,
            subbatch_idx,
            subbatch,
            save_causal_graph,
            intervention_dict,
        )
        for subbatch_idx, subbatch in enumerate(
            batch_info_df["subbatch_ID_unique"].unique()
        )
    ]

    # Use ProcessPoolExecutor to parallelize the loop
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(parallel_sample_batch, args_list))

    # Unpack results
    batchdata_df_list = [result[0] for result in results]
    path_df_list = [result[1] for result in results]
    production_line = [result[2] for result in results][0]
    dag_level_2 = [result[3] for result in results][0]
    interventional_table_list = [result[4] for result in results]

    return (
        production_line,
        dag_level_2,
        batchdata_df_list,
        path_df_list,
        interventional_table_list,
    )
