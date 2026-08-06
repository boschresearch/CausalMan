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
from datetime import datetime
from enum import Enum
from numbers import Integral

import pandas as pd

from .graph_projections import (
    admg2mag,
    get_latent_projection_single as latent_projection,
    validate_mag,
)
from .utils.data import clean_df
from .utils.sampling import run_sampling_parallel, run_sampling_sequential


class CausalManChoice(Enum):
    CAUSALMAN_SMALL = "causalman_small"
    CAUSALMAN_MEDIUM = "causalman_medium"
    CAUSALMAN_LARGE = "causalman_large"
    CAUSALMAN_MICRO = "causalman_micro"


class CausalMan:
    _APPROXIMATE_SAMPLES_PER_BATCH = 14_000

    def __init__(
        self,
        name: str = "causalman_small",
        seed: int = 42,
        batch_multiplier: int = 1,
        parallelize: bool = False,
        max_workers: int = 5,
        debug_mode: bool = False,
        save_path: str = None,
    ):
        if name not in CausalManChoice._value2member_map_:
            raise ValueError(
                f"Invalid choice '{name}'. Valid choices are: {[e.value for e in CausalManChoice]}"
            )
        self.name = CausalManChoice(name)
        self.simulations = self._get_simulations()
        self.write_merged_csv_after_each_iteration = False
        self.random_state_seed = seed
        self.batch_multiplier = batch_multiplier
        self.parallelize = parallelize
        self.max_workers = max_workers
        self.debug_mode = debug_mode
        self.save_path = save_path
        self.save_causal_graph = False
        self.intervention_dict = {}
        return

    def _get_simulations(self):
        # What we call "simulations" are different products on the same production line
        # Each product e.g. causalman_small_1 and causalman_small_2, has the same causal structure
        # But they have a different parameterization of the SCM. One might be more noisy than the other, or exhibit an higher amount of anomalies.
        # Different products have also different material and geometrical properties, so their failure modes might be different.

        if self.name == CausalManChoice.CAUSALMAN_SMALL:
            return ["causalman_small_1", "causalman_small_2"]
        elif self.name == CausalManChoice.CAUSALMAN_MEDIUM:
            return ["causalman_medium_1", "causalman_medium_2"]
        elif self.name == CausalManChoice.CAUSALMAN_LARGE:
            return ["causalman_large_1", "causalman_large_2", "causalman_large_3"]
        elif self.name == CausalManChoice.CAUSALMAN_MICRO:
            return ["causalman_micro_1"]
        else:
            raise ValueError(f"Invalid choice '{self.name}'.")

    def apply_interventions(self, intervention_string):
        self.intervention_dict = {}  ## NO INTERVENTION FOR NOW
        return

    def sample(self, n_samples: int = None):
        """Generate data and optionally return exactly ``n_samples`` aligned rows."""
        if n_samples is not None:
            if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
                raise TypeError("n_samples must be a positive integer or None.")
            if n_samples <= 0:
                raise ValueError("n_samples must be a positive integer.")
            n_samples = int(n_samples)

            # A multiplier of one produces roughly 14,000 rows. Use ceiling
            # division so enough batches are generated before exact row sampling.
            required_batch_multiplier = max(
                1,
                (
                    n_samples
                    + self._APPROXIMATE_SAMPLES_PER_BATCH
                    - 1
                )
                // self._APPROXIMATE_SAMPLES_PER_BATCH,
            )
            effective_batch_multiplier = max(
                1,
                self.batch_multiplier,
                required_batch_multiplier,
            )
        else:
            effective_batch_multiplier = max(1, self.batch_multiplier)

        all_batches_df_list = []
        all_interventional_tables_list = []

        # Parse the configuration
        fileName_prefix = self.name.value
        for k, simulation in enumerate(self.simulations):
            print(
                f"Starting simulation for production line {k} out of {len(self.simulations)}"
            )

            index_subbatch_start = 0

            # Generate Batch description
            batch_info_path = os.path.join(
                os.path.dirname(__file__),
                "dataset_objects",
                self.name.value,
                f"{simulation}_batch_info.csv",
            )
            if batch_info_path is not None:
                batch_info_df = pd.read_csv(batch_info_path)
            else:
                raise ValueError("No batch info file found")

            if effective_batch_multiplier > 1:
                batch_info_df = pd.concat(
                    [batch_info_df] * effective_batch_multiplier,
                    ignore_index=True,
                )

            # Initialize the batch data
            ind_subbatch = 0

            if self.parallelize:
                (
                    _production_line,
                    dag,
                    batchdata_df_list,
                    _path_df_list,
                    int_table_list,
                ) = run_sampling_parallel(
                    simulation,
                    fileName_prefix,
                    self.save_path,
                    self.random_state_seed,
                    False,
                    self.debug_mode,
                    index_subbatch_start,
                    batch_info_df,
                    ind_subbatch,
                    max_workers=self.max_workers,
                    save_causal_graph=self.save_causal_graph,
                    intervention_dict=self.intervention_dict,
                )
            else:
                (
                    _production_line,
                    dag,
                    batchdata_df_list,
                    _path_df_list,
                    int_table_list,
                ) = run_sampling_sequential(
                    simulation,
                    fileName_prefix,
                    self.save_path,
                    self.random_state_seed,
                    False,
                    self.debug_mode,
                    index_subbatch_start,
                    batch_info_df,
                    ind_subbatch,
                    save_causal_graph=self.save_causal_graph,
                    intervention_dict=self.intervention_dict,
                )

            all_batches_df_list.append(
                pd.concat(batchdata_df_list, ignore_index=True)
            )
            all_interventional_tables_list.append(
                pd.concat(int_table_list, ignore_index=True)
            )

        # Merge all dataframes.
        fully_observable_dataset = pd.concat(
            all_batches_df_list, ignore_index=True
        )
        interventional_table = pd.concat(
            all_interventional_tables_list,
            axis=0,
            sort=False,
            ignore_index=True,
        ).fillna(0)
        interventional_table = interventional_table.astype(int)

        print("Finished sampling")

        # Sometimes sympy outputs a table where the values are still sympy objects.
        # So it's better to cast everything into numeric format.
        fully_observable_dataset = clean_df(
            fully_observable_dataset, list(fully_observable_dataset.columns)
        ).reset_index(drop=True)
        interventional_table = interventional_table.reset_index(drop=True)

        if len(interventional_table) != len(fully_observable_dataset):
            raise RuntimeError(
                "The generated dataset and interventional table have different "
                "numbers of rows and cannot be sampled consistently."
            )

        # Keep only graph nodes declared observable and carrying non-constant data.
        # Intervention targets remain observable when a hard intervention makes them
        # constant, so observed columns stay available in interventional datasets.
        dataset_columns = set(fully_observable_dataset.columns)
        intervention_targets = set(self.intervention_dict)
        observable_nodes = []
        for node, attrs in dag.nodes(data=True):
            is_observable = attrs.get("Observable") in (True, "Observable")
            if is_observable and node in dataset_columns:
                is_constant = (
                    fully_observable_dataset[node].nunique(dropna=False) <= 1
                )
                if is_constant and node not in intervention_targets:
                    is_observable = False
            else:
                is_observable = False

            dag.nodes[node]["Observable"] = is_observable
            if is_observable:
                observable_nodes.append(node)

        obs_dataset = fully_observable_dataset.loc[:, observable_nodes].copy()

        admg = latent_projection(dag.copy())
        mag = admg2mag(admg)
        validate_mag(mag)

        if n_samples is not None:
            if len(fully_observable_dataset) < n_samples:
                raise RuntimeError(
                    f"The simulator generated {len(fully_observable_dataset):,} rows "
                    f"with batch_multiplier={effective_batch_multiplier}, but "
                    f"n_samples={n_samples:,} were requested."
                )

            sampled_indices = fully_observable_dataset.sample(
                n=n_samples,
                random_state=self.random_state_seed,
            ).index

            obs_dataset = obs_dataset.loc[sampled_indices].reset_index(drop=True)
            fully_observable_dataset = fully_observable_dataset.loc[
                sampled_indices
            ].reset_index(drop=True)
            interventional_table = interventional_table.loc[
                sampled_indices
            ].reset_index(drop=True)
        else:
            obs_dataset = obs_dataset.reset_index(drop=True)

        return (
            obs_dataset,
            mag,
            admg,
            fully_observable_dataset,
            dag,
            interventional_table,
        )


# Example usage
if __name__ == "__main__":
    random_state_seed = 234
    debug_mode = False
    parallelize = False
    max_workers = 5
    choice = "causalman_small"
    n_samples = 10_000

    experiments_path = os.path.join(os.getcwd(), "output")

    # Create a new directory name with the current date, time, and experiment name
    filename_prefix = choice
    now = datetime.now().strftime("%H_%M_%S")
    exp_dir_name = f"{filename_prefix}_{now}"
    save_path = os.path.join(experiments_path, exp_dir_name)
    os.makedirs(save_path, exist_ok=True)

    simulator = CausalMan(
        name=choice,
        seed=random_state_seed,
        parallelize=parallelize,
        max_workers=max_workers,
        debug_mode=debug_mode,
        save_path=save_path,
    )

    simulator.intervention_dict = {"PF_M1_T1_sgrad": 18500}

    (
        obs_dataset,
        mag,
        admg,
        fully_observable_dataset,
        dag,
        interventional_table,
    ) = simulator.sample(n_samples=n_samples)

    ### END HERE ###
