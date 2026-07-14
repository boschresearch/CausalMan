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

from typing import Dict, Optional
import networkx as nx
from numpy import NaN
from pandas import DataFrame, concat

from line_structure.parallelsection import ParallelSection
from line_structure.baseline import BaselineStructure


class line_structure(BaselineStructure):
    """
    Class to create a single production line. Most of the baseline
    class "baseline_structure" can be inherited.

    The next substructure level are the "parallelSections",
    which are of type "pSection".

    """

    def add_parallelSection(
        self,
        section,
        previous=None,
        enforce=False,
        time_0: float = 0.0,
        nodeName_lineID: str = "line_ID_num",
    ):
        self.add_Structure(
            section, previous=previous, typeList=["pSection"], enforce=enforce
        )
        # add dependency of selected nodes to machine_ID
        graph_lineID = nx.DiGraph()
        graph_lineID.add_edge(
            nodeName_lineID, f"{section.name}_pSection_ID_num", dagLevel=2
        )
        graph = nx.compose(self.causal_dag, graph_lineID)
        self.add_causal_dag(graph)
        if previous is None:
            self.set_time_0(time_0)

    def set_activeSections(self, active_pSections_dict: Dict) -> None:
        """Set the active path for each parallel section.

        Args:
            active_pSections_dict (Dict): Dictionary with the active path for 
            each parallel section.
        """

        for pSection_name, pSection in self.pSections.items():
            if pSection_name in active_pSections_dict.keys():
                pSection.set_active_Section_byName(active_pSections_dict[pSection_name])
        return

    def set_time_0(self, time: float):
        self.time_0 = time

    @property
    def pSections(self):
        return self.get_structures()

    def store_paths_df(self, df: DataFrame):
        self.paths_df = df

    def sample_all_paths(
        self,
        sample_size: Optional[int] = 10,
        random_state: int = 10,
        separatorStruct: str = "__",
        nodeName_cycleTime: str = "",
        store_internally: bool = True,
    ) -> tuple[DataFrame, DataFrame]:
        """For every sample, which correspond to a part, sample the path that it
          will follow through the production line.

        Args:
            sample_size (int, optional): Number of paths to sample. Defaults to 10.
            add (bool, optional): _description_. Defaults to True.
            random_state (int, optional): _description_. Defaults to 10.
            separatorStruct (str, optional): _description_. Defaults to "__".
            nodeName_cycleTime (str, optional): _description_. Defaults to "".

        Returns:
            tuple[DataFrame, DataFrame]: Dataframe with paths, and dataframe 
            with timestamps.
        """
        # FIXME: Write the separatorStruct as an attribute of the class itself
        pSection_order = list(nx.topological_sort(self.structure_dag))

        if sample_size == 0:
            raise ValueError("Sample size cannot be 0")

        df_list = []
        # initialize the part IDs:
        df_ID_start = DataFrame(range(sample_size), columns=["partID"])
        df_ID_start["time"] = range(sample_size)
        col_time_min = None
        col_time_max = None
        for ind, pSectionName in enumerate(pSection_order):
            pSection: ParallelSection = self.pSections[pSectionName]

            (
                df_sections_sorted,
                firstNodeTimeInitial,
                firstNodeTime,
                lastNodeTime,
            ) = pSection.sample_paths(
                sample_size=sample_size,
                add=False,
                random_state=(ind + 1) * random_state,
                separatorStruct=separatorStruct,
                nodePrefix=f"{self.name}{separatorStruct}",
            )
            if df_sections_sorted.shape[0] > 0:
                df_sections_sorted = df_sections_sorted.reset_index()
                # now extract the first parts
                df_ID_start[f"{pSectionName}_pSection_ID"] = pSectionName
                df_ID_start[f"{pSectionName}_section_ID"] = (
                    df_sections_sorted["section_ID"]
                    .iloc[0 : df_ID_start.shape[0]]
                    .values
                )
                df_ID_start[f"{pSectionName}_time_end"] = (
                    df_sections_sorted["time_cumsum"]
                    .iloc[0 : df_ID_start.shape[0]]
                    .values
                )
                df_ID_start[f"{pSectionName}_delta_time"] = (
                    df_sections_sorted[lastNodeTime]
                    .iloc[0 : df_ID_start.shape[0]]
                    .values
                )

            if ind > 0:
                df_ID_start[f"{pSectionName}_time_end_tot"] = (
                    df_ID_start[f"{pSectionName}_time_end"]
                    + df_ID_start[f"{pSection_order[ind-1]}_time_end_tot"]
                )
                for col in df_sections_sorted.columns:
                    if col.endswith("_time"):
                        df_sections_sorted[col] = (
                            df_sections_sorted[col]
                            + df_ID_start[f"{pSection_order[ind-1]}_time_end_tot"]
                        )
            else:
                df_ID_start[f"{pSectionName}_time_end_tot"] = df_ID_start[
                    f"{pSectionName}_time_end"
                ]

            # remember the column names to calculate the total time delta
            if lastNodeTime is not None:
                col_time_max = lastNodeTime
            # FIXME: Doing this with firstNodeTimeInitial is currently
            #  not possible as it is not added with the time delta
            if (col_time_min is None) and (firstNodeTimeInitial is not None):
                col_time_min = firstNodeTimeInitial

            # add
            df_list.append(df_sections_sorted)

        df_timeStamps = concat(df_list, axis=1)
        if nodeName_cycleTime != "":
            if (col_time_max is not None) and (col_time_min is not None):
                df_timeStamps[nodeName_cycleTime] = (
                    df_timeStamps[col_time_max] - df_timeStamps[col_time_min]
                )
            else:
                df_timeStamps[nodeName_cycleTime] = NaN

        if store_internally:
            self.store_paths_df(df_ID_start)

        return df_ID_start, df_timeStamps
