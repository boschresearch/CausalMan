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

from typing import Optional

from networkx import DiGraph
from pandas import DataFrame, concat

from FCM_Definitions.parallel_section import make_parallelSection
from line_structure.section import Section
from utils.graph import getSourceNodes
from line_structure.baseline import BaselineStructure

class ParallelSection(BaselineStructure):
    """
    Class to create a single parallelSection. Most of the baseline class
    "baseline_structure" can be inherited.

    The next substructure level are the "Sections",
    which are of type "Section".

    """

    def __init__(
        self,
        name: str,
        name_short: Optional[str] = None,
        config: dict = {},
        structure_type: str = "not specified",
        pSection_ID_num: int = 0,
        observed: bool = True,
    ):
        self._set_default_properties(
            name, name_short=name_short, config=config, structure_type=structure_type
        )
        self.section_dict = {}
        self.section_dict_ID = {}
        self.observed = observed
        self.add_pSection_graph(pSection_ID_num=pSection_ID_num, observed=observed)

    def set_active_Section(self, section: Section, section_ID_num: int = 0, time_0: float = 0.0):
        """
        Method to set a given section in self.structure_dag.
        This is necessary as a parallel section can only have one
        active Section for sampling.
        """
        self.structure_dag = DiGraph()
        # Changing the name of the active section to match the parallel section name
        # (otherwise the merged observed node
        # names will be different.
        section.change_name("Section")
        self.add_Structure(section, typeList=["Section"], enforce=True)
        self.set_time_0(time_0)
        self.add_pSection_graph(pSection_ID_num=section_ID_num, observed=self.observed)
        return

    def set_active_Section_byName(self, sectionName: str, time_0: float = 0.0) -> None:
        """
        Sets the active section in the structure by calling the sectionName.

        Args:
            sectionName (str): The name of the section to activate.
            time_0 (float, optional): The initial time value. Defaults to 0.0.
        """
        if sectionName in self.section_dict.keys():
            self.set_active_Section(
                self.section_dict.get(sectionName, "DummyName"),
                section_ID_num=self.section_dict_ID.get(sectionName, 0),
                time_0=time_0,
            )
        else:
            raise ValueError(
                f"""Impossible to activate Section {sectionName},"""
                """ as it was not found in {self.name}"""
            )
        return

    def add_pSection_graph(self, pSection_ID_num: int = 0, observed: bool = True):
        graph_pSection = make_parallelSection(
            pSection_ID_value="pSection_dummy",
            pSection_ID_mapDict={"pSection_dummy": pSection_ID_num},
            node_prefix=f"{self.name}_",
        )
        self.add_causal_dag(graph_pSection)
        if observed:
            self.define_observed_causal_nodes(
                {f"{self.name}_pSection_ID_num": "Section_ID_num"}
            )
        return

    def add_Section(
        self,
        section,
        section_ID_num: int = 0,
        enforce: bool = False,
        time_0: float = 0.0,
        setActive: bool = True,
    ):
        if (section.name not in self.section_dict.keys()) or enforce:
            self.section_dict.update({section.name: section})
            self.section_dict_ID.update({section.name: section_ID_num})
            if setActive:
                self.set_active_Section(
                    section, section_ID_num=section_ID_num, time_0=time_0
                )

    def set_time_0(self, time: float):
        self.time_0 = time

    @property
    def Sections(self):
        return self.section_dict

    @property
    def activeSection(self):
        return self.get_structures()

    def store_paths_df(self, df):
        self.paths_df = df

    def sample_paths(
        self,
        sample_size=10,
        add=True,
        random_state=10,
        separatorStruct: str = "__",
        nodePrefix: str = "",
        nodeName_cycleTime: str = "",
    ):
        """
        Function to sample the paths in the parallel section.
        It is assumed, that as soon as a section has
        completed all of its steps (independent of the number
        of machines in the section),
        it starts working on the next raw components.
        """
        # FIXME: Currently, a new cycle of a section starts only
        #  if the previous part has been completed.
        # FIXME: However, this does not consider that only the
        #  first machine has to be "free"
        # FIXME: --> Hence, this needs to be adapted to only
        #  consider the first machine of a section.
        self.section_dict.keys()

        list_df = []
        for ind, sectionName in enumerate(self.section_dict.keys()):
            section = self.section_dict[sectionName]

            nodePrefix_new = f"{nodePrefix}{self.name}{separatorStruct}"

            # if available: get the first machine

            name_1stmachine = getSourceNodes(section.structure_dag)
            if len(name_1stmachine) > 0:
                (
                    time_dag_1stMachine,
                    firstNodeTimeInitial_m1,
                    firstNodeTime_m1,
                    lastNodeTime_m1,
                ) = section.Machines[name_1stmachine[0]].get_firstLast_nodes_time_dag(
                    separatorStruct=separatorStruct,
                    nodeName_cycleTime=nodeName_cycleTime,
                )

            # ASSUMPTION: As soon as the first machine in the section
            # has finished and is "empty" again,
            # the next part will enter automatically
            # (--> no time delay in part feeding to machine)
            # --> For this reason nodeAbsTime is chosen
            # to be lastNodeTime_m1
            (
                df_sectionTime,
                firstNodeTimeInitial,
                firstNodeTime,
                lastNodeTime,
            ) = section.sample_time_dag(
                sample_size=sample_size,
                add=True,
                random_state=(ind + 1) * random_state,
                nodeAbsTime=lastNodeTime_m1,
                separatorStruct=separatorStruct,
                nodePrefix=nodePrefix_new,
                nodeName_cycleTime=nodeName_cycleTime,
            )

            if lastNodeTime is not None:
                # FIXME: Adapt this to not automatically work on the lastNodeTime
                #  (which is the last step in the section)
                # FIXME: But that it works based on a pre-defined step.
                df_sectionTime["time_cumsum"] = df_sectionTime[lastNodeTime]
                # df_sectionTime["time_cumsum"] = df_sectionTime[lastNodeTime].cumsum()
                df_sectionTime["section_ID"] = sectionName
            else:
                # FIXME: write proper initialization of df_sectionTime
                df_sectionTime = DataFrame(range(sample_size), columns=["time_cumsum"])
                df_sectionTime["section_ID"] = sectionName
            list_df.append(df_sectionTime)
        if len(list_df) > 0:
            df_sections = concat(list_df, axis=0)
            df_sections_sorted = df_sections.sort_values(by="time_cumsum")
            if lastNodeTime is None:
                df_sections_sorted["time_cumsum"] = 0
                lastNodeTime = "delta_time"
                df_sections_sorted[lastNodeTime] = 0
        else:
            df_sections_sorted = DataFrame()

        if add:
            self.store_paths_df(
                df_sections_sorted.iloc[
                    0 : min(sample_size, df_sections_sorted.shape[0]), :
                ]
            )

        return (
            df_sections_sorted.iloc[
                0 : min(sample_size, df_sections_sorted.shape[0]), :
            ],
            firstNodeTimeInitial,
            firstNodeTime,
            lastNodeTime,
        )
