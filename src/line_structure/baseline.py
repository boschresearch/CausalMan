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
import numpy as np
from pandas import DataFrame
from typing import Optional
from utils.equation import create_distrib_fixed, create_distrib_normal
from utils.graph import (
    dag2fcm,
    get_attr_names,
    get_ltl_of_mp,
    get_mpgood_of_mp,
    get_raw_mp_fromList,
    get_utl_of_mp,
    getSourceNodes,
    getTargetNodes,
    rename_nodes_and_terms,
    set_node_distributions,
    set_nodes_equal,
)


class AbstractBaseline:
    # FIXME: adapt class name? Is it even necessary? Its only purpose
    #  is to enable defining it as argument type

    def __init__(self, name, config={}):
        self.name = name
        self.structure_dag = nx.DiGraph()
        self.causal_dag = nx.DiGraph()


class BaselineStructure(AbstractBaseline):
    """
    Baseline class for "parallel line section", "line section",
    "machine", "step".

    Each instance has three types of DAGs/FCMs attached
    (which can be empty nx.DiGraphs(), though):
    1) a structure_dag (automatically created)
    --> e.g. containing the sequence of substructures, e.g.
          the sections in a line
          the machines in a section
          the steps in a machine
    2) a causal_dag (optional!)
    --> e.g. representing the process executed in a step.
    This can be any DAG/FCM if it is a
    nx.DiGraph(). Note that the structure can only have one causal_dag.
    This causal_dag only acts within the
    baseline_structure, but when merged with other baseline_structures
    will be merged if the node_names are identical.
    3) time_dag (optional!)--> to create a time_stamp for execution.
    The time_stamp can be called using the property
    "time_stamp"
    4) monitoring_dag (optional!) --> to define monitored parameters
    and their tolerance checks. The monitored nodes
    can be called using the property "monitored_nodes"
    5) processResult_dag (optional!) --> to define a combination of
    several monitored parameters to a final
    ProcessResult evaluation. The processResult can be called using
    the property "processResult"

    Also the observed_causalNodes can be stored for each instance,
    which can be set using the method
    define_observed_causal_node(). They can be collected using the
    property "observed_nodes".
    """

    name: str
    name_short: str
    config: dict
    structure_type: str

    def __init__(
        self,
        name: str,
        name_short: Optional[str] = None,
        config: dict = {},
        structure_type: str = "not specified",
    ):
        self._set_default_properties(
            name, name_short=name_short, config=config, structure_type=structure_type
        )

    def _set_default_properties(
        self,
        name: str,
        name_short: Optional[str] = None,
        config: dict = {},
        structure_type: str = "not specified",
    ) -> None:
        """Set default properties of the class.

        Args:
            name (str): Name of the structure.
            name_short (str, optional): Short Name. Defaults to None.
            config (dict, optional): Config dictionary. Defaults to {}.
            structure_type (str, optional): Structure type. Defaults to "not specified".
        """
        self.name = name
        self.add_name_short(name_short)
        self.structure_dag = nx.DiGraph()
        self.causal_dag = nx.DiGraph()
        self.add_config(config)
        self.structure_type = structure_type
        self.observed_causalNodes = {}
        self.prefix_dict = {}
        self.monitoring_dag = nx.DiGraph()
        self.processResult_dag = nx.DiGraph()
        self.time_dag = nx.DiGraph()
        self.time_initial = 0.0
        self.yield_dag = nx.DiGraph()
        self.yield_structure = 1.0
        return

    @property
    def structures(self):
        return self.get_structures()

    def add_name_short(self, name_short: str) -> None:
        """Short name for the structure.

        Args:
            name_short (str): _description_

        Raises:
            ValueError: If name_short is not a string.
        """

        if not isinstance(name_short, str):
            raise ValueError("name_short must be a string")

        if name_short is None:
            self.name_short = self.name
        else:
            self.name_short = name_short
        return

    def add_config(self, config: Dict) -> None:
        """
        The config defines the attribute name used in the nx.DiGraph() in
        self.structure_dag to describe the structure.

        Args:
            config (Dict): Configuration dictionary.
        """
        config_default = {"structure": "structure"}
        self.config = config_default
        # FIXME: add flexible config definition
        return

    # COMMENT THIS  FUNCTION
    def _add_single_Structure(
        self, structure: AbstractBaseline, enforce: bool = False
    ) -> None:
        """
        Adds a single substructure. Only a node will be added to the DAG
        in self.structure_dag. The structure will
        be attached to this node as node attribute in the networkx.DiGraph().
        The name of the node attribute is defined
        in self.config under the key "structure".

        If enforce is True an existing substructure with the same name will
        be replaced by the new one.
        """
        if (structure.name not in self.structure_dag.nodes) or enforce:
            self.structure_dag.add_node(structure.name)
            nx.set_node_attributes(
                self.structure_dag,
                {structure.name: structure},
                self.config.get("structure", None),
            )
        else:
            print(
                f"structure {structure.name} is already in dag and could not be added. "
                f"Set enforce=True to overwrite"
            )
        return

    def _add_single_Structure_typeCheck(
        self, structure: AbstractBaseline, typeList: list[str], enforce: bool = False
    ):
        """
        add a substructure only if the typeCheck is passed
        or if no typeCheck list is defined.
        """
        if (structure.structure_type in typeList) or (len(typeList) == 0):
            self._add_single_Structure(structure, enforce=enforce)
        else:
            print(
                f"the structure {structure.name} is not among allowed "
                f"structure types {typeList}"
            )

    def change_name(self, newName: str) -> None:
        """Changes the name of the structure.

        Args:
            newName (str): New name of the structure.
        """
        if not isinstance(newName, str):
            raise ValueError("newName must be a string")

        self.name = newName
        return

    def add_Structure(
        self,
        structure: AbstractBaseline,
        previous: Optional[AbstractBaseline] = None,
        enforce: bool = False,
        typeList: list[str] = [],
    ) -> None:
        """
        Adds a substructure to this structure (e.g. a section to a station,
        or a machine to a section, or a step to a
        machine).

        If enforce is True an existing substructure with the same name
        will be replaced by the new one. Otherwise the
        new structure will not be added.

        If a previous structure is defined the new structure will be added
        after the previous structure in the
        self.structure_dag.

        If a typeList is provided only structures of a certain typeList
        can be added.

        Args:
            structure (baseline_class): Structure to be added.
            previous (baseline_class, optional): Previous structure. Defaults to None.
            enforce (bool, optional): Enforce adding. Defaults to False.
            typeList (list[str], optional): Type list. Defaults to [].

        """
        # FIXME: What happens if the previous structure already has a
        #  successor structure? --> will this simply be
        # added as another child?
        # FIXME: What happens if a structure shall be added in between
        #  two existing structures?
        self._add_single_Structure_typeCheck(structure, typeList, enforce=enforce)
        if previous is not None:
            if all(
                [
                    previous.name != structure.name,
                    # 2nd criteria,
                ]
            ):
                self.structure_dag.add_edge(previous.name, structure.name)
                if not nx.is_directed_acyclic_graph(self.structure_dag):
                    raise ValueError(
                        "With this edge, the graph is not a directed acyclic graph."
                    )
            elif previous.name == structure.name:
                print(
                    f"previous {previous.name} is equal to current "
                    f"structure {structure.name}"
                )
        return

    # FIXME: Add method to retrieve a structure with a specific name
    #  (optional specifying hierarchical levesl in which
    # to search or to specify a list of eligible structure types).

    def get_structures(self) -> Dict:
        """Get all structures.

        Returns:
            Dict: Dictionary of structures.
        """
        structure_dict = dict(self.structure_dag.nodes(data=True))
        return {
            n: v.get(self.config.get("structure", None), None)
            for n, v in structure_dict.items()
        }

    def get_structureType_dags(
        self,
        getSubs: bool = True,
        typeList: list[str] = ["not specified"],
        flipHierarchy: bool = False,
        addPrefix: bool = True,
        separator_string: str = "__",
        nodeAttrName: str = "StructureType",
        dagList: list[str] = [],
    ):
        """
        get the structure_dag for all types.
        This represents the logical order of execution
        of the various structures.

        If getSubs is True the right structure_type will
        be searched in the
        substructures.
        Only the first structure level will be kept.
        """
        if ("not specified" in typeList) or (self.structure_type in typeList):
            structure_dag = nx.DiGraph()
            structure_dag.add_node(self.name)
            # add the structure type as attribute
            nx.set_node_attributes(structure_dag, self.structure_type, nodeAttrName)
            if len(dagList) > 0:
                if "causal_dag" in dagList:
                    nx.set_node_attributes(structure_dag, self.causal_dag, "causal_dag")
                if "time_dag" in dagList:
                    nx.set_node_attributes(structure_dag, self.time_dag, "time_dag")
                if "monitoring_dag" in dagList:
                    nx.set_node_attributes(
                        structure_dag, self.monitoring_dag, "monitoring_dag"
                    )
                if "processResult_dag" in dagList:
                    nx.set_node_attributes(
                        structure_dag, self.processResult_dag, "processResult_dag"
                    )
                if "yield_dag" in dagList:
                    nx.set_node_attributes(structure_dag, self.yield_dag, "yield_dag")
                # FIXME: This is strictly not a dag --> change type of input
                #  into function
                if "observed_nodes" in dagList:
                    # FIXME: For some reason the following command did not work
                    #  as an attribute can't be a dictionary
                    # nx.set_node_attributes(structure_dag, self.observed_nodes.copy(),
                    #                       "observed_nodes")
                    nx.set_node_attributes(
                        structure_dag, [self.observed_nodes.copy()], "observed_nodes"
                    )
        else:
            structure_dag = nx.DiGraph()
        if getSubs and len(self.structures) > 0:
            structure_dag_subs = self.get_structureType_dags_fromSubs(
                typeList=typeList,
                nodeAttrName=nodeAttrName,
                flipHierarchy=flipHierarchy,
                separator_string=separator_string,
                addPrefix=addPrefix,
                dagList=dagList,
            ).copy()
            if addPrefix:
                map_dict = {
                    n: f"{self.name}{separator_string}{n}"
                    for n in list(structure_dag_subs.nodes)
                }
                structure_dag_subs = nx.relabel_nodes(structure_dag_subs, map_dict)
        else:
            structure_dag_subs = nx.DiGraph()

        if len(structure_dag.nodes) > 0 and len(structure_dag_subs.nodes) > 0:
            if flipHierarchy:
                for t in getTargetNodes(structure_dag_subs):
                    structure_dag.add_edge(t, self.name)
            else:
                for s in getSourceNodes(structure_dag_subs):
                    # FIXME: If I add an edge towards a source node, I am adding a loop.
                    structure_dag.add_edge(self.name, s)
            structure_dag_merged = nx.compose(structure_dag, structure_dag_subs)

        elif len(structure_dag.nodes) > 0:
            structure_dag_merged = structure_dag
        elif len(structure_dag_subs.nodes) > 0:
            structure_dag_merged = structure_dag_subs
        else:
            structure_dag_merged = nx.DiGraph()

        structure_dag_merged = self.add_structureName_to_DAG(structure_dag_merged)

        return structure_dag_merged

    def get_structureType_dags_fromSubs(
        self,
        typeList: list[str] = ["not specified"],
        flipHierarchy: bool = False,
        addPrefix: bool = True,
        separator_string: str = "__",
        nodeAttrName: str = "StructureType",
        dagList: list[str] = [],
    ):
        """
        search through all subStructures to collect and merge the
        structure dags.

        Note that this method will call get_structureType_dags()
        on substructure level,
        hence automatically cascading down all structure levels.

        """
        structure_dag_merged = nx.DiGraph()

        list_structures = list(nx.topological_sort(self.structure_dag))
        dict_sub_structure_dags = {}
        target_nodes = []
        for ind, structureName in enumerate(list_structures):
            # print(f"subs {stepName}")
            structure = self.structures[structureName]
            # FIXME: Use improved merge function to account for
            #  merge issues (e.g. identical node names, inconsistent
            # terms for FCM definitions
            sub_structure_dag = structure.get_structureType_dags(
                getSubs=True,
                typeList=typeList,
                flipHierarchy=flipHierarchy,
                addPrefix=addPrefix,
                separator_string=separator_string,
                nodeAttrName=nodeAttrName,
                dagList=dagList,
            ).copy()
            structure_dag_merged = nx.compose(structure_dag_merged, sub_structure_dag)
            dict_sub_structure_dags.update({structureName: sub_structure_dag})
            for pred in self.structure_dag.predecessors(structureName):
                target_nodes = getTargetNodes(dict_sub_structure_dags[pred])
                source_nodes = getSourceNodes(sub_structure_dag)
                edge_list = [(t, s) for t in target_nodes for s in source_nodes]
                structure_dag_merged.add_edges_from(edge_list)

        return structure_dag_merged

    def get_structureType_dag(
        self,
        getSubs: bool = True,
        type: str = "not specified",
        nodeAttrName: str = "StructureType",
    ):
        """
        Get the structure_dag matching a certain type

        If getSubs is True the right structure_type will be searched
        in the substructures.
        Only the first structure level will be kept.
        """
        structure_dict = self.get_structures()

        structure_type_list = []
        for name, structure in structure_dict.items():
            if structure.structure_type == type:
                structure_type_list.append(structure.structure_type)

        # check the type:
        if len(structure_type_list) > 0:
            structure_dag_merged = self.structure_dag.copy()
            # add the structure type as attribute
            nx.set_node_attributes(
                structure_dag_merged, structure_type_list[0], nodeAttrName
            )
        elif getSubs and len(self.structures) > 0:
            structure_dag_merged = self.get_structure_dags_fromSubs(
                type=type, nodeAttrName=nodeAttrName
            ).copy()
        else:
            structure_dag_merged = nx.DiGraph()

        structure_dag_merged = self.add_structureName_to_DAG(structure_dag_merged)

        return structure_dag_merged

    def get_structure_dags_fromSubs(
        self, type: str = "not specified", nodeAttrName: str = "StructureType"
    ):
        """
        Search through all subStructures to collect and merge the structure
        dags.

        Note that this method will call get_structureType_dag() on substructure
        level, hence automatically cascading down all structure levels.
        """
        structure_dag_merged = nx.DiGraph()

        list_structures = list(nx.topological_sort(self.structure_dag))
        target_nodes = []
        for ind, structureName in enumerate(list_structures):
            # print(f"subs {stepName}")
            structure = self.structures[structureName]
            # FIXME: Use improved merge function to account
            #  for merge issues (e.g. identical node names, inconsistent
            # terms for FCM definitions
            structure_dag = structure.get_structureType_dag(
                getSubs=True, type=type, nodeAttrName=nodeAttrName
            ).copy()
            if ind > 0:
                if list_structures[ind - 1] in list(
                    self.structure_dag.predecessors(structureName)
                ):
                    source_nodes = getSourceNodes(structure_dag)
                    if (len(source_nodes) > 0) and (len(target_nodes) > 0):
                        for src in source_nodes:
                            for target in target_nodes:
                                structure_dag_merged.add_edge(target, src)
            structure_dag_merged = nx.compose(structure_dag_merged, structure_dag)
            # nodes_time_end = [n for n in structure_time_dag.nodes
            # if n.endswith("time")]
            target_nodes = getTargetNodes(structure_dag_merged)

        return structure_dag_merged

    def get_downstream_structures(
        self,
        structurePath: list[str],
        typeList=["not specified"],
        flipHierarchy: bool = False,
        separator_string="__",
    ):
        structure_dag = self.get_structureType_dags(
            typeList=typeList,
            flipHierarchy=flipHierarchy,
            addPrefix=True,
            separator_string=separator_string,
            nodeAttrName="StructureType",
        )

        # FIXME: Make proper use of the path_dict
        # desc = nx.descendants(structure_dag, map_dict[structureName])
        anc = list(nx.ancestors(structure_dag, separator_string.join(structurePath)))
        anc.append(separator_string.join(structurePath))

        structure_dag.remove_nodes_from(anc)

        return structure_dag

    def get_upstream_structures(
        self,
        structurePath: list[str],
        typeList=["not specified"],
        flipHierarchy: bool = False,
        separator_string="__",
    ):
        structure_dag = self.get_structureType_dags(
            typeList=typeList,
            flipHierarchy=flipHierarchy,
            addPrefix=True,
            separator_string=separator_string,
            nodeAttrName="StructureType",
        )

        desc = list(nx.descendants(structure_dag, separator_string.join(structurePath)))
        desc.append(separator_string.join(structurePath))

        return structure_dag.remove_nodes_from(desc)

    def add_causal_dag(self, causalDag: nx.DiGraph):
        self.causal_dag = causalDag
        return

    def add_causal_dag_merged(self, causalDag: nx.DiGraph):
        self.causal_dag_merged = causalDag
        return

    def add_all_dags_merged(self, mergedDag: nx.DiGraph):
        self.all_dags_merged = mergedDag
        return

    def add_all_dags(self, mergedDag: nx.DiGraph):
        self.all_dags = mergedDag
        return

    def define_observed_causal_node(
        self,
        node: str,
        displayName: Optional[str] = None,
        overwrite: bool = False,
        allow_nonCausal_nodes: bool = False,
    ):
        """
        method to add a single node to the dictionary self.observed_causalNodes.
        Only nodes which are in the causal_dag
        can be added. No warning will be issued if the node is
        not in the causal_dag.

        If displayName is defined this will be stored as value in
        self.observed_causalNodes.

        If the node is already defined in observed_causalNodes it
        will only be overwritten if the keyword overwrite
        is True (this only matters if the displayName is different).
        """
        # FIXME: also add method to remove a observed causal node?
        if displayName is None:
            displayName = node
        if (node not in self.observed_causalNodes.keys()) or overwrite:
            if node in self.causal_dag.nodes or allow_nonCausal_nodes:
                self.observed_causalNodes.update({node: displayName})

        return

    def define_observed_causal_nodes(
        self,
        nodeDict: dict,
        overwrite: bool = False,
        allow_nonCausal_nodes: bool = False,
    ):
        """
        method to add a dictionary with node definitions to
        self.observed_causalNodes.
        Only nodes which are in the causal_dag can be added.
        No warning will be issued if a node is not
        in the causal_dag.

        If the node is already defined in observed_causalNodes
        it will only be overwritten if the keyword overwrite
        is True (this only matters if the displayName is different).
        """
        # FIXME: Also enable nodeList as input?
        for node, displayName in nodeDict.items():
            self.define_observed_causal_node(
                node,
                displayName=displayName,
                overwrite=overwrite,
                allow_nonCausal_nodes=allow_nonCausal_nodes,
            )

    def remove_observed_causal_node(self, node: str):
        """
        method to remove a single node from the dictionary
        self.observed_causalNodes.
        """
        # FIXME: also add method to remove a observed causal node?
        if node in self.observed_causalNodes.keys():
            self.observed_causalNodes.pop(node)
            # FIXME: add warning if node is not in DAG

    def remove_observed_causal_nodes(self, nodeList: list):
        """
        method to remove a single node from the dictionary
        self.observed_causalNodes.
        """
        for node in nodeList:
            self.remove_observed_causal_node(node)

    def add_structureName_to_DAG(self, DAG):
        """
        Add the name of the structure (self.name) as attribute to the DAG.

        The name of that attribute is f"{self.structure_type}_Name" or
        structureName if self.structure_type is note None
        WARNING: if the same attribute name was already in the DAG it
        will automatically get the suffix "_merge".
        """
        # add structure name to attributes
        if self.structure_type is not None:
            addNodeAttrName = f"{self.structure_type}_Name"
        else:
            addNodeAttrName = "structureName"
        # check if attribute is already in DAG (to prevent overwriting)
        attr_names = get_attr_names(DAG)
        if addNodeAttrName in attr_names:
            addNodeAttrName = f"{addNodeAttrName}_merge"
        nx.set_node_attributes(DAG, self.name, addNodeAttrName)

        return DAG

    def merge_causal_dags(
        self, add: bool = True, add_observedNode_attribute: bool = False
    ):
        """
        merge all causal dags defined in the structure and all substructures.
        Note that this method will automatically
        collect and merge all causal dags from substructures
        (as opposed to the method merge_all_dags())

        If add is True the merged causal dag will be stored in self.causal_dag_merged

        Each node in the networkX DAG will automatically
        get the attribute containing the name of the structure (self.name) as value

        Warning: Identical nodes (= nodes with identical names) will
        automatically be merged into a single node.
        this may break any FCM definition derived from the DAG.
        """
        causal_dags_merged = self.get_causal_dag_fromSubs(add=False)

        causal_dags_merged = self.add_structureName_to_DAG(causal_dags_merged)

        causal_dags_merged_tot = nx.compose(causal_dags_merged, self.causal_dag)

        if add_observedNode_attribute:
            observed_nodes = self.merge_observed_nodes()
            attr_dict = {}
            for n in list(causal_dags_merged_tot.nodes):
                if n in observed_nodes.keys():
                    attr_dict.update({n: "Observable"})
                else:
                    attr_dict.update({n: "not observed"})
            nx.set_node_attributes(causal_dags_merged_tot, attr_dict, "Observable")

        if add:
            self.add_causal_dag_merged(causal_dags_merged_tot)

        return causal_dags_merged_tot

    def get_causal_dag_fromSubs(self, add: bool = True):
        """
        search through all subStructures to collect and merge the causal dags.

        Note that this method will call merge_causal_dags() on
        substructure level, hence automatically cascading
        down all structure levels.

        If add is True the merged dag will be stored in self.causal_dag.

        Warning: Identical nodes (= nodes with identical names)
        will automatically be merged into a single node.
        this may break any FCM definition derived from the DAG.

        """
        causalDag = nx.DiGraph()
        for subStructure in self.structures.values():
            graphSub = subStructure.merge_causal_dags(add=False)
            # FIXME: Use improved merge function to account for merge
            #  issues (e.g. identical node names, inconsistent
            # terms for FCM definitions
            causalDag = nx.compose(causalDag, graphSub)
        if add:
            self.add_causal_dag(causalDag)
        return causalDag

    def merge_all_dags(
        self,
        getSubs: bool = True,
        add: bool = True,
        dagList: list = [
            "causal_dag",
            "time_dag",
            "monitoring_dag",
            "processResult_dag",
        ],
        merge_time_dag: bool = False,
        merge_PR_dag: bool = False,
        merge_yield_dag: bool = False,
        nodeAttrName_dagType: str = "DAG_Type",
        add_observedNode_attribute: bool = False,
        nodeAttrName_observation: str = "Observable",
        separatorStruct: str = "__",
        nodeName_cycleTime: str = "",
        addNameAsPrefix: bool = False
    ):
        """
        merge all dags defined in the structure (causal_dag, time_dag,
         monitoring_dag, processResult_dag).

        If getSubs is True all dags from the substructures will also
        be included in the merged dag (calling the method
        get_all_dags_fromSubs()

        If add is True the merged dag will be stored in self.all_dags.
        """
        if getSubs:
            # remove "time_dag", etc. to make sure it is only merged once
            dagList_red = [dag for dag in dagList]
            dagList_red = [
                dag
                for dag in dagList_red
                if dag not in ["time_dag", "processResult_dag", "yield_dag"]
            ]

            all_dags_merged_tot = self.get_all_dags_fromSubs(
                add=False, dagList=dagList_red
            )
        else:
            all_dags_merged_tot = nx.DiGraph()
        if "time_dag" in dagList:
            if merge_time_dag:
                # getSubs will automatically rename the nodes accordingly
                time_dag = self.get_time_dag(
                    getSubs=getSubs,
                    add=False,
                    linkSteps=True,
                    separatorStruct=separatorStruct,
                    nodeName_cycleTime=nodeName_cycleTime,
                )
            else:
                time_dag = self.get_time_dag(
                    getSubs=getSubs,
                    add=False,
                    linkSteps=False,
                    separatorStruct=separatorStruct,
                    nodeName_cycleTime=nodeName_cycleTime,
                    addNameAsPrefix=addNameAsPrefix
                )

                # time_dag = self.time_dag
            nx.set_node_attributes(time_dag, "time_dag", nodeAttrName_dagType)
            all_dags_merged_tot = nx.compose(all_dags_merged_tot, time_dag)
        if "yield_dag" in dagList:
            if "processResult_dag" in dagList:
                merge_with_PR_dag = True
            else:
                merge_with_PR_dag = False
            if merge_yield_dag:
                yield_dag = self.get_yield_dag(
                    getSubs=getSubs,
                    add=False,
                    # add_observed_causal_node=True,
                    linkSteps=True,
                    merge_with_PR_dag=merge_with_PR_dag,
                )
            else:
                yield_dag = self.get_yield_dag(
                    getSubs=getSubs,
                    add=False,
                    # add_observed_causal_node=True,
                    linkSteps=False,
                    merge_with_PR_dag=merge_with_PR_dag,
                )
                # yield_dag = self.yield_dag
            nx.set_node_attributes(yield_dag, "yield_dag", nodeAttrName_dagType)
            all_dags_merged_tot = nx.compose(all_dags_merged_tot, yield_dag)
        if "processResult_dag" in dagList:
            # if len(self.processResult_dag.nodes) > 0 and "yield_dag" in dagList:
            #    self.merge_yieldDAG_processResult_dags(add=True)
            #    self.merge_processResult_yield_dags(add=True)
            if merge_PR_dag:
                processResult_dag = self.get_processResult_dag(
                    getSubs=getSubs,
                    add=False,
                    # add_observed_causal_node=True,
                )
            else:
                processResult_dag = self.get_processResult_dag(
                    getSubs=getSubs,
                    add=False,
                    # add_observed_causal_node=True,
                )
            nx.set_node_attributes(
                processResult_dag, "processResult_dag", nodeAttrName_dagType
            )
            all_dags_merged_tot = nx.compose(all_dags_merged_tot, processResult_dag)
        if "monitoring_dag" in dagList:
            monitoring_dag = self.monitoring_dag
            nx.set_node_attributes(
                monitoring_dag, "monitoring_dag", nodeAttrName_dagType
            )
            all_dags_merged_tot = nx.compose(all_dags_merged_tot, monitoring_dag)
        if "causal_dag" in dagList:
            causal_dag = self.causal_dag
            nx.set_node_attributes(causal_dag, "causal_dag", nodeAttrName_dagType)
            all_dags_merged_tot = nx.compose(all_dags_merged_tot, causal_dag)
        all_dags_merged_tot = self.add_structureName_to_DAG(all_dags_merged_tot)

        if add_observedNode_attribute:
            # For every node, add a boolean attribute "Observable".
            observed_nodes = self.merge_observed_nodes()
            observed_nodes_raw = [v for dic in observed_nodes for v in dic.keys()]
            attr_dict = {}
            for n in list(all_dags_merged_tot.nodes):
                if n in observed_nodes_raw:
                    attr_dict.update({n: True})
                else:
                    attr_dict.update({n: False})
            nx.set_node_attributes(
                all_dags_merged_tot, attr_dict, nodeAttrName_observation
            )

        if add:
            self.add_all_dags_merged(all_dags_merged_tot)
        return all_dags_merged_tot

    def get_all_dags_fromSubs(
        self,
        add: bool = True,
        dagList: list[str] = [
            "causal_dag",
            "time_dag",
            "monitoring_dag",
            "processResult_dag",
        ],
    ):
        """
        search through all subStructures to collect and merge the dags.

        Note that this method will call merge_all_dags() on
        substructure level, hence automatically cascading
        down all structure levels.

        If add is True the merged dag will be stored in self.all_dags.

        Warning: Identical nodes (= nodes with identical names)
        will automatically be merged into a single node.
        this may break any FCM definition derived from the DAG.

        """
        allDags = nx.DiGraph()
        for subStructure in self.structures.values():
            graphSub = subStructure.merge_all_dags(
                getSubs=True, add=False, dagList=dagList
            )
            # FIXME: Use improved merge function to account for
            #  merge issues (e.g. identical node names, inconsistent
            # terms for FCM definitions
            allDags = nx.compose(allDags, graphSub)
        if add:
            self.add_all_dags(allDags)
        return allDags

    def add_monitoring_dag(self, monitoringDag: nx.DiGraph):
        self.monitoring_dag = monitoringDag

    def create_monitoring_dag(
        self,
        list_monitored_nodes: list,
        node_prefix: str = "",
        sDistributions: dict = {},
        add_processResult: bool = False,
        add_scrapRate: bool = False,
        add_observedCausalNodes: bool = False,
        allow_nonCausal_nodes: bool = False,
    ):
        """
        add a monitoring_dag to the structure. For each node in
        "list_monitored_nodes" an upper and lower tolerance
        limit will be created as source node. The value of the
        monitored node will then be compared to the tolerance
        limits to create MpGood evaluations (1 = ok, 0 = nok).

        Note that only observed_causalNodes can be defined as
        monitored nodes.
        """
        # check which of the nodes is in the causal DAG
        if add_observedCausalNodes:
            self.define_observed_causal_nodes(
                {n: n for n in list_monitored_nodes},
                allow_nonCausal_nodes=allow_nonCausal_nodes,
            )

        list_monitored_nodes = [
            n for n in list_monitored_nodes if n in self.observed_causalNodes.keys()
        ]
        graph = make_processMonitoring(
            list_monitoredNodes=list_monitored_nodes,
            sDistributions=sDistributions,
            node_prefix=node_prefix,
            add_processResult=add_processResult,
        )
        self.add_monitoring_dag(graph)
        return

    def add_processResult_dag(self, processResultDag: nx.DiGraph):
        self.processResult_dag = processResultDag
        self.define_observed_causal_nodes(
            {
                n: "ProcessResult"
                for n in list(processResultDag.nodes)
                if n.endswith("ProcessResult")
            },
            overwrite=True,
            allow_nonCausal_nodes=True,
        )

    @property
    def processResult(self):
        return getTargetNodes(self.processResult_dag)

    def create_processResult_dag(
        self,
        list_monitored_nodes: list,
        node_prefix: str = "",
        sDistributions: dict = {},
        add_yield_dag: bool = True,
    ):
        """
        add a processResult_dag to the structure. For each node in
        "list_monitored_nodes" an upper and lower tolerance
        limit will be created as source node. The value of the
        monitored node will then be compared to the tolerance
        limits to create MpGood evaluations (1 = ok, 0 = nok).
        """
        # check which of the nodes is in the causal DAG
        # list_monitored_nodes = [n for n in list_monitored_nodes if n
        # in self.observed_causalNodes.keys()]
        graph = make_processResult(
            list_monitoredNodes=list_monitored_nodes,
            sDistributions=sDistributions,
            node_prefix=node_prefix,
        )

        # relabel the ProcessResult to make that name unique.
        graph = rename_nodes_and_terms(
            dag=graph,
            mapping={
                f"{node_prefix}ProcessResult": f"{self.name}_{node_prefix}ProcessResult"
            },
        )

        graph = self.add_structureName_to_DAG(graph)

        self.add_processResult_dag(graph)

        if add_yield_dag:
            self.create_yield_dag_normalDistr(prefix=node_prefix)

    def add_merged_processResultDag(self, PR_DAG: nx.DiGraph):
        self.processResult_dag_merged = PR_DAG

    def get_processResult_dag(
        self,
        getSubs: bool = True,
        add: bool = True,
        addNameAsPrefix: bool = False,
        separatorStruct: str = "__",
        separatorParam: str = "_*_",
        # add_observed_causal_node: bool = False,
    ):
        """
        get the process result dag defined in the structure (time_dag).

        If getSubs is True all process result dags from the substructures
        will be included in the merged dag (calling the method
        get_processResult_dags_fromSubs()

        If add is True the merged time dag will be stored in
        self.processResult_merged.

        If addNameAsPrefix is True then all the names of the respective
        substructure will be added as suffix to the
        node name using the specified separator string (keyword "separatorStruct").
        As a consequence, the resulting
        node name will automatically represent the hierarchical
        topology of the substructure
        (e.g. "nodeNameOld" in substructure level 3 becomes
        "substructureNameLevel1__substructureNameLevel2__
        substructureNameLevel3_*_nodeNameOld".
        In this case, the old node name will be the suffix of the new
        old name with separator string specified in "separatorParam".

        If add_observed_causal_node is True, the resulting nodes
        in the dag will be added
        (which is important when merging the function)
        """
        if getSubs and len(self.structures) > 0:
            PR_dag_merged = self.get_processResult_dags_fromSubs(
                add=False,
                addNameAsPrefix=addNameAsPrefix,
                separatorStruct=separatorStruct,
                separatorParam=separatorParam,
                # remove_old_observed_causal_nodes=True,
            ).copy()

            # If we have a yield DAG defined on different structure levels
            # (e.g. on step level and on machine level) we need to merge both
            PR_dag = self.processResult_dag.copy()

            PR_dag_merged = nx.compose(PR_dag_merged, PR_dag)

        else:
            PR_dag_merged = self.processResult_dag.copy()

        PR_dag_merged = self.add_structureName_to_DAG(PR_dag_merged)

        if len(PR_dag_merged.nodes) > 0:
            if add:
                self.add_merged_processResultDag(PR_dag_merged)

        return PR_dag_merged

    def get_processResult_dags_fromSubs(
        self,
        add: bool = True,
        addNameAsPrefix: bool = True,
        separatorStruct: str = "__",
        separatorParam: str = "_*_",
        # remove_old_observed_causal_nodes:
        # bool = False,
    ):
        """
        search through all subStructures to collect and merge the time dags.

        Note that this method will call get_time_dag() on substructure level,
        hence automatically cascading
        down all structure levels.

        If add is True the merged dag will be stored in self.time_dag_merged.

        Warning: Identical nodes (= nodes with identical names) will
        automatically be merged into a single node.
        this may break any FCM definition derived from the DAG.

        If addNameAsPrefix is True then all the names of the respective
        substructure will be added as suffix to the
        node name using the specified separator string (keyword "separatorStruct").
        As a consequence, the resulting
        node name will automatically represent the hierarchical topology
        of the substructure
        (e.g. "nodeNameOld" in substructure level 3 becomes
        "substructureNameLevel1__substructureNameLevel2__substructureNameLevel3
        _*_nodeNameOld".
        In this case, the old node name will be the suffix of the new old name
        with separator string specified in "separatorParam".

        """
        PR_Dag = nx.DiGraph()

        list_structures = list(nx.topological_sort(self.structure_dag))
        for ind, structureName in enumerate(list_structures):
            # print(f"subs {stepName}")
            structure = self.structures[structureName]
            # FIXME: Use improved merge function to account for merge issues
            #  (e.g. identical node names, inconsistent
            # terms for FCM definitions
            structure_PR_dag = structure.get_processResult_dag(
                getSubs=True,
                add=False,
                addNameAsPrefix=addNameAsPrefix,
                separatorStruct=separatorStruct,
                separatorParam=separatorParam,
            ).copy()

            PR_Dag = nx.compose(PR_Dag, structure_PR_dag)

            """
            if len(structure_PR_dag.nodes)>0 and remove_old_observed_causal_nodes:
                structure.remove_observed_causal_nodes(list(structure_PR_dag.nodes))
            """

        # add node prefix
        # relabel the nodes and terms
        if addNameAsPrefix:
            map_dict = {
                name: f"{self.name}{separatorStruct}{name}"
                for name in list(PR_Dag.nodes)
                if name.endswith("ProcessResult")
            }
            PR_Dag = rename_nodes_and_terms(dag=PR_Dag, mapping=map_dict)
            # then we also need to update the observed_nodes_list?

        if add:
            self.add_merged_processResultDag(PR_Dag)
        return PR_Dag

    def adapt_DF_processResult_scrap(
        self,
        DF: DataFrame(),
        mapDict_dag_to_DF: dict,
        separatorStruct: str = "__",
        separatorParam: str = "_*_",
    ):
        # PR_dag = self.get_processResult_dag(separatorStruct="__")
        observed_nodes = self.merge_observed_nodes(
            addNameAsPrefix=True,
            separatorStruct=separatorStruct,
            separatorParam=separatorParam,
        )

        observed_nodes_list = [
            value for entry in observed_nodes for key, value in entry.items()
        ]

        observed_nodes_dict = {
            key: value for entry in observed_nodes for key, value in entry.items()
        }

        mapDict = {
            valueDAG: valueDF
            for keyDAG, valueDAG in observed_nodes_dict.items()
            for keyDF, valueDF in mapDict_dag_to_DF.items()
            if keyDAG == keyDF
        }

        observed_nodes_PR = self.merge_observed_nodes(
            dagList=["processResult_dag"],
            addNameAsPrefix=True,
            separatorStruct=separatorStruct,
            separatorParam=separatorParam,
        )

        observed_nodes_PR_list = [
            value for entry in observed_nodes_PR for key, value in entry.items()
        ]

        for PR in [n for n in observed_nodes_PR_list if n.endswith("ProcessResult")]:
            PR_structureName = PR.split(separatorParam)[0].split(separatorStruct)
            down_dag = self.get_downstream_structures(
                PR_structureName, flipHierarchy=True, separator_string=separatorStruct
            )
            # now get all node names that start with the structure_name
            nodes_NaN = []
            # FIXME: At the moment this is a very hacky solution
            #  (e.g. the flipHierarchy is too strict, that means Line parameters
            #  would be ignored without special treatment)
            PR_spl = PR.split(separatorParam)[0]
            PR_1stTwo = separatorStruct.join(PR_spl.split(separatorStruct)[:2])
            for nObs in observed_nodes_list:
                nObs_spl = nObs.split(separatorParam)[0]
                nObs_spl.split(separatorStruct)
                for nDown in list(down_dag.nodes):
                    if (nObs_spl == nDown and nObs_spl != self.name) and (
                        PR_1stTwo != nObs_spl
                    ):
                        # do not include the line_structure
                        nodes_NaN.append(nObs)

            DF.loc[
                DF[mapDict[PR]] == 0,
                [
                    mapDict[n]
                    for n in nodes_NaN
                    if n in mapDict.keys() and mapDict[n] in DF.columns
                ],
            ] = np.NaN

        return DF

    def add_time_dag(self, timeDAG: nx.DiGraph):
        self.time_dag = timeDAG
        self.define_observed_causal_nodes(
            # FIXME: Logic will break if the time_dag nodes are renamed
            {n: "timeStamp" for n in list(timeDAG.nodes) if n.endswith("time")},
            overwrite=True,
            allow_nonCausal_nodes=True,
        )

    def create_time_dag_normalDistr(
        self,
        delta_t_fromPre_mean: float = 10.0,
        delta_t_fromPre_std: float = 1.0,
        delta_t_step_mean: float = 10.0,
        delta_t_step_std: float = 1.0,
        prefix: str = "",
        time_initial=None,
    ):
        """
        add a time_dag to the structure. The keywords can be used to define
        the respective distribution for the
        - time since previous structure
        - time for executing this structure.

        Note that the time spans can also be defined as zero if both mean
        and std are set to 0.
        If only std is set to zero the time span will be a constant value.

        If time_initial is passed it will be written to self.time_initial.
        """
        sDistributions = {}
        if delta_t_fromPre_std == 0.0:
            Equ = create_distrib_fixed(
                "delta_t_fromPre",
                delta_t_fromPre_mean,
            )
            sDistributions.update({"delta_t_fromPre": Equ})
        else:
            Equ = create_distrib_normal(
                "delta_t_fromPre", delta_t_fromPre_mean, delta_t_fromPre_std
            )
            sDistributions.update({"delta_t_fromPre": Equ})
        if delta_t_fromPre_std == 0.0:
            Equ = create_distrib_fixed(
                "delta_t_step",
                delta_t_step_mean,
            )
            sDistributions.update({"delta_t_step": Equ})
        else:
            Equ = create_distrib_normal(
                "delta_t_step", delta_t_step_mean, delta_t_step_std
            )
            sDistributions.update({"delta_t_step": Equ})
        timeDAG = make_time_graph(
            # input
            time_initial="time_initial",
            delta_time_pre="delta_t_fromPre",
            delta_time="delta_t_step",
            time="time",
            ############
            # interior_template: str = "{}",
            sDistributions=sDistributions,
            node_prefix=f"{self.name}_",
        )

        # FIXME: add method to automatically merge the time dags on Step
        #  level (as the output of the previous dag
        # is input to the subsequent structure).

        if time_initial is not None:
            self.set_time_initial(self, time_initial)

        timeDAG = self.add_structureName_to_DAG(timeDAG)

        self.add_time_dag(timeDAG)

    def set_time_initial(self, time_initial):
        self.time_initial = time_initial

    def add_merged_timeDag(self, timeDAG: nx.DiGraph):
        self.time_dag_merged = timeDAG

    def get_time_dag(
        self,
        getSubs: bool = True,
        add: bool = True,
        linkSteps: bool = True,
        addNameAsPrefix: bool = True,
        separatorStruct: str = "__",
        separatorParam: str = "_*_",
        prefix: str = "",
        nodeName_cycleTime: str = "",
    ):
        """
        Get the time DAG defined in the structure (time_dag).

        Parameters:
            getSubs (bool): If True, include all time DAGs from the
            substructures in the merged DAG.
            add (bool): If True, store the merged time DAG in
            self.time_dag_merged.
            linkSteps (bool): If True, link the individual time DAGs by setting
              the initial_time node equal to the last node of the previous step.
            addNameAsPrefix (bool): If True, add the names of the respective
              substructures as suffixes to the node names using the specified
                separator string (keyword "separatorStruct").
            separatorStruct (str): The separator string used to represent the
              hierarchical topology of the substructure in the node names.
            separatorParam (str): The separator string used to separate the old
              node name from the new node name suffix.

        Returns:
            time_dag_merged (nx.DiGraph): The merged time DAG.

        Notes:
            - If getSubs is True and there are substructures, the time DAGs from
              the substructures will be included in the merged DAG.
            - If add is True, the merged time DAG will be stored in
              self.time_dag_merged.
            - If linkSteps is True, the individual time DAGs will be linked by
              setting the initial_time node equal to the last node of the
                previous step.
            - If addNameAsPrefix is True, the names of the respective
              substructures will be added as suffixes to the node names using
                the specified separator string.
            - If add_observed_causal_node is True, the resulting nodes in the
              time DAG will be added (which is important when merging the
                function).
        """
        if getSubs and len(self.structures) > 0:
            time_dag_merged = self.get_time_dags_fromSubs(
                add=False,
                addNameAsPrefix=addNameAsPrefix,
                separatorStruct=separatorStruct,
                separatorParam=separatorParam,
                linkSteps=linkSteps,
            ).copy()
        else:
            time_dag_merged = self.time_dag.copy()

        time_dag_merged = self.add_structureName_to_DAG(time_dag_merged)

        if len(time_dag_merged.nodes) > 0:
            # set the source node to the initial time
            source_nodes_merged = getSourceNodes(time_dag_merged)
            nodes_time_initial = [
                n for n in source_nodes_merged if n.endswith("time_initial")
            ]
            # nodes_time_initial = getSourceNodes(time_dag_merged)
            if len(nodes_time_initial) > 0:
                # FIXME: write utils function for this
                Equ = create_distrib_fixed(
                    nodes_time_initial[0],
                    self.time_initial,
                )
                sDistributions = {nodes_time_initial[0]: Equ}
                time_dag_merged = set_node_distributions(
                    time_dag_merged, sDistributions
                )

            if nodeName_cycleTime != "":
                # FIXME: Logic breaks if the naming convention for
                #  the time nodes changes
                time_initial = [
                    n
                    for n in getSourceNodes(time_dag_merged)
                    if n.endswith("_time_initial")
                ]
                time_end = [
                    n for n in getTargetNodes(time_dag_merged) if n.endswith("_time")
                ]
                Equ_time_initial = create_distrib_fixed(
                    time_initial[0],
                    0.0,
                )
                Equ_time_end = create_distrib_fixed(
                    time_end[0],
                    999.9,
                )
                graph_cycleTime = make_cycleTime_graph(
                    time_initial=time_initial[0],
                    time_end=time_end[0],
                    delta_time=nodeName_cycleTime,
                    sDistributions={
                        time_initial[0]: Equ_time_initial,
                        time_end[0]: Equ_time_end,
                    },
                )
                self.define_observed_causal_nodes(
                    {nodeName_cycleTime: "delta_time_cycle"},
                    overwrite=True,
                    allow_nonCausal_nodes=True,
                )

                time_dag_merged = nx.compose(time_dag_merged, graph_cycleTime)

            if add:
                self.add_merged_timeDag(time_dag_merged)

        return time_dag_merged

    def get_time_dags_fromSubs(
        self,
        add: bool = True,
        linkSteps: bool = True,
        addNameAsPrefix: bool = True,
        separatorStruct: str = "__",
        separatorParam: str = "_*_",
    ):
        """
        search through all subStructures to collect and merge the time dags.

        Note that this method will call get_time_dag() on substructure level,
        hence automatically cascading
        down all structure levels.

        If add is True the merged dag will be stored in self.time_dag_merged.

        Warning: Identical nodes (= nodes with identical names) will automatically
        be merged into a single node.
        this may break any FCM definition derived from the DAG.

        If linkSteps is true the individual time_dags will be linked by setting
        the initial_time node equal
        to the last node of the previous step.

        If addNameAsPrefix is True then all the names of the respective
        substructure will be added as suffix to the
        node name using the specified separator string (keyword "separatorStruct").
        As a consequence, the resulting
        node name will automatically represent the hierarchical topology
        of the substructure
        (e.g. "nodeNameOld" in substructure level 3 becomes
        "substructureNameLevel1__substructureNameLevel2__substructureNameLevel3
        _*_nodeNameOld".
        In this case, the old node name will be the suffix of the new old name
        with separator string specified in "separatorParam".

        """
        timeDag = nx.DiGraph()

        list_structures = list(nx.topological_sort(self.structure_dag))
        nodes_time_end = []
        for ind, structureName in enumerate(list_structures):
            # print(f"subs {stepName}")
            structure = self.structures[structureName]
            # FIXME: Use improved merge function to account for merge issues
            #  (e.g. identical node names, inconsistent
            # terms for FCM definitions
            structure_time_dag = structure.get_time_dag(
                getSubs=True,
                add=False,
                linkSteps=linkSteps,
                addNameAsPrefix=addNameAsPrefix,
                separatorStruct=separatorStruct,
                separatorParam=separatorParam,
                nodeName_cycleTime="",
            ).copy()
            # FIXME: Also add removal of causal dag nodes on step level for
            #  the time_dag as for yield and ProcessResult

            if ind > 0:
                if list_structures[ind - 1] in list(
                    self.structure_dag.predecessors(structureName)
                ):
                    source_nodes = getSourceNodes(structure_time_dag)
                    nodes_time_initial = [
                        n for n in source_nodes if n.endswith("time_initial")
                    ]
                    # nodes_time_initial = getSourceNodes(structure_time_dag)
                    if (len(nodes_time_initial) > 0) and (len(nodes_time_end) > 0):
                        # FIXME: old way of linking the time dags caused
                        #  unexplainable error.
                        #  Hence set nodes equal instead
                        if linkSteps:
                            structure_time_dag = set_nodes_equal(
                                structure_time_dag,
                                [(nodes_time_end[0], nodes_time_initial[0])],
                            )
            timeDag = nx.compose(timeDag, structure_time_dag)
            nodes_time_end = getTargetNodes(timeDag)

        # add node prefix
        # relabel the nodes and terms
        if addNameAsPrefix:
            map_dict = {
                name: f"{self.name}{separatorStruct}{name}"
                for name in list(timeDag.nodes)
            }
            timeDag = rename_nodes_and_terms(dag=timeDag, mapping=map_dict)

        if add:
            self.add_merged_timeDag(timeDag)
        return timeDag

    def store_time_df(self, df):
        self.time_df = df

    def get_firstLast_nodes_time_dag(
        self,
        separatorStruct: str = "__",
        nodePrefix: str = "",
        nodeName_cycleTime: str = "",
    ):
        timeDag = self.get_time_dag(
            getSubs=True,
            linkSteps=True,
            addNameAsPrefix=True,
            separatorStruct=separatorStruct,
            nodeName_cycleTime=nodeName_cycleTime,
        )
        timeDag = rename_nodes_and_terms(
            timeDag, {n: f"{nodePrefix}{n}" for n in list(timeDag.nodes)}
        )
        if len(timeDag) > 0:
            timeDag_copy = timeDag.copy()
            if nodeName_cycleTime != "":
                timeDag_copy.remove_nodes_from(
                    [
                        n
                        for n in list(timeDag_copy.nodes)
                        if n.endswith(nodeName_cycleTime)
                    ]
                )
            lastNodesTime = getTargetNodes(timeDag_copy)
            # FIXME: Logic will break if the node "time_initial"
            #  gets another name
            firstNodesTime = [
                n for n in getSourceNodes(timeDag_copy) if n.endswith("time_initial")
            ]

            firstNodeTimeInitial = firstNodesTime[0]
            lastNodeName = lastNodesTime[0]
            firstNodeName = list(timeDag_copy.successors(firstNodesTime[0]))[0]
        else:
            firstNodeTimeInitial = None
            lastNodeName = None
            firstNodeName = None

        return timeDag, firstNodeTimeInitial, firstNodeName, lastNodeName

    def sample_time_dag(
        self,
        sample_size=10,
        add: bool = True,
        random_state=10,
        nodeAbsTime=None,
        separatorStruct: str = "__",
        nodePrefix: str = "",
        nodeName_cycleTime: str = "",
    ):
        (
            timeDag,
            firstNodeTimeInitial,
            firstNodeName,
            lastNodeName,
        ) = self.get_firstLast_nodes_time_dag(
            separatorStruct=separatorStruct,
            nodePrefix=nodePrefix,
            nodeName_cycleTime=nodeName_cycleTime,
        )

        if len(timeDag) > 0:
            # transform DAG into pymc Graph Model
            M = dag2fcm(
                timeDag,
                fcm_name="time_dag",
                seed=random_state,
                distributionAttr="source_distribution",
                distributionDefaultAttr="source_distribution_default",
                termAttr="term",
            )
            # M = data.get_wrapping_model(timeDag)
            # set random state
            # M.random = random_state

            # Sample from Model
            if sample_size == 1:
                # is required as the old ml-bn-synthetic-data can't deal with
                # sample_size = 1
                # df = M.sample(n=2, normalize=False, exclude_hidden_nodes=False)
                df = M.sample(size=2, additive_gaussian_noise=False)
                df = df.iloc[0:1, :]
            else:
                # df = M.sample(n=sample_size, normalize=False,
                # exclude_hidden_nodes=False)
                df = M.sample(size=sample_size, additive_gaussian_noise=False)

            # FIXME: Add proper type check
            if nodeAbsTime is not None:
                # note: correction is necessary as nodeAbsTime only describes the ending
                nodeAbsTime_list = [
                    c for c in list(timeDag.nodes) if c.endswith(nodeAbsTime)
                ]
                if nodeAbsTime_list[0] in df.columns:
                    time_add = (
                        df[nodeAbsTime_list[0]].cumsum() - df[nodeAbsTime_list[0]]
                    )
                else:
                    time_add = df[lastNodeName].cumsum() - df[lastNodeName]
                # FIXME: Logic will fail if the time_dag node will be renamed
                col_time = [col for col in df.columns if col.endswith("time")]
                for col in col_time:
                    df[col] = df[col] + time_add
                col_time_initial = [
                    col for col in df.columns if col.endswith("time_initial")
                ]
                for col in col_time_initial:
                    df[col] = df[col] + time_add

            if add:
                self.store_time_df(df)

        else:
            df = DataFrame()

        return df, firstNodeTimeInitial, firstNodeName, lastNodeName

    @property
    def time_stamp(self):
        return getTargetNodes(self.time_dag)

    def add_yield_dag(self, yieldDAG: nx.DiGraph):
        self.yield_dag = yieldDAG
        self.define_observed_causal_nodes(
            {n: "yield" for n in list(yieldDAG.nodes) if n.endswith("yield_absolute")},
            overwrite=True,
            allow_nonCausal_nodes=True,
        )

    def set_yield(self, yield_structure):
        self.yield_structure = yield_structure

    def add_merged_yieldDag(self, yieldDAG: nx.DiGraph):
        self.yield_dag_merged = yieldDAG

    def create_yield_dag_normalDistr(
        self,
        yield_initial_mean: float = 1.0,
        yield_initial_std: float = 0.0,
        yield_mean: float = 1.0,
        yield_std: float = 0.0,
        prefix: str = "",
    ):
        """
        add a yield_dag to the structure. The keywords can be used to define
        the respective distribution for the
         incoming absolute yield (from previous structure)
         yield of the respective process yield (from previous structure)

        If only std is set to zero the yield will be a constant value.

        If yield_mean is passed it will be written to self.yield_structure
        """

        sDistributions = {}
        # sDistributions.update({"time_initial": (pm.DiracDelta.dist, dict(c=0.0))})
        if yield_initial_std == 0.0:
            Equ = create_distrib_fixed(
                "yield_initial",
                yield_initial_mean,
            )
            sDistributions.update({"yield_initial": Equ})
        else:
            Equ = create_distrib_normal(
                "yield_initial", yield_initial_mean, yield_initial_std
            )
            sDistributions.update({"yield_initial": Equ})
        if yield_std == 0.0:
            Equ = create_distrib_fixed(
                "yield_relative",
                yield_mean,
            )
            sDistributions.update({"yield_relative": Equ})
        else:
            Equ = create_distrib_normal("yield_relative", yield_mean, yield_std)
            sDistributions.update({"yield_relative": Equ})

        yieldDAG = make_yield_graph(
            # input
            yield_initial="yield_initial",
            yield_relative="yield_relative",
            yield_absolute="yield_absolute",
            ############
            # interior_template: str = "{}",
            sDistributions=sDistributions,
            node_prefix=f"{self.name}_",
        )

        # FIXME: add method to automatically merge the time dags on Step level
        #  (as the output of the previous dag
        # is input to the subsequent structure).

        yieldDAG = self.add_structureName_to_DAG(yieldDAG)

        if yield_mean is not None:
            self.set_yield(yield_mean)

        self.add_yield_dag(yieldDAG)

    @property
    def yieldRel(self):
        # sNodes = getSourceNodes(self.yield_dag)
        # yieldRel = [n for n in sNodes if n.endswith("_yield_relative")]
        # FIXME: make sure we always find the right node
        yieldRel = [
            n for n in list(self.yield_dag.nodes) if n.endswith("_yield_relative")
        ]
        return yieldRel  # [f"{self.name}_yield_relative"]

    @property
    def yieldAbs(self):
        return getTargetNodes(self.yield_dag)

    def get_yield_dag(
        self,
        getSubs: bool = True,
        add: bool = True,
        linkSteps: bool = True,
        addNameAsPrefix: bool = True,
        separatorStruct: str = "__",
        separatorParam: str = "_*_",
        merge_with_PR_dag: bool = False,
    ):
        """
        get the yield dag defined in the structure (yield_dag).

        If getSubs is True all yield dags from the substructures will
        be included in the merged dag (calling the method
        get_yield_dags_fromSubs()

        If linkSteps is true the individual yield_dags will be linked
        by setting the initial_yield node equal
        to the last node of the previous step.

        If add is True the merged yield dag will be stored in
        self.yield_dag_merged.

        """
        if merge_with_PR_dag and len(self.yield_dag.nodes) > 0:
            self.merge_processResult_yield_dags(add=True)

        if getSubs and len(self.structures) > 0:
            yield_dag_merged = self.get_yield_dags_fromSubs(
                add=False,
                linkSteps=linkSteps,
                addNameAsPrefix=addNameAsPrefix,
                separatorStruct=separatorStruct,
                separatorParam=separatorParam,
            ).copy()

            yield_dag = self.yield_dag.copy()

            # If we have a yield DAG defined on different structure levels
            # (e.g. on step level and on machine level) we need to merge both
            nodes_yield_end = getTargetNodes(yield_dag_merged)

            source_nodes = getSourceNodes(yield_dag)
            nodes_yield_initial = [
                n for n in source_nodes if n.endswith("yield_initial")
            ]

            yield_dag_merged = nx.compose(yield_dag_merged, yield_dag)

            if (len(nodes_yield_initial) > 0) and (len(nodes_yield_end) > 0):
                # FIXME: old way of linking the time dags caused
                #  unexplainable error. Hence set nodes equal instead
                if linkSteps:
                    yield_dag_merged = set_nodes_equal(
                        yield_dag_merged, [(nodes_yield_end[0], nodes_yield_initial[0])]
                    )
        else:
            yield_dag_merged = self.yield_dag.copy()

        yield_dag_merged = self.add_structureName_to_DAG(yield_dag_merged)

        # set the source node to the initial yield
        if len(yield_dag_merged.nodes) > 0:
            target_node = getTargetNodes(yield_dag_merged)
            source_nodes_merged = list(yield_dag_merged.predecessors(target_node[0]))
            # source_nodes_merged = getSourceNodes(yield_dag_merged)
            nodes_yield_initial = [
                n for n in source_nodes_merged if n.endswith("yield_initial")
            ]
            # nodes_yield_initial = getSourceNodes(yield_dag_merged)
            if len(nodes_yield_initial) > 0:
                Equ = create_distrib_fixed(
                    nodes_yield_initial[0],
                    1.0,
                )
                sDistributions = {nodes_yield_initial[0]: Equ}
                yield_dag_merged = set_node_distributions(
                    yield_dag_merged, sDistributions
                )

            if add:
                self.add_merged_yieldDag(yield_dag_merged)

        return yield_dag_merged

    def get_yield_dags_fromSubs(
        self,
        add: bool = True,
        linkSteps: bool = True,
        addNameAsPrefix: bool = True,
        # add: bool = False,
        separatorStruct: str = "__",
        separatorParam: str = "_*_",
        # remove_old_observed_causal_nodes: bool = False,
    ):
        """
        search through all subStructures to collect and merge the yield dags.

        Note that this method will call get_yield_dag() on substructure level,
        hence automatically cascading
        down all structure levels.

        If add is True the merged dag will be stored in self.yield_dag_merged.

        If linkSteps is true the individual yield_dags will be linked by
        setting the initial_yield node equal
        to the last node of the previous step.

        Warning: Identical nodes (= nodes with identical names) will
        automatically be merged into a single node.
        this may break any FCM definition derived from the DAG.

        If addNameAsPrefix is True then all the names of the respective
        substructure will be added as suffix to the
        node name using the specified separator string (keyword "separatorStruct").
        As a consequence, the resulting
        node name will automatically represent the hierarchical
        topology of the substructure
        (e.g. "nodeNameOld" in substructure level 3 becomes
        "substructureNameLevel1__substructureNameLevel2__substructureNameLevel3
        _*_nodeNameOld".
        In this case, the old node name will be the suffix of the new old name
        with separator string specified in "separatorParam".

        """
        yieldDag = nx.DiGraph()

        nodes_yield_end = []

        list_structures = list(nx.topological_sort(self.structure_dag))
        for ind, structureName in enumerate(list_structures):
            # print(f"subs {structureName}")
            structure = self.structures[structureName]
            # FIXME: Use improved merge function to account for merge
            #  issues (e.g. identical node names, inconsistent
            # terms for FCM definitions
            structure_yield_dag = structure.get_yield_dag(
                getSubs=True,
                add=False,
                linkSteps=linkSteps,
                addNameAsPrefix=addNameAsPrefix,
                separatorStruct=separatorStruct,
                separatorParam=separatorParam,
            ).copy()

            # if len(structure_yield_dag.nodes)>0 and remove_old_observed_causal_nodes:
            #    structure.remove_observed_causal_nodes(list(structure_yield_dag.nodes))

            if ind > 0:
                if list_structures[ind - 1] in list(
                    self.structure_dag.predecessors(structureName)
                ):
                    source_nodes = getSourceNodes(structure_yield_dag)
                    nodes_yield_initial = [
                        n for n in source_nodes if n.endswith("yield_initial")
                    ]
                    # nodes_yield_initial = getSourceNodes(structure_yield_dag)
                    if (len(nodes_yield_initial) > 0) and (len(nodes_yield_end) > 0):
                        if linkSteps:
                            # FIXME: old way of linking the time dags caused
                            #  unexplainable error. Hence set nodes equal instead
                            structure_yield_dag = set_nodes_equal(
                                structure_yield_dag,
                                [(nodes_yield_end[0], nodes_yield_initial[0])],
                            )
                    # yieldDag = utils.set_nodes_equal(yieldDag,
                    # {node_yield_end: node_yield_initial})
            yieldDag = nx.compose(yieldDag, structure_yield_dag)
            # nodes_yield_end = [n for n in structure_time_dag.nodes if
            # n.endswith("time")]
            if len(structure_yield_dag.nodes) > 0:
                nodes_yield_end = getTargetNodes(yieldDag)

        # add node prefix
        # relabel the nodes and terms
        if addNameAsPrefix:
            map_dict = {
                name: f"{self.name}{separatorStruct}{name}"
                for name in list(yieldDag.nodes)
            }
            yieldDag = rename_nodes_and_terms(dag=yieldDag, mapping=map_dict)

        if add:
            self.add_merged_yieldDag(yieldDag)
        return yieldDag

    def merge_processResult_yield_dags(self, add: bool = True):
        """
        merge the ProcessResult DAG and Yield dag defined in the structure
        (processResult_dag, yield_dag).

        If add is True the merged dag will be stored in self.yield_dag.
        """
        # dags_merged = self.processResult_dag
        # dags_merged = nx.compose(dags_merged, self.yield_dag)
        dags_merged = self.yield_dag

        # add Level 2 edge from Process Result to Yield
        dags_merged.add_edge(self.processResult[0], self.yieldRel[0], dagLevel=2)

        if add:
            self.add_yield_dag(dags_merged)

        return dags_merged

    @property
    def observed_nodes(self):
        """
        returns all observed_nodes defined in self.observed_causalNodes.
        The following nodes are automatically added to
        this list:
            - all monitored_nodes including their lower and upper tolerance
            limits (LTL, UTL) as well as the
              "MpGood" observation (if monitored_nodes are defined)
            - processResult (if defined)
            - time_stamp (if defined)
            - yield (if defined)

        returns a dictionary with all observed nodes as keys()
        and their display name as values().

        """
        observed_nodes = self.observed_causalNodes
        for mp in self.monitored_nodes:
            observed_nodes.update(
                {get_ltl_of_mp(mp): get_ltl_of_mp(observed_nodes[mp])}
            )
            observed_nodes.update(
                {get_utl_of_mp(mp): get_utl_of_mp(observed_nodes[mp])}
            )
            observed_nodes.update(
                {get_mpgood_of_mp(mp): get_mpgood_of_mp(observed_nodes[mp])}
            )

        return observed_nodes

    def merge_observed_nodes(
        self,
        getSubs: bool = True,
        addNameAsPrefix: bool = False,
        separatorStruct: str = "__",
        separatorParam: str = "_*_",
        dagList: list = [
            "causal_dag",
            "time_dag",
            "monitoring_dag",
            "processResult_dag",
            "yield_dag",
        ],
        rename_observed_nodes: bool = True,
        use_name_short: bool = False
    ):
        """
        Merges the observed nodes by searching through all substructures and
            collecting them.

        Args:
            getSubs (bool, optional): If True, cascades down all structure 
            levels. Defaults to True.
            addNameAsPrefix (bool, optional): If True, adds the names of the
                respective substructures as suffix to the node name. Defaults
                to True.
            separatorStruct (str, optional): Separator string for adding 
            substructure names as suffix. Defaults to "__".
            separatorParam (str, optional): Separator string for adding the
                old node name as suffix. Defaults to "_*_".
            dagList (list, optional): List of substructures to search for
                observed nodes. Defaults to ["causal_dag", "time_dag", 
                "monitoring_dag", "processResult_dag", "yield_dag"].
            rename_observed_nodes (bool, optional): If True, renames the
                observed nodes to ensure only the right nodes are returned.
                Defaults to True.
            use_name_short (bool, optional): If True, uses the short name
                of the substructure as prefix. Defaults to False.

        Returns:
            list: A list of all observed nodes.
        """

        if use_name_short:
            name_prefix = self.name_short
        else:
            name_prefix = self.name

        if addNameAsPrefix:
            observed_nodes = {
                name: f"{name_prefix}{separatorParam}{displayName}"
                for name, displayName in self.observed_nodes.items()
            }
        else:
            observed_nodes = {
                name: displayName for name, displayName in self.observed_nodes.items()
            }

        list_observed_nodes = [{key: value} for key, value in observed_nodes.items()]

        # print(list_observed_nodes)
        if getSubs:
            list_observed_nodes_subs = []
            list_observed_nodes_subs = self.get_observed_nodes_fromSubs(  # add=False,
                dagList=dagList,
                separatorStruct=separatorStruct,
                separatorParam=separatorParam,
                use_name_short=use_name_short,
            )

            # print(list_observed_nodes_subs)
            for entry in list_observed_nodes_subs:
                observed_node_subs = []
                if addNameAsPrefix:
                    observed_node_subs = [
                        {name: f"{name_prefix}{separatorStruct}{displayName}"}
                        for name, displayName in entry.items()
                    ]
                else:
                    observed_node_subs = [
                        {name: displayName} for name, displayName in entry.items()
                    ]
                list_observed_nodes.extend(observed_node_subs)

        if rename_observed_nodes:
            # FIXME: Find more elegant way to ensure only the right
            #  observed nodes are returned in the list
            merged_dag = self.merge_all_dags(
                getSubs=True,
                add=False,
                dagList=dagList,
                separatorStruct=separatorStruct,
            )
            map_dict = {
                value: key
                for key, value in self.get_nodePrefix(
                    separatorParam=separatorStruct
                ).items()
                if key in list(merged_dag.nodes)
            }

            observed_nodes_merged = {
                map_dict.get(key, key): value
                for entry in list_observed_nodes
                for key, value in entry.items()
            }

            observed_nodes_merged_missing = {
                key: value
                for key, value in observed_nodes_merged.items()
                if key not in list(merged_dag.nodes)
            }

            if len(observed_nodes_merged_missing) > 0:
                observed_nodes_merged_missing
        else:
            observed_nodes_merged = {
                key: value
                for entry in list_observed_nodes
                for key, value in entry.items()
            }

        return [{key: value} for key, value in observed_nodes_merged.items()]

    def get_observed_nodes_fromSubs(
        self,
        # add: bool = True,
        separatorStruct: str = "__",
        separatorParam: str = "_*_",
        dagList: list = [
            "causal_dag",
            "time_dag",
            "monitoring_dag",
            "processResult_dag",
            "yield_dag",
        ],
        use_name_short: bool = False,
    ):
        """
        search through all subStructures to collect and merge the observed nodes.

        The name of the respective substructure will be added as suffix to the
        node name using the specified
        separator string (keyword "separatorStruct"). As a consequence, the
        resulting node name will automatically
        represent the hierarchical topology of the substructure (e.g. "nodeNameOld"
        in substructure level 3 becomes
        "substructureNameLevel1__substructureNameLevel2__substructureNameLevel3
        _*_nodeNameOld".

        The old node name will be the suffix of the new old name with separator
        string specified in "separatorParam".

        Note that this method will call merge_observed_nodes() on substructure
         level, hence automatically cascading
        down all structure levels.
        """
        list_observed_nodes_merged = []
        for subStructureName in list(nx.topological_sort(self.structure_dag)):
            # print(f"subs {subStructureName}")
            subStructure = self.structures[subStructureName]
            # print(f"subs2 {subStructure.name}")
            list_observed_nodes_Sub = subStructure.merge_observed_nodes(
                getSubs=True,
                addNameAsPrefix=True,
                dagList=dagList,
                separatorStruct=separatorStruct,
                separatorParam=separatorParam,
                rename_observed_nodes=False,
                use_name_short=use_name_short,
            )
            list_observed_nodes_merged.extend(list_observed_nodes_Sub)
            # print(list_observed_nodes_Sub)
        # if add:
        #    self.define_observed_causal_nodes(observed_nodes, overwrite = True)
        return list_observed_nodes_merged

    def get_structurePrefix(
        self, separatorParam: str = "__", typeList=["not specified"]
    ):
        prefix_dict = {}

        prefix_dict_subs = self.get_structurePrefix_fromSubs(
            separatorParam=separatorParam, typeList=typeList
        )

        if ("not specified" in typeList) or (self.structure_type in typeList):
            prefix_dict.update({self.name: self.name})

            prefix = f"{self.name}{separatorParam}"
            for key, value in prefix_dict_subs.items():
                prefix_dict.update({f"{prefix}{key}": value})

        return prefix_dict

    def get_structurePrefix_fromSubs(
        self, separatorParam: str = "__", typeList=["not specified"]
    ):
        node_prefix_dict = {}

        for subStructureName in list(nx.topological_sort(self.structure_dag)):
            # print(f"subs {subStructureName}")
            subStructure = self.structures[subStructureName]
            # print(f"subs2 {subStructure.name}")
            node_prefix_dict_subs = subStructure.get_structurePrefix(
                separatorParam=separatorParam, typeList=typeList
            )

            for key, value in node_prefix_dict_subs.items():
                node_prefix_dict.update({key: value})

        return node_prefix_dict

    def get_nodePrefix(self, add: bool = True, separatorParam: str = "__"):
        # observed_nodes = {name: f"{self.name}{separatorParam}"
        # for name, displayName in self.observed_nodes.items()}
        prefix_dict = self.prefix_dict.copy()

        prefix_dict.update({key: key for key in self.observed_nodes.keys()})

        prefix_dict_subs = self.get_nodePrefix_fromSubs(separatorParam=separatorParam)

        prefix = f"{self.name}{separatorParam}"
        for key, value in prefix_dict_subs.items():
            prefix_dict.update({key: value})
            prefix_dict.update({f"{prefix}{key}": value})

        if add:
            self.prefix_dict = prefix_dict

        return prefix_dict

    def get_nodePrefix_fromSubs(self, separatorParam: str = "__"):
        node_prefix_dict = {}

        for subStructureName in list(nx.topological_sort(self.structure_dag)):
            # print(f"subs {subStructureName}")
            subStructure = self.structures[subStructureName]
            # print(f"subs2 {subStructure.name}")
            node_prefix_dict_subs = subStructure.get_nodePrefix(
                add=True, separatorParam=separatorParam
            )

            for key, value in node_prefix_dict_subs.items():
                node_prefix_dict.update({key: value})

        return node_prefix_dict

    def sample_causalDAG_merged(self, nSamples: int = 5):
        self.merge_causal_dags()
        # transform DAG into pymc Graph Model
        # Model = data.get_wrapping_model(self.causal_dag_merged)
        Model = dag2fcm(
            self.causal_dag_merged,
            FCMname="causal_dag_merged",
            # seed=random_state,
            distributionAttr="source_distribution",
            distributionDefaultAttr="source_distribution_default",
            termAttr="term",
        )

        # Sample from Model
        # df = Model.sample(n=nSamples, normalize=False, exclude_hidden_nodes=False)
        df = Model.sample(size=nSamples, additive_gaussian_noise=False)

        return df

    @property
    def monitored_nodes(self):
        return get_raw_mp_fromList(list(self.monitoring_dag.nodes))
