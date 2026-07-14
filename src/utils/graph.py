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

import pickle

from sympy.stats.rv import RandomSymbol
from fcm import FCM
import networkx as nx
import pandas as pd
import os
from sympy import Symbol, symbols, Eq

from typing import Dict, List
from node import NodeModel_sympy


def get_attr_names(dag: nx.DiGraph) -> List[str]:
    """Get all attribute names of the nodes in a directed graph.

    Args:
        dag (nx.DiGraph): Input dag.

    Returns:
        List[str]: _description_
    """
    # FIXME: Is there a networkX function doing the same?
    node_dict = dict(dag.nodes(data=True))
    attr_names = list(
        set([value for item in node_dict.values() for value in list(item.keys())])
    )
    return attr_names


def getSourceNodes(dag: nx.DiGraph) -> List:
    """Given a directed graph, return a list of source nodes.

    Args:
        dag (nx.DiGraph): Input DAG.

    Returns:
        List: List of source nodes
    """

    # Check if DAG
    # if not nx.is_directed_acyclic_graph(dag):
    #    raise ValueError("Input graph is not a directed acyclic graph.")

    listSources = [n for n in dag.nodes if len(dag.in_edges(n)) == 0]
    return listSources


def getTargetNodes(dag: nx.DiGraph) -> List:
    """Given a directed graph, return a list of target nodes.

    Args:
        dag (nx.DiGraph): Input DAG.

    Returns:
        List: List of target nodes
    """

    # Check if DAG
    # if not nx.is_directed_acyclic_graph(dag):
    #    raise ValueError("Input graph is not a directed acyclic graph.")

    listTargets = [n for n in dag.nodes if len(dag.out_edges(n)) == 0]
    return listTargets


def add_default_GraphAttributes_perNode(
    dag: nx.DiGraph, attributes_dict: dict
) -> nx.DiGraph:
    """Given a directed graph and a dictionary of attributes, add the attributes
      to the nodes.

    Args:
        graph (nx.DiGraph): Graph to which the attributes should be added.
        attributes_dict (dict): Dictionary of attributes to be added to the
          nodes.

    Returns:
        nx.DiGraph: Graph with added attributes.
    """

    # Check if DAG
    # if not nx.is_directed_acyclic_graph(dag):
    #    raise ValueError("Input graph is not a directed acyclic graph.")

    for name, node_map_dict in attributes_dict.items():
        nx.set_node_attributes(dag, node_map_dict, name=name)
    return dag


def add_default_GraphAttributes(graph: nx.DiGraph, attributes_dict: dict) -> nx.DiGraph:
    for name, value in attributes_dict.items():
        # node_map_dict.update({name:{n: value for n in graph.nodes}})
        nx.set_node_attributes(graph, {n: value for n in graph.nodes}, name=name)
    # graph = add_default_GraphAttributes_perNode(graph, node_map_dict)
    return graph


def remove_attributes_dag(dag: nx.DiGraph, nodeAttributes_remove: list):
    dagOut = dag.copy()
    for node in list(dagOut.nodes):
        for attr in nodeAttributes_remove:
            dagOut.nodes[node].pop(attr, None)
    return dagOut


def get_mp_with_suffix(mp, suffix=""):
    return f"{mp}_{suffix}"


def get_ltl_of_mp(mp):
    return get_mp_with_suffix(mp, "LTL")


def get_utl_of_mp(mp):
    return get_mp_with_suffix(mp, "UTL")


def get_mpgood_of_mp(mp):
    return get_mp_with_suffix(mp, "MpGood")


def get_raw_mp_fromList(mp_list):
    LTL_list = set([mp for mp in mp_list if mp.endswith("_LTL")])
    UTL_list = set([mp for mp in mp_list if mp.endswith("_UTL")])
    MpGood_list = set([mp for mp in mp_list if mp.endswith("_MpGood")])
    return list(
        set(mp_list).difference(LTL_list).difference(UTL_list).difference(MpGood_list)
    )


def set_node_distributions(
    dag: nx.DiGraph,
    distribution_dict: Dict[str, Eq],
    setRaw=True,
    sourceDistributionAttr: str = "source_distribution",
    sourceDistributionRawAttr: str = "source_distribution_raw",
):
    # FIXME: source_distribution_raw is no longer needed, can be removed
    distribution_dict_transf = {
        src_node: dist_definition  # dist_definition[0](**dist_definition[1])
        for src_node, dist_definition in distribution_dict.items()
    }
    # Assign default distributions to source nodes
    nx.set_node_attributes(dag, distribution_dict_transf, name=sourceDistributionAttr)
    if setRaw:
        nx.set_node_attributes(dag, distribution_dict, name=sourceDistributionRawAttr)
    return dag


def set_nodes_equal(graph, edge_list, add_attribute=True, add_node=True):
    attributes_dict = {}
    for edge in edge_list:
        if (edge[0] not in graph.nodes) and add_node:
            graph.add_nodes_from([edge[0]])
        if (edge[1] not in graph.nodes) and add_node:
            graph.add_nodes_from([edge[1]])

        # delta_p = AleakSum * V_chamber_REF / V_chamber
        Equ = Eq(symbols(edge[1]), symbols(edge[0]))
        NM = NodeModel_sympy(Equ)
        graph = NM.add_to_DAG(graph)

        # FIXME: Deal with the fact that a node can be equal to several other nodes
        attributes_dict.update({"node_is_equal_to": {edge[1]: edge[0]}})

    if add_attribute:
        graph = add_default_GraphAttributes_perNode(graph, attributes_dict)
    return graph


def rename_nodes_and_terms(
    dag: nx.DiGraph,
    mapping: dict,
    termAttr: str = "term",
    sourceDistributionAttr: str = "source_distribution",
    sourceDistributionDefaultAttr: str = "source_distribution_default",
):
    """
    rename the nodes and terms in a network x dag.
    The dag needs the attribute "term", which is either
         None or defined using the bn_testing.terms class

    Returns the transformed input DAG containing renamed nodes and adjusted terms

    """
    # mapping_inv = {value: key for key, value in mapping.items()}
    dag = nx.relabel_nodes(dag, mapping)
    # cycle through the nodes with the attribute term to rename the parents
    # in the term equation
    for node in list(dag.nodes):
        term = dag.nodes[node].get(termAttr, None)
        distribution = dag.nodes[node].get(sourceDistributionAttr, None)
        distribution_default = dag.nodes[node].get(sourceDistributionDefaultAttr, None)

        if term is not None:
            term_lhs = symbols(
                node
            )  # term.lhs.subs({symbols(mapping_inv[node]): symbols(node)})
            # dag.nodes[node][termAttr] = Eq(term_lhs, term.rhs)
            term_rhs = term.rhs
            if len(term.rhs.free_symbols) > 0:
                for symbol in term.rhs.free_symbols:
                    term_rhs = term_rhs.subs(
                        {
                            symbols(str(symbol)): symbols(
                                mapping.get(str(symbol), str(symbol))
                            )
                        }
                    )  # term.parents = [mapping.get(p, p) for p in term.parents]
            elif len(term.rhs.free_symbols) == 0:
                term_rhs = symbols(mapping.get(str(term.rhs), str(term.rhs)))

            dag.nodes[node][termAttr] = Eq(term_lhs, term_rhs)

        if distribution is not None:
            term_lhs = symbols(
                node
            )  # distribution.lhs.subs({symbols(mapping_inv[node]): symbols(node)})
            term_rhs = distribution.rhs
            if len(distribution.rhs.free_symbols) > 0:
                for symbol in distribution.rhs.free_symbols:
                    term_rhs = term_rhs.subs(
                        {
                            symbols(str(symbol)): symbols(
                                mapping.get(str(symbol), str(symbol))
                            )
                        }
                    )  # term.parents = [mapping.get(p, p) for p in term.parents]
            # elif len(distribution.rhs.free_symbols) == 0:
            #    term_rhs = symbols(mapping.get(str(distribution.rhs),
            #                                   str(distribution.rhs)))

            # term_rhs = distribution.rhs.subs({symbols(mapping_inv[node]):
            # symbols(node)})

            dag.nodes[node][sourceDistributionAttr] = Eq(term_lhs, term_rhs)

        if distribution_default is not None:
            term_lhs = symbols(
                node
            )  # distribution_default.lhs.subs({symbols(mapping_inv[node]):
            # symbols(node)})
            term_rhs = distribution_default.rhs
            if len(distribution_default.rhs.free_symbols) > 0:
                for symbol in distribution_default.rhs.free_symbols:
                    term_rhs = term_rhs.subs(
                        {
                            symbols(str(symbol)): symbols(
                                mapping.get(str(symbol), str(symbol))
                            )
                        }
                    )  # term.parents = [mapping.get(p, p) for p in term.parents]
            # elif len(distribution_default.rhs.free_symbols) == 0:
            #    term_rhs = symbols(mapping.get(str(distribution_default.rhs),
            #                                   str(distribution_default.rhs)))

            # term_rhs = distribution_default.rhs.subs({symbols(mapping_inv[node]):
            # symbols(node)})

            dag.nodes[node][sourceDistributionDefaultAttr] = Eq(term_lhs, term_rhs)

    return dag


def remove_nonFCMedges(dag: nx.DiGraph, termAttr: str = "term") -> nx.DiGraph:
    """
    Remove all edges from the DAG which do not match the parents defined in the
      term equation.


    Args:
        dag (nx.DiGraph): Dag to be cleaned.
        termAttr (str, optional): _description_. Defaults to "term".

    Returns:
        nx.DiGraph: Dag with removed edges.
    """
    dag = dag.copy()
    wrongEdges = []
    for node in list(dag.nodes):
        term = dag.nodes[node].get(termAttr, None)
        # FIXME: Implement proper type check.
        if term is not None:
            if len(term.rhs.free_symbols) > 0:
                parents = [str(symbol) for symbol in term.rhs.free_symbols]
            elif len(term.rhs.free_symbols) == 0:
                # catch the situation where the term represents a simple "equality"
                # for example when the term looks like this:
                # Eq(symbols("y"), symbols("x1"))
                parents = [str(term.rhs)]

            preds = list(dag.predecessors(node))

            wrongEdges_node = [(pre, node) for pre in preds if pre not in parents]
            wrongEdges.extend(wrongEdges_node)

            [(p, node) for p in parents if p not in preds]

        else:
            preds = list(dag.predecessors(node))
            wrongEdges.extend([(pre, node) for pre in preds])

    if len(wrongEdges) > 0:
        dag.remove_edges_from(wrongEdges)
        print(f"Removed {len(wrongEdges)} edges.")
        # raise ValueError("There are edges which do not match the term definition.")

    return dag


def remove_nonFCMnodes(
    dag: nx.DiGraph,
    termAttr: str = "term",
    distributionAttr: str = "source_distribution",
    distributionDefaultAttr: str = "source_distribution_default",
) -> nx.DiGraph:
    """Remove all nodes from the DAG which do not contain a valid term or
    distribution required for conversion into the FCM model.

    Args:
        dag (nx.DiGraph): _description_
        termAttr (str, optional): _description_. Defaults to "term".
        distributionAttr (str, optional): _description_. Defaults to "source_distribution".
        distributionDefaultAttr (str, optional): _description_. Defaults to "source_distribution_default".

    Returns:
        nx.DiGraph: _description_
    """
    dag = dag.copy()
    wrongNodes = []
    for node in list(dag.nodes):
        term = dag.nodes[node].get(termAttr, None)
        distribution = dag.nodes[node].get(distributionAttr, None)
        distribution_default = dag.nodes[node].get(distributionDefaultAttr, None)

        # FIXME: Implement type check to verify that the Equations are of the right type
        if (term is None) and (distribution is None) and (distribution_default is None):
            wrongNodes.append(node)

    if len(wrongNodes) > 0:
        dag.remove_nodes_from(wrongNodes)
        print(f"Removed {len(wrongNodes)} nodes.")
        # raise ValueError("There are nodes which do not contain a valid term or distribution.")
    return dag


def dag2fcm(
    dag: nx.DiGraph,
    fcm_name: str = "converted FCM",
    seed: int = 1000,
    distributionAttr: str = "source_distribution",
    distributionDefaultAttr: str = "source_distribution_default",
    termAttr: str = "term",
):
    """
    Function to convert a DAG into a working causalAssembly FCM
    model which can be used for sampling.

    For this to work, the DAG requires at least the following
    2 attributes:
    1)  an attribute with the name specified in "distributionAttr"
     OR an attribute with the name specified in "distributionDefaultAttr":
     which contains a SymPy equation with the implemented distribution
     for example:
        Eq(symbols("x1"), Normal("x1", 100, 1))
    2) an attribute with the name specified in "termAttr"
     which contains a SymPy equation with the implemented equation
     for example:
        Eq(symbols("y"), symbols("x1") + symbols("x2"))

    Note: All nodes which do not have the required attributes will
    automatically be removed without warning.

    Note: All edges which do not match the term definition (that means
    where the equation does not contain a respective parent as a free
    symbol) will automatically be removed without warning.

    """
    fcm = FCM(name=fcm_name, seed=seed)

    DAG_conv = dag.copy()

    DAG_conv = remove_nonFCMedges(DAG_conv, termAttr=termAttr)
    DAG_conv = remove_nonFCMnodes(
        DAG_conv,
        termAttr=termAttr,
        distributionAttr=distributionAttr,
        distributionDefaultAttr=distributionDefaultAttr,
    )

    # get all nodes which are NOT source term:
    non_sources = [n for n in DAG_conv.nodes if len(DAG_conv.in_edges(n)) > 0]
    # catch missing definitions:
    missing_termDef = [
        ns for ns in non_sources if DAG_conv.nodes[ns].get(termAttr, None) is None
    ]
    if len(missing_termDef) > 0:
        print("Missing Definitions:" + missing_termDef)
        # FIXME: Implement proper catch of wrong definitions / class
        #  definitions
        pass
    equ_dict = {
        ns: DAG_conv.nodes[ns][termAttr]
        for ns in non_sources
        if DAG_conv.nodes[ns].get(termAttr, None) is not None
    }
    mismatch_in_lhs_Term = [n for n, v in equ_dict.items() if n != str(v.lhs)]
    if len(mismatch_in_lhs_Term) > 0:
        print("Mismatch" + mismatch_in_lhs_Term)
        # FIXME: Implement proper catch of wrong definitions / class
        #  definitions
        pass
    # get equations
    equ_list = [
        DAG_conv.nodes[ns][termAttr]
        for ns in non_sources
        if DAG_conv.nodes[ns].get(termAttr, None) is not None
        and ns not in mismatch_in_lhs_Term
    ]

    fcm.input_fcm(equ_list)

    #############################
    # get all nodes which are source terms:
    sources = [n for n in DAG_conv.nodes if len(DAG_conv.in_edges(n)) == 0]
    # catch missing distributions
    missing_sourceDef = [
        s for s in sources if DAG_conv.nodes[s].get(distributionAttr, None) is None
    ]
    if len(missing_sourceDef) > 0:
        print(
            "Missing definition for the following source nodes:"
            + str(missing_sourceDef)
        )
        # FIXME: Implement proper catch of wrong definitions / class
        #  definitions
    missing_sourceDefaultDef = [
        s
        for s in missing_sourceDef
        if DAG_conv.nodes[s].get(distributionDefaultAttr, None) is None
    ]
    if len(missing_sourceDefaultDef) > 0:
        print(missing_sourceDefaultDef)
        pass
    equ_dict_s = {
        s: DAG_conv.nodes[s][distributionAttr]
        for s in sources
        if DAG_conv.nodes[s].get(distributionAttr, None) is not None
    }
    mismatch_in_lhs_Distr = [n for n, v in equ_dict_s.items() if n != str(v.lhs)]
    if len(mismatch_in_lhs_Distr) > 0:
        print("Mismatch in LHS for those nodes:" + list(mismatch_in_lhs_Distr))
        raise ValueError("Mismatch in LHS of distribution definition.")
    # get equations
    equ_list_s = [
        DAG_conv.nodes[s][distributionAttr]
        for s in sources
        if DAG_conv.nodes[s].get(distributionAttr, None) is not None
        and s not in mismatch_in_lhs_Distr
    ]

    ##########################
    # Repeat for s2
    equ_dict_s2 = {
        s: DAG_conv.nodes[s][distributionDefaultAttr]
        for s in sources
        if (DAG_conv.nodes[s].get(distributionAttr, None) is None)
        and (DAG_conv.nodes[s].get(distributionDefaultAttr, None) is not None)
    }
    mismatch_in_lhs_Distr2 = [n for n, v in equ_dict_s2.items() if n != str(v.lhs)]
    if len(mismatch_in_lhs_Distr2) > 0:
        print(mismatch_in_lhs_Distr2)
        # FIXME: Implement proper catch of wrong definitions / class
        #  definitions
        pass
    equ_list_s2 = [
        DAG_conv.nodes[s][distributionDefaultAttr]
        for s in sources
        if (DAG_conv.nodes[s].get(distributionAttr, None) is None)
        and (DAG_conv.nodes[s].get(distributionDefaultAttr, None) is not None)
        and s not in mismatch_in_lhs_Distr2
    ]

    # FIXME: Verify that the Source nodes are transfered correctly
    # fcm.input_fcm(equ_list_s)
    fcm.input_sources(equ_list_s)
    # fcm.input_fcm(equ_list_s2)
    fcm.input_sources(equ_list_s2)

    return fcm


def remove_node_chain_edges(dag: nx.DiGraph, node) -> nx.DiGraph:
    """
    Remove a node and replace all incoming/outgoing edges by chaining them.

    For every combination of incoming and outgoing edges, we create a direct
    edge from the corresponding parent of the node to the corresponding child
    of the node.

    Args:
        g: A directed acyclic graph
        node: The name/id of the node to be removed.
    Returns:
        A copy of the origina graph with node removed and its edges replaced as
        described above
    """
    dag = dag.copy()
    edges_to_delete = set()
    edges_to_add = set()
    for p in dag.predecessors(node):
        edges_to_delete.add((p, node))
        for s in dag.successors(node):
            edges_to_delete.add((node, s))
            edges_to_add.add((p, s))
    dag.add_edges_from(edges_to_add)
    dag.remove_edges_from(edges_to_delete)
    dag.remove_node(node)
    return dag


def remove_nodes_chain_edges(dag: nx.DiGraph, nodeList) -> nx.DiGraph:
    """
    Remove a list of nodes and replace all incoming/outgoing edges by chaining them.

    Applies the function remove_node_chain_edges() to the nodeList

    Args:
        g: A directed acyclic graph
        nodeList: List of names/ids of the node sto be removed.
    Returns:
        A copy of the original graph with nodes removed and its edges replaced as
        described in remove_node_chain_edges()
    """
    dag = dag.copy()
    for node in nodeList:
        if node in list(dag.nodes):
            dag = remove_node_chain_edges(dag, node)
    return dag


def find_pathsBetwNodesets(
    dag: nx.DiGraph,
    nodeSet1: list,
    nodeSet2: list,
    maxPathLength: int = 1000,
    storeAllResults=False,
) -> dict:
    """
    Find all paths between two nodeSets in the graph
    (only searching in direction FROM nodeSet1 TO nodeSet2)

    Nodes which are not in g.nodes will be ignored without warning!

    Args:
        dag (nx.DiGraph): A directed acyclic graph
        nodeSet1 (list): a list/set of source nodes
        nodeSet2 (list): a list/set of target nodes
        maxPathLength (int): integer to define maximum pathLength searched for.
          Defaults to 1000.
        storeAllResults (bool) : Boolean to specify whether individual results
          shall be returned (returned in key dict_all['__individualResults']).
            Defaults to False.

    Returns:
        dict_all: a dictionary on the results
    """
    list_all = []
    dict_all_n1 = {}

    nodeSet1 = [n for n in nodeSet1 if n in list(dag.nodes)]
    nodeSet2 = [n for n in nodeSet2 if n in list(dag.nodes)]

    for n1 in nodeSet1:
        paths_to_n2 = [
            list(nx.all_simple_paths(dag, n1, n2, cutoff=maxPathLength))
            for n2 in nodeSet2
        ]
        paths_to_n2 = [path for path in paths_to_n2 if len(path) > 0]
        paths_to_n2 = cleanStrings_pathPath(paths_to_n2)

        list_all.extend(paths_to_n2)

        if storeAllResults:
            n1_connected = list(
                set([path[0] for pathPath in paths_to_n2 for path in pathPath])
            )
            n2_connected = list(
                set([path[-1] for pathPath in paths_to_n2 for path in pathPath])
            )

            # the nodes that could be found in all paths
            n_inPaths = list(
                set([n for pathPath in paths_to_n2 for path in pathPath for n in path])
            )

            dict_out = {
                "paths_to_n2": paths_to_n2,
                "n1_connected": n1_connected,
                "n2_connected": n2_connected,
                "n_inPaths": n_inPaths,
            }
            dict_all_n1[n1] = dict_out

            assert (n1_connected[0] == n1) and (
                len(n1_connected) == 0
            ), "there should only be one n1"

    n1_connected_all = list(
        set([path[0] for pathPath in list_all for path in pathPath])
    )
    n2_connected_all = list(
        set([path[-1] for pathPath in list_all for path in pathPath])
    )

    # the nodes that could be found in all paths
    n_inPaths_all = list(
        set([n for pathPath in list_all for path in pathPath for n in path])
    )

    dict_all = {
        "paths_NS1_to_NS2": list_all,
        "n1_connected": n1_connected_all,
        "n2_connected": n2_connected_all,
        "n_inPaths": n_inPaths_all,
    }

    if storeAllResults:
        dict_all["__individualResults"] = dict_all_n1

    return dict_all


def cleanStrings_path(pathList: list[list], returnTuples: bool = True) -> list:
    path_new = []
    for path in pathList:
        listNew = [str(n) for n in path]
        if returnTuples:
            listNew = tuple(listNew)
        path_new.append(listNew)
    return path_new


def cleanStrings_pathPath(pathPathList, returnTuples=False) -> list:
    pathPath_new = []
    for pathPath in pathPathList:
        pathPath_new.append(cleanStrings_path(pathPath, returnTuples=returnTuples))
    return pathPath_new


def read_all_csv_files_from_simulation(
    storage_path: str,
    fileName_prefix: str,
    random_state_seed: int,
    suffix: str = "_allColumns_raw",
    write_to_csv: bool = True,
) -> pd.DataFrame:
    df_overview_available_csv_files = pd.read_csv(
        os.path.join(
            storage_path,
            f"Overview_available_csvFiles_{fileName_prefix}_RS{random_state_seed}.csv",
        )
    )
    df_all3_list = []
    for ind in range(df_overview_available_csv_files.shape[0]):
        df_temp = df_overview_available_csv_files.iloc[ind, :]
        storage_path2 = df_temp["storage_path"]
        path_rel = df_temp["path_csv_file_prefix"]
        df_data = pd.read_csv(
            os.path.join(
                storage_path2,
                f"{path_rel}{suffix}.csv",
            )
        )
        df_all3_list.append(df_data)

    df_all3 = pd.concat(df_all3_list)

    if write_to_csv:
        df_all3.to_csv(
            os.path.join(
                storage_path,
                f"{fileName_prefix}_RS{random_state_seed}{suffix}_merged.csv",
            )
        )

    return df_all3


def sample_CausalGraph(
    dag: nx.DiGraph,
    sample_size: int = 10,
    random_state: int = 10,  # np.random.RandomState(10)):
    interventions: dict[Symbol, RandomSymbol | float] = {},
) -> pd.DataFrame:
    # Transform networkx DAG into FCM class
    fcm = dag2fcm(
        dag,
        fcm_name="graph_model",
        seed=random_state,
        distributionAttr="source_distribution",
        distributionDefaultAttr="source_distribution_default",
        termAttr="term",
    )

    # Apply interventions if possible
    if interventions:
        # Convert the interventions to the correct format
        fcm.intervene_on(interventions)

        # Convert interventions to string
        interventions_str = [str(k) for k, v in interventions.items()]
        interventions_str = ", ".join(interventions_str)
        interventions_str = f"do([{interventions_str}])"

        # Sample interventional data from Model
        if sample_size == 1:
            samples_df = fcm.interventional_sample(
                size=2,
                additive_gaussian_noise=False,
                which_intervention=interventions_str,
            )
            samples_df = samples_df.iloc[0:1, :]
        else:
            samples_df = fcm.interventional_sample(
                size=sample_size,
                additive_gaussian_noise=False,
                which_intervention=interventions_str,
            )

    else:
        # Sample observational data from Model
        if sample_size == 1:
            samples_df = fcm.sample(size=2, additive_gaussian_noise=False)
            samples_df = samples_df.iloc[0:1, :]
        else:
            samples_df = fcm.sample(size=sample_size, additive_gaussian_noise=False)

    df_out = samples_df[[str(n) for n in fcm.graph.nodes]]

    return df_out


def to_graphml(dag: nx.Graph, filename: str, save_path: str):
    """Export the graph to a graphml file. Furthermore, the edgeList and nodeAttributes
    are exported to csv files.

    Args:
        dag (nx.Graph): Graph to be saved.
        fileName_prefix (str): Prefix for the file name.
        save_path (str): Path to save the files.
    """
    # Remove attributes
    dag = clear_graph(dag)

    # Export edges
    df_edgeList = nx.to_pandas_edgelist(dag)
    df_edgeList.to_csv(os.path.join(save_path, f"{filename}_graph_edgeList.csv"))
    dict_nodeAttributes = dict(dag.nodes(data=True))

    with open(
        os.path.join(save_path, f"{filename}_graph_nodeAttributes.pickle"),
        "wb",
    ) as handle:
        pickle.dump(dict_nodeAttributes, handle)


        
    nx.write_graphml(
        dag,
        os.path.join(save_path, f"{filename}.graphml"),
    )
    return

def clear_graph(dag: nx.DiGraph):
    dag = remove_attributes_dag(
        dag, ["distribution", "distribution_raw", "term", "structure"]
    )
    # Remove all node attributes
    for node in dag.nodes:
        dag.nodes[node].clear()
    return dag
