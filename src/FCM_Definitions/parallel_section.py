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

import networkx as nx
from sympy import Eq, symbols
from sympy.stats import FiniteRV

from fcm import FCM
from node import InputNode
from utils.graph import (add_default_GraphAttributes,
                         add_default_GraphAttributes_perNode, getSourceNodes,
                         rename_nodes_and_terms, set_node_distributions)


def make_parallelSection(
    graph_default_name: str = "make_parallelSection",
    pSection_ID_value: str = "pSection_1",
    pSection_ID_mapDict: dict = {"pSection_1": 1},
    pSection_ID_num: InputNode = "pSection_ID_num",
    ############
    # interior_template: str = "{}",
    sDistributions: dict = {},
    node_prefix: str = "",
) -> nx.DiGraph:
    """
    Build graph to initialize a parallel section and its
    ID as a graph node.

    The nodes are simply initialized as source nodes to
     represent the "raw state" of the MV.

    """

    # FIXME: dummy to avoid adapting code (interior_template is
    #  redundant due to node_prefix input)
    interior_template = "{}"

    node_attributes = {}

    DAG = nx.DiGraph()
    fcm = FCM(name=graph_default_name, seed=2023)

    pSection_ID_num = interior_template.format(pSection_ID_num)

    # INPUT nodes
    symDict = {
        pSection_ID_num: symbols(pSection_ID_num),
    }

    # Default distributions
    dict_sDistr_default = {}

    parents = [
        pSection_ID_num,
    ]
    DAG.add_nodes_from(parents)

    # add pSection_ID_num
    symDict.update({pSection_ID_num: symbols(pSection_ID_num, integer=True)})
    pSection_ID_value_num = pSection_ID_mapDict.get(pSection_ID_value, 0)
    Equ = Eq(
        symDict[pSection_ID_num], FiniteRV(pSection_ID_num, {pSection_ID_value_num: 1})
    )
    DAG.add_node(pSection_ID_num)
    # FIXME: input_fcm does not take the single node as
    #  additional node in this case.
    fcm.input_fcm([Equ])

    sDistributions.update({pSection_ID_num: Equ})

    sDistributions = {
        s: v for s, v in sDistributions.items() if s in getSourceNodes(DAG)
    }
    if len(sDistributions) > 0:
        DAG = set_node_distributions(
            DAG,
            sDistributions,
            sourceDistributionAttr="source_distribution",
            sourceDistributionRawAttr="source_distribution_raw",
        )

    if len(dict_sDistr_default) > 0:
        DAG = set_node_distributions(
            DAG,
            dict_sDistr_default,
            sourceDistributionAttr="source_distribution_default",
            sourceDistributionRawAttr="source_distribution_default_raw",
        )

    node_attributes_transformed = {}
    for node, attribute in node_attributes.items():
        for attribute_name, value in attribute.items():
            node_attributes_transformed.update({attribute_name: {node: value}})
            add_default_GraphAttributes_perNode(DAG, {attribute_name: {node: value}})

    # graph = add_default_GraphAttributes_perNode(graph, node_attributes_transformed)

    DAG = add_default_GraphAttributes(
        DAG,
        {
            "graph_default_name": graph_default_name,
            "node_prefix": node_prefix,
        },
    )

    # add the default node name to the graph as attribute
    add_default_GraphAttributes_perNode(
        DAG, {"node_default_name": {n: n for n in list(DAG.nodes)}}
    )

    # relabel the nodes and terms
    DAG = rename_nodes_and_terms(
        dag=DAG, mapping={n: f"{node_prefix}{n}" for n in DAG.nodes}
    )

    return DAG
