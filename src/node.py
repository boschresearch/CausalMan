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
from sympy import symbols, Eq
import networkx as nx
import pandas as pd


class InputNode(str):
    pass


class OutputNode(str):
    pass

class NodeModel:
    def __init__(
        self,
        node: str,
        parents: list,
        term,
        term_str: Optional[str] = None,
        distribution=None,
        distribution_default=None,
        distribution_parents=None,
        config_attr={
            "term": "term",
            "source_distribution": "source_distribution",
            "source_distribution_default": "source_distribution_default",
            "source_distribution_parents": "source_distribution_parents",
            "noise_distribution": "noise_distribution",
            "NodeModel": "NodeModel",
        },
    ):
        self.name = node
        self.parents = parents
        self.term = term
        self.term_string = term_str
        # FIXME: Add methods to manage the attribute names config attribute
        self.config_attr = config_attr
        self.distribution = distribution
        self.distribution_default = distribution_default
        if distribution_parents is not None:
            self.distribution_parents = {
                p: dist for p, dist in distribution_parents.items() if p in self.parents
            }
        else:
            self.distribution_parents = {}

    def predict(self, df: pd.DataFrame):
        pass

    def add_to_DAG(self, DAG: nx.DiGraph, addIndependentNodes: bool = True):
        DAG.add_edges_from([(p, self.name) for p in self.parents])
        if (len(self.parents) == 0) and addIndependentNodes:
            DAG.add_node(self.name)

        # FIXME: Check if the "term" attribute has already been used in DAG?
        nx.set_node_attributes(
            DAG, {self.name: self.term}, name=self.config_attr["term"]
        )
        if self.distribution is not None:
            nx.set_node_attributes(
                DAG,
                {self.name: self.distribution},
                name=self.config_attr["source_distribution"],
            )
        if len(self.distribution_parents) > 0:
            nx.set_node_attributes(
                DAG,
                {
                    p: self.distribution_parents.get(p, None)
                    for p in self.parents
                    if p in self.distribution_parents.keys()
                },
                name=self.config_attr["source_distribution"],
            )
        if self.distribution_default is not None:
            nx.set_node_attributes(
                DAG,
                {self.name: self.distribution_default},
                name=self.config_attr["source_distribution_default"],
            )
        """
        nx.set_node_attributes(
            DAG,
            {p: self.distribution_parents.get(p, None)
            for p in self.parents},
            name=self.config_attr["source_distribution"],
        )
        """
        nx.set_node_attributes(
            DAG, {self.name: self}, name=self.config_attr["NodeModel"]
        )

        return DAG

    def rename(self, mapping: dict):
        self.rename_nodes(mapping)
        self.rename_terms(mapping)
        self.rename_distributions(mapping)
        self.rename_distributions_default(mapping)

    def rename_terms(self, mapping: dict):
        pass

    def rename_distributions(self, mapping: dict):
        pass

    def rename_distributions_default(self, mapping: dict):
        pass

    def rename_nodes(self, mapping: dict):
        self.name = mapping.get(self.name, self.name)
        self.parents = [mapping.get(p, p) for p in self.parents]


class NodeModel_sympy(NodeModel):
    def __init__(
        self,
        symPyEquation,
        distribution=None,
        distribution_default=None,
        distribution_parents=None,
        config_attr={
            "term": "term",
            "source_distribution": "source_distribution",
            "source_distribution_default": "source_distribution_default",
            "source_distribution_parents": "source_distribution_parents",
            "noise_distribution": "noise_distribution",
            "NodeModel": "NodeModel",
        },
    ):
        self.name = str(symPyEquation.lhs)
        self.parents = [
            str(symb)
            for symb in symPyEquation.free_symbols
            if str(symb) is not self.name
        ]
        self.term = symPyEquation
        self.term_string = str(symPyEquation.rhs)
        # FIXME: Add methods to manage the attribute names config attribute
        self.config_attr = config_attr
        self.distribution = distribution
        self.distribution_default = distribution_default
        if distribution_parents is not None:
            self.distribution_parents = {
                p: dist for p, dist in distribution_parents.items() if p in self.parents
            }
        else:
            self.distribution_parents = {}

    def predict(self, df: pd.DataFrame):
        pass

    def addDistribution(self, distribution):
        pass

    def rename_terms(self, mapping: dict) -> None:
        term_lhs = self.term.lhs
        term_rhs = self.term.rhs

        for node in self.term.free_symbols:
            if node in mapping.keys():
                term_lhs = term_lhs.subs(
                    {symbols(node): symbols(mapping.get(node, node))}
                )
                term_rhs = term_rhs.subs(
                    {symbols(node): symbols(mapping.get(node, node))}
                )

        self.term = Eq(term_lhs, term_rhs)
        return

    def rename_distributions(self, mapping: dict) -> None:
        term_lhs = self.distribution.lhs
        term_rhs = self.distribution.rhs

        for node in self.distribution.free_symbols:
            if node in mapping.keys():
                term_lhs = term_lhs.subs(
                    {symbols(node): symbols(mapping.get(node, node))}
                )
                term_rhs = term_rhs.subs(
                    {symbols(node): symbols(mapping.get(node, node))}
                )

        self.distribution = Eq(term_lhs, term_rhs)
        return

    def rename_distributions_default(self, mapping: dict) -> None:
        term_lhs = self.distribution_default.lhs
        term_rhs = self.distribution_default.rhs

        for node in self.distribution_default.free_symbols:
            if node in mapping.keys():
                term_lhs = term_lhs.subs(
                    {symbols(node): symbols(mapping.get(node, node))}
                )
                term_rhs = term_rhs.subs(
                    {symbols(node): symbols(mapping.get(node, node))}
                )

        self.distribution_default = Eq(term_lhs, term_rhs)
        return