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
import numpy as np

import os

# R_HOME can be set via the R_HOME environment variable before importing this module.
# Example: export R_HOME=/usr/lib/R  or  set R_HOME=C:/Program Files/R/R-4.4.2
_r_home = os.environ.get("R_HOME")
if _r_home:
    os.environ["R_HOME"] = _r_home

from rpy2 import robjects
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

# activate automatic conversion between numpy <-> R
#numpy2ri.activate()

#from rpy2.robjects.vectors import StrVector
#packnames = ('dagitty',)
#utils.install_packages(StrVector(packnames))
# Import igraph with suppressed conflicts
#import rpy2.robjects.lib.ggplot2
#from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

# Alternative: use robjects.r directly for igraph functions
#robjects.r('library(igraph)')


dagitty = importr("dagitty")

def nx_to_dagitty_string(nx_graph, latent_nodes):
    """
    Convert networkx DAG and latent set into DAGitty-format string.
    Mark latent nodes using [latent].
    """
    parts = []
    for u, v in nx_graph.edges():
        parts.append(f"{u} -> {v}")
    for l in latent_nodes:
        parts.append(f"{l} [latent]")
    return "dag { " + "; ".join(parts) + " }"

def marginalize_to_mag(nx_graph, latent_nodes):
    """
    Marginalize latent nodes in DAG via dagitty::toMAG → return MAG with edge types.
    """
    dag_str = nx_to_dagitty_string(nx_graph, latent_nodes)
    robjects.globalenv["g"] = dagitty.dagitty(dag_str)
    # Convert to MAG
    mag_obj = dagitty.toMAG(robjects.globalenv["g"])
    # Extract edges with types
    edf = robjects.r("edges")(mag_obj)
    # DataFrame columns: v (from), w (to), e (edge type: '->', '<->', '--')
    nx_mag = nx.MultiDiGraph()
    for row in zip(edf.rx2('v'), edf.rx2('w'), edf.rx2('e')):
        u, v, etype = row
        if etype == "->":
            nx_mag.add_edge(u, v, type="directed")
        elif etype == "<->":
            nx_mag.add_edge(u, v, type="bidirected")
            nx_mag.add_edge(v, u, type="bidirected")
        elif etype == "--":
            nx_mag.add_edge(u, v, type="undirected")
            nx_mag.add_edge(v, u, type="undirected")
    return nx_mag

# example usage
if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edges_from([
        ("X1", "X2"),
        ("X2", "X3"),
        ("L", "X2"),
        ("L", "X3")
    ])

    mag = marginalize_to_mag(G, latent_nodes=["L"])
    print("MAG edges:", mag.edges())
