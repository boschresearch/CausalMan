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
from pyvis.network import Network
import pickle
import os
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

scale_x = 0.5
scale_y = 3
rotate_angle = 0

save_folder = "figures"
graphs_folder = "graphs"

graph_dicts = [{"filename": "ground_truth_partially_observable_dataset_0.pkl"}]

def generate_png(url_file, name):
    
    with sync_playwright() as p:
        for browser_type in [p.chromium]:
            browser = browser_type.launch()
            page = browser.new_page()
            file = open(url_file, "r").read()
            page.set_content(file, wait_until="load")
            page.wait_for_timeout(30000) # this timeout for correctly render big data page
            page.screenshot(path=f'{name}.png', full_page=True)
            browser.close()
            #file.close()



def rotate_position(pos: dict, angle: float) -> dict:
    """Given a dictionary of positions, rotate them by a given angle.

    Args:
        pos (dict): Dictionary of positions.
        angle (float): Angle in degrees.

    Returns:
        dict: Dictionary of rotated positions.
    """
    new_pos = {}
    for node in pos:
        x, y = pos[node]
        x_new = x * np.cos(np.radians(angle)) - y * np.sin(np.radians(angle))
        y_new = x * np.sin(np.radians(angle)) + y * np.cos(np.radians(angle))
        new_pos[node] = (x_new, y_new)
    return new_pos


def plot_graph(
    G: nx.Graph,
    scale_x: float = 1,
    scale_y: float = 1,
    rotate_angle: float = 0,
    partially_observable: bool = False,
):
    
    observable_nodes_list = []
    for node in G.nodes(data=True):
        if "Observable" in node[1] and node[1]["Observable"] is True:
            observable_nodes_list.append(node[0])
        if partially_observable:
            observable_nodes_list.append(node[0])

        # Store nodes and edges
    nodes_list = list(G.nodes(data=True))
    edges_list = list(G.edges(data=True))
    nodes_dict = dict(G.nodes(data=True))

    # Generate layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # Rotate
    if rotate_angle != 0:
        pos = rotate_position(pos, rotate_angle)
        # Swap x and y scaling factors
        scale_x, scale_y = scale_y, scale_x

    # Export nx graph to pyvis for visualization purposes
    pyvis_graph = nx.empty_graph(0, create_using=nx.DiGraph)
    initials = "" #"PF_M1_T1_Force"
    for node in nodes_list:
        if node[0].startswith(initials) or node[0].startswith("Sec_C2_Machine1_ProcessResult"):
            color = "#ff5d00" if node[0] in observable_nodes_list else "#1DA1F2"
            pyvis_graph.add_node(node[0], color = color, value = 400, borderWidth = 3, borderWidthSelected = 5) #, label = [""])

    for n1, n2, data in edges_list:
        if n1.startswith(initials) and n2.startswith("Sec_C2_Machine1_ProcessResult"):
            pyvis_graph.add_edge(n1, n2)
        elif n1.startswith("Sec_C2_Machine1_ProcessResult") and n2.startswith(initials):
            pyvis_graph.add_edge(n1, n2)
        elif n1.startswith(initials) and n2.startswith(initials):
            pyvis_graph.add_edge(n1, n2)

        # Initialize visualization
    nt = Network("2400px", "5000px", directed=True)
    nt.show_buttons()
    nt.from_nx(pyvis_graph)

    for node in nt.get_nodes():
        nt.get_node(node)["x"] = scale_x * pos[node][0]
        nt.get_node(node)["y"] = scale_y * pos[node][1]
        nt.get_node(node)["physics"] = False
        #nt.get_node(node)["label"] = str(node)
    return nt


if __name__ == "__main__":
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for key in graph_dicts:
        graph_file: os.PathLike[str] = key["filename"]
        partially_observable: bool = True if "partially_observable" in graph_file else False

        # Load file and print basic information
        graph_path = os.path.join(os.getcwd(), graphs_folder, graph_file)

        if graph_file.endswith(".gml"):
            G = nx.read_graphml(graph_path)
        elif graph_file.endswith(".pkl"):
            with open(graph_path, "rb") as f:
                G = pickle.load(f)
        elif graph_file.endswith(".csv"):
            G = nx.from_edgelist(pd.read_csv(graph_path, index_col  = 0).values)
        else:
            err = f"File {graph_file} not supported. Has to be in .gml or .pkl format."
            raise ValueError(err)

        num_nodes = len(G.nodes)
        num_edges = len(G.edges)
        print(f"Graph in {graph_file} has {num_nodes} and {num_edges} edges.")

        nt = plot_graph(G, scale_x, scale_y, rotate_angle, partially_observable)

        # Remove extension from filename
        filename = f"{os.path.splitext(graph_file)[0]}.html"

        path = os.path.join(save_folder, filename)
        nt.show(path, local = True, notebook=False)
        generate_png(path, "filename")
