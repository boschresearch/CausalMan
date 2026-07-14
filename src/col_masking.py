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
import pickle as pk
import argparse
from typing import Optional
import networkx as nx
import pandas as pd


def mask_columns(nodes_to_keep: Optional[list], df: pd.DataFrame) -> pd.DataFrame:
    """
    Masks the dataframe based on a list of variables.

    Args:
        nodes_to_keep (list): List of variables to keep
        df (pd.DataFrame): Dataframe to mask
    
    Returns:
        pd.DataFrame: Masked dataframe
    """
    
    cols_to_drop = [node for node in list(df.columns) if node not in nodes_to_keep]
    masked_dataframe = df.drop(columns=cols_to_drop)

    return masked_dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask the dataframe based on the observable nodes in the graph.")
    parser.add_argument(
        "--graph", help="Path to the graph file (pickle format)", required=True
    )
    parser.add_argument("--csv", help="Path to the csv file.", required=True)
    parser.add_argument(
        "--output_dir", help="Path to the output directory", required=True
    )
    args = parser.parse_args()

    # Open pickle file and read
    with open(args.graph, "rb") as f:
        G: nx.DiGraph = pk.load(f)

    df = pd.read_csv(args.csv)
    masked_dataframe = mask_columns(G, df)

    # Construct the output filename
    input_filename = os.path.basename(args.csv)
    filename_without_extension = os.path.splitext(input_filename)[0]
    output_filename = f"{filename_without_extension}_masked.csv"
    output_path = os.path.join(args.output_dir, output_filename)

    masked_dataframe.to_csv(output_path, index=False)
    print(f"Process Completed Successfully. File saved to {args.output_dir}")
