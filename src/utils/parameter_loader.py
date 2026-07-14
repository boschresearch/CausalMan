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

import json
import os
from typing import Dict, Optional, Tuple

from utils.equation import (
    create_distrib_fixed,
    create_distrib_normal,
    create_distrib_uniform,
)


###
def init_distributions(dictionary: Dict):
    """
    Initialize the distributions for the source nodes in the dictionary
    """
    for key in dictionary.keys():
        if isinstance(dictionary[key], Dict):
            dictionary[key] = init_distributions(dictionary[key])
        if isinstance(dictionary[key], list):
            if dictionary[key][1] in ["normal", "fixed", "uniform"]:
                dictionary[key] = sample_distribution(dictionary[key])

    return dictionary


def sample_distribution(pars_list: list):
    """
    Initialize the distribution for the source nodes in the list
    """
    var_name = pars_list[0]
    dist_name = pars_list[1]
    dist_pars = pars_list[2]
    if dist_name == "normal":
        distrib = create_distrib_normal(var_name, dist_pars[0], dist_pars[1])
    elif dist_name == "uniform":
        distrib = create_distrib_uniform(var_name, dist_pars[0], dist_pars[1])
    elif dist_name == "fixed":
        distrib = create_distrib_fixed(var_name, dist_pars)
    else:
        raise ValueError(f"Unknown distribution type: {dist_name}")
    return distrib


def load_dict_from_json(filepath: str) -> dict:
    """Load a dictionary from a json file

    Args:
        filepath (os.PathLike): Path to the json file

    Raises:
        FileNotFoundError: If the file does not exist

    Returns:
        dict: The dictionary loaded from the json file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist")
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
    return data


def load_EOL_TrueQuality(path: str) -> Dict:
    filepath: str = os.path.join(
        path, "EOL_Monitoring_TRUE_default_sourceDistributions.json"
    )
    return load_dict_from_json(filepath)


# Used in MVgen.py
def load_MVgen(storage_dir: os.PathLike) -> Dict:
    path: str = os.path.join(storage_dir, "MVgen_distributions.json")
    return init_distributions(load_dict_from_json(path))


# Used in HUgen.py
def load_HU(storage_dir: os.PathLike) -> Dict:
    path: str = os.path.join(storage_dir, "HUgen", "HUgen_distributions.json")
    return init_distributions(load_dict_from_json(path))

def load_HU_Chambers(storage_dir: os.PathLike):
    path: str = os.path.join(
        storage_dir, "HUgen", "HU_Chambers.json"
    )
    return load_dict_from_json(path)

def load_HU_BorePositions(storage_dir: os.PathLike):
    path: str = os.path.join(
        storage_dir, "HUgen", "HU_BorePositions.json"
    )
    return load_dict_from_json(path)

def load_HUmeas(storage_dir: os.PathLike, section_ID: int, block_ID: int):
    section_ID_str = "section_" + str(section_ID)
    if block_ID is None:
        path: str = os.path.join(storage_dir, "HUmeas", section_ID_str)
    else:
        block_ID_str = str(block_ID)
        path: str = os.path.join(
            storage_dir, "HUmeas", section_ID_str, block_ID_str
        )

    path: str = os.path.join(
        path, "EOL_Monitoring_default_sourceDistributions.json"
    )
    return init_distributions(load_dict_from_json(path))


def load_cycle_time_HU_boreMeas(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)

    
    path: str = os.path.join(
        storage_dir, "HU_boreMeas", section_ID_str, "cycle_time.json"
    )
    return load_dict_from_json(path)


def load_HU_boreMeas(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)
    path: str = os.path.join(
        storage_dir,
        "HU_boreMeas",
        section_ID_str,
        "HU_boreMeas_default_sourceDistributions.json",
    )
    return init_distributions(load_dict_from_json(path))


### EOL MONITORING ###
def load_EOL_Monitoring(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)
    path: str = os.path.join(
        storage_dir, section_ID_str, "EOL_Monitoring_default_sourceDistributions.json"
    )
    return init_distributions(load_dict_from_json(path))


### MVmeas_Aleak ###


def load_MVmeas_EOL_Monitoring(
    storage_dir: os.PathLike, section_ID: int, block_ID: str
):
    section_ID_str = "section_" + str(section_ID)
    path: str = os.path.join(
        storage_dir,
        "MVmeas",
        section_ID_str,
        f"{block_ID}_EOL_Monitoring_default_sourceDistributions.json",
    )
    return init_distributions(load_dict_from_json(path))


def load_cycle_time_MVmeas(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)

    path: str = os.path.join(
        storage_dir, "MVmeas", section_ID_str, "cycle_time.json"
    )
    return load_dict_from_json(path)


def load_MVmeas(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)
    path: str = os.path.join(
        storage_dir,
        "MVmeas",
        section_ID_str,
        "MV_meas_default_sourceDistributions.json",
    )
    return init_distributions(load_dict_from_json(path))


### MVmeas_Aleak ###


def load_MVmeas_Aleak_EOL_Monitoring(storage_dir: os.PathLike):

    path: str = os.path.join(
        storage_dir,
        "MVmeas_Aleak",
        "EOL_Monitoring_default_sourceDistributions.json",
    )
    return init_distributions(load_dict_from_json(path))


def load_cycle_time_MVmeas_Aleak(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)

    path: str = os.path.join(
        storage_dir, "MVmeas_Aleak", section_ID_str, "cycle_time_delta_t.json"
    )
    return load_dict_from_json(path)


def load_MVmeas_Aleak(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)
    path: str = os.path.join(
        storage_dir,
        "MVmeas_Aleak",
        section_ID_str,
        "MV_Meas_default_sourceDistributions.json",
    )
    return init_distributions(load_dict_from_json(path))


### PressFitting ###


def load_PF_EOL_Monitoring(storage_dir: os.PathLike, section_ID: int, product_type_ID: str):
    section_ID_str = "section_" + str(section_ID)

    block_ID_str = str(product_type_ID)
    filename = block_ID_str + "_EOL_Monitoring_default_sourceDistributions.json"
    path: str = os.path.join(storage_dir, "PF", section_ID_str, filename)

    return init_distributions(load_dict_from_json(path))


def load_PressFitting(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)
    path: str = os.path.join(
        storage_dir,
        "PF",
        section_ID_str,
        "PressFitting_default_sourceDistributions.json",
    )
    return init_distributions(load_dict_from_json(path))


def load_PF_cycle_time(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)
    path: str = os.path.join(
        storage_dir,
        "PF",
        section_ID_str,
        "cycle_time_delta_t.json",
    )
    return load_dict_from_json(path)


### EOL MONITORING


def load_EOL_Monitoring(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)
    path: str = os.path.join(
        storage_dir,
        "EOL_monitoring",
        section_ID_str,
        "EOL_Monitoring_default_sourceDistributions.json",
    )
    return init_distributions(load_dict_from_json(path))


def load_EOL(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)
    path: str = os.path.join(
        storage_dir,
        "EOL_monitoring",
        section_ID_str,
        "EOL_default_sourceDistributions.json",
    )
    return init_distributions(load_dict_from_json(path))


def load_EOL_cycle_time(storage_dir: os.PathLike, section_ID: int):
    section_ID_str = "section_" + str(section_ID)
    path: str = os.path.join(
        storage_dir,
        "EOL_monitoring",
        section_ID_str,
        "cycle_time_delta_t.json",
    )
    return load_dict_from_json(path)


### TRUE QUALITY


def load_EOL_Monitoring_True(storage_dir: os.PathLike):
    path: str = os.path.join(
        storage_dir, "EOL_Monitoring_TRUE_default_sourceDistributions.json"
    )
    return init_distributions(load_dict_from_json(path))


def load_edge_list(storage_dir: os.PathLike):
    """
    Load an edge list from a JSON file.

    Args:
        storage_dir (os.PathLike): The directory where the JSON file is stored.

    Returns:
        dict: The loaded edge list as a dictionary.
    """
    path: str = os.path.join(storage_dir, "edge_list.json")
    return load_dict_from_json(path)


### Utils for parsing parameters:


def parse_cycle_time(
    cycle_time_dict: Dict, machine_name_short: Optional[str] = None
) -> Tuple[float, float, float, float]:
    """Given the cycle_time_dict and the machine_name_short, return the cycle
    time parameters.

    Args:
        cycle_time_dict (Dict): Dictionary containing the cycle time parameters
        machine_name_short (str): String containing the name of the machine

    Returns:
        Tuple[float, float, float, float]: Tuple containing the cycle time
        parameters.
    """
    if machine_name_short is None:
        cycle_time_delta_t_fromPre_mean = cycle_time_dict[
            "cycle_time_delta_t_fromPre_mean"
        ]
        cycle_time_delta_t_fromPre_std = cycle_time_dict[
            "cycle_time_delta_t_fromPre_std"
        ]
        cycle_time_delta_t_step_mean = cycle_time_dict["cycle_time_delta_t_step_mean"]
        cycle_time_delta_t_step_std = cycle_time_dict["cycle_time_delta_t_step_std"]

    else:
        cycle_time_delta_t_fromPre_mean = cycle_time_dict[machine_name_short][
            "cycle_time_delta_t_fromPre_mean"
        ]
        cycle_time_delta_t_fromPre_std = cycle_time_dict[machine_name_short][
            "cycle_time_delta_t_fromPre_std"
        ]
        cycle_time_delta_t_step_mean = cycle_time_dict[machine_name_short][
            "cycle_time_delta_t_step_mean"
        ]
        cycle_time_delta_t_step_std = cycle_time_dict[machine_name_short][
            "cycle_time_delta_t_step_std"
        ]

    return (
        cycle_time_delta_t_fromPre_mean,
        cycle_time_delta_t_fromPre_std,
        cycle_time_delta_t_step_mean,
        cycle_time_delta_t_step_std,
    )
