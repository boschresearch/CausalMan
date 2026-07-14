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

from line_structure.baseline import BaselineStructure
from networkx import DiGraph


class Machine(BaselineStructure):
    """
    Class to create a single Machine. Most of the baseline class
    "baseline_structure" can be inherited.

    The next substructure level are the "Steps", which are of
    type "Step".

    """

    def add_Step(self, step, previous=None, enforce=False):
        self.add_Structure(step, previous=previous, typeList=["Step"], enforce=enforce)

    @property
    def Steps(self):
        return self.get_structures()

    def add_merged_timeDag(self, timeDAG: DiGraph):
        self.time_dag_merged = timeDAG
