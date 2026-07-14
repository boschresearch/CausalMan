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

class Section(BaselineStructure):
    """
    Class to create a single Section. Most of the baseline class
    "baseline_structure" can be inherited.

    The next substructure level are the "Machines",
    which are of type "Machine".

    """

    def add_Machine(self, machine, previous=None, enforce=False):
        self.add_Structure(
            machine, previous=previous, typeList=["Machine"], enforce=enforce
        )

    @property
    def Machines(self):
        return self.get_structures()

