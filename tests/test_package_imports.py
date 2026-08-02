import unittest

from causalman import CausalMan, CausalManChoice
from causalman.utils.serialization import _current_module_name


class PackageImportTests(unittest.TestCase):
    def test_public_api_can_be_imported_and_constructed(self):
        simulator = CausalMan(name="causalman_micro")

        self.assertEqual(CausalManChoice.CAUSALMAN_MICRO.value, "causalman_micro")
        self.assertEqual(simulator.simulations, ["causalman_micro_1"])

    def test_legacy_pickle_modules_are_mapped_into_the_package(self):
        self.assertEqual(
            _current_module_name("line_structure.line_structure"),
            "causalman.line_structure.line_structure",
        )
        self.assertEqual(_current_module_name("node"), "causalman.node")


if __name__ == "__main__":
    unittest.main()
