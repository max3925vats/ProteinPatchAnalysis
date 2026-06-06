import numpy as np
from protein_patch.atom_types import atom_channel, ELEMENTS


def test_known_elements_one_hot():
    assert ELEMENTS == ("C", "N", "O", "S")
    np.testing.assert_array_equal(atom_channel("C"), [1, 0, 0, 0])
    np.testing.assert_array_equal(atom_channel("N"), [0, 1, 0, 0])
    np.testing.assert_array_equal(atom_channel("O"), [0, 0, 1, 0])
    np.testing.assert_array_equal(atom_channel("S"), [0, 0, 0, 1])


def test_unknown_element_is_all_zero():
    np.testing.assert_array_equal(atom_channel("H"), [0, 0, 0, 0])
    np.testing.assert_array_equal(atom_channel("FE"), [0, 0, 0, 0])
