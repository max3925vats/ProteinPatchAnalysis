"""Data acquisition and dataset preparation.

Note: `prep` and `fetch` are intentionally NOT re-exported here. `prep`
imports from `protein_patch.patches`, which imports `data.attributes`, so
re-exporting them would create a circular import. Import them by full path:
`from protein_patch.data.prep import prep_dataset`.
"""
from .attributes import KYTE_DOOLITTLE, ResidueAttributes, residue_attributes

__all__ = ["KYTE_DOOLITTLE", "ResidueAttributes", "residue_attributes"]
