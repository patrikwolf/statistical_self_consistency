from dataclasses import dataclass
from data_loader_wvs.wvs_node_values import WvsNodeValues


@dataclass()
class WvsDecompositionSplit:
    """Canonical WVS decomposition split representation (binary or more children)."""

    identifier: str
    description: str
    value_splits: list[WvsNodeValues]
