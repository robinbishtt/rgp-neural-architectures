from __future__ import annotations
from typing import Dict
MASTER_HASHES: Dict[str, str] = {
    :     None,
    :    None,
    :    None,
    :    None,
    :    None,
    :     None,
    :   None,
    :    None,
    :        None,
    :         None,
}
def get_expected_hash(dataset_name: str) -> str | None:
    return MASTER_HASHES.get(dataset_name)
def is_registered(dataset_name: str) -> bool:
    return dataset_name in MASTER_HASHES