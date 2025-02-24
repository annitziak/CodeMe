import os
import logging

from indexor.in_memory_index import InMemoryIndex
from indexor.on_disk_index import OnDiskIndex

logger = logging.getLogger(__name__)


def build_index(load_path, use_disk_index=True, **kwargs):
    if not os.path.exists(load_path):
        raise FileNotFoundError("The file does not exist")

    if os.path.isdir(load_path) and use_disk_index:
        logger.info("Loading index from disk")
        return OnDiskIndex(load_path)

    if os.path.isfile(load_path) and not use_disk_index:
        logger.info("Loading index from memory")
        return InMemoryIndex(load_path)

    raise ValueError(
        f"Invalid index path {load_path}  or use_disk_index {use_disk_index} value"
    )
