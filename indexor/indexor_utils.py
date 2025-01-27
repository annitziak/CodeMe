import os
import logging

from indexor.in_memory_index import InMemoryIndex
from indexor.on_disk_index import OnDiskIndex

logger = logging.getLogger(__name__)


def build_index(load_path):
    if not os.path.exists(load_path):
        raise FileNotFoundError("The file does not exist")

    if os.path.isfile(load_path):
        logger.info("Loading index from memory")
        return InMemoryIndex(load_path)

    if os.path.isdir(load_path):
        logger.info("Loading index from disk")
        return OnDiskIndex(load_path)

    raise ValueError("Invalid index path")
