#!/usr/bin/env python3
"""Evict a file's pages from the page cache (posix_fadvise DONTNEED)."""
import os
import sys

for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    print(f"evicted {path}")
