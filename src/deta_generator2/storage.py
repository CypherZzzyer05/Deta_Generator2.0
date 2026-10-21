"""Bounded Windows output retries without changing physical-source fingerprints."""
from contextlib import contextmanager
import functools
import logging
import sys
import threading
import time

from . import pipeline

LOG = logging.getLogger(__name__)
_write_context_lock = threading.RLock()


def retry_io(writer, attempts=101, delay=.1):
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1 or delay < 0:
        raise ValueError('Retry attempts must be positive and delay nonnegative')
    if getattr(writer, '_deta_output_retry', False) is True:
        return writer
    @functools.wraps(writer)
    def save(*args, **kwargs):
        for attempt in range(attempts):
            try:
                return writer(*args, **kwargs)
            except PermissionError as error:
                if (sys.platform != 'win32' or getattr(error, 'winerror', None) not in (5, 32, 33)
                        or attempt + 1 == attempts):
                    raise
                if attempt == 0:
                    LOG.warning('Windows denied an output write/replacement; retrying a bounded number of times')
                time.sleep(delay)
    save._deta_output_retry = True
    return save


atomic_json = retry_io(pipeline.atomic_json)


@contextmanager
def pipeline_write_retries():
    # Merge and packing use functions imported by the unchanged pipeline module.
    # Serialize this temporary binding in one process, including nested callers.
    with _write_context_lock:
        original_json, original_npz = pipeline.atomic_json, pipeline.atomic_npz
        pipeline.atomic_json = retry_io(original_json)
        pipeline.atomic_npz = retry_io(original_npz)
        try:
            yield
        finally:
            pipeline.atomic_json, pipeline.atomic_npz = original_json, original_npz


def resilient_writes(function):
    @functools.wraps(function)
    def call(*args, **kwargs):
        with pipeline_write_retries():
            return function(*args, **kwargs)
    return call
