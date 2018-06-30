import numpy as np

def normalize(data, exclude=0.1):
    """
    Normalize data per frequency channel so that the noise level in data is
    controlled. Excludes a fraction of brightest pixels to better isolate noise.
    """
    # TODO: Implement exclusion, test that it removes artifacts from bright signals
