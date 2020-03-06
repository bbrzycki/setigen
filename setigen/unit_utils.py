"""
This module contains a couple unit conversion utilities used in frame.Frame.

In general, we rely on astropy units for conversions, and note that float
values are assumed to be in SI units (e.g. Hz, s).
"""
from astropy import units as u


def cast_value(value, unit):
    """
    If value is already an astropy Quantity, then cast it into the desired
    unit. Otherwise, value is assumed to be a float and converted directly to
    the desired unit.
    """
    if isinstance(value, u.Quantity):
        return value.to(unit)
    return value * unit


def get_value(value, unit=None):
    """
    This function converts a value, which may be a float or astropy Quantity,
    into a float (in terms of a desired unit).

    If we know that value is an astropy Quantity, then grabbing the value
    is simple (and we can cast this to a desired unit, if we need to change
    this.

    If value is already a float, it simply returns value.
    """
    if isinstance(value, u.Quantity):
        if unit is not None:
            return value.to(unit).value
        else:
            return value.value
    return value
