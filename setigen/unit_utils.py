"""
This module contains a couple unit conversion utilities used in frame.Frame.

In general, we rely on astropy units for conversions, and note that float
values are assumed to be in SI units (e.g. Hz, s).
"""
from __future__ import annotations

from astropy import units as u


def cast_value(value: float | u.Quantity, unit: u.UnitBase) -> u.Quantity:
    """Cast a scalar value into the requested astropy unit.

    Args:
        value: Numeric scalar or quantity to cast.
        unit: Destination unit.

    Returns:
        Value expressed as an astropy quantity in ``unit``.
    """
    if isinstance(value, u.Quantity):
        return value.to(unit)
    return value * unit


def get_value(value: float | u.Quantity,
              unit: u.UnitBase | None = None) -> float:
    """Extract a float from a scalar or astropy quantity.

    Args:
        value: Numeric scalar or quantity to unwrap.
        unit: Optional unit to convert into before extracting the numeric
            value.

    Returns:
        Floating-point value in the requested unit when provided.
    """
    if isinstance(value, u.Quantity):
        if unit is not None:
            return value.to(unit).value
        return value.value
    return value
