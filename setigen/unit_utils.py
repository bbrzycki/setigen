from astropy import units as u


def cast_value(value, unit):
    if type(value) == u.Quantity:
        return value.to(unit)
    return value * unit


def get_value(value, unit=None):
    if type(value) == u.Quantity:
        if unit is not None:
            return value.to(unit).value
        else:
            return value.value
    return value
