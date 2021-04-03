import numpy


def single_geoms(geoms):
    assert all(len(g) == 1 for g in geoms)


def same_crs(crs1, crs2):
    assert crs1 == crs2


def at_least_empty_list(value):
    if isinstance(value, numpy.ndarray):
        value = value.tolist()
    elif numpy.isscalar(value) and value != "":
        value = [value]
    elif not value:
        value = []

    return value
