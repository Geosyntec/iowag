def single_geoms(geoms):
    assert all(len(g) == 1 for g in geoms)


def same_crs(crs1, crs2):
    assert crs1 == crs2
