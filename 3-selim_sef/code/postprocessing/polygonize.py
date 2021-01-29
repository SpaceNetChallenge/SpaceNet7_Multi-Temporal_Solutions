from typing import Dict

import numpy as np
import rasterio
import rasterio.features
import shapely
import shapely.affinity
import shapely.geometry
import shapely.ops
import shapely.wkt
from shapely.geometry import MultiPolygon


def mask_to_poly(mask: np.ndarray) -> Dict:
    """Generates geojson polygon from a single instance binary mask
    Parameters
    ----------
    mask : binary masks representing single object

    Returns
    -------
    geo : dict
        Geo polygon as dict
    """
    mp = mask_to_shapely_polygon(mask)
    geo = shapely.geometry.mapping(mp)

    return geo


def mask_to_shapely_polygon(mask: np.ndarray) -> MultiPolygon:
    """Generates shapely polygon from a single instance binary mask
    Parameters
    ----------
    mask : binary masks representing single object

    Returns
    -------

    mp : shapely.geometry.MultiPolygon
        Geo polygon as shapely object
    """

    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))

    return mp
