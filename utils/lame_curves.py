import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point

from dataclasses import dataclass, replace
from typing import List, Tuple

from geometry_parameterizations.MMC import MMC

from skimage import measure
from rasterio.features import shapes

@dataclass
class LameCurveConfig:
    x: float
    y: float
    a: float
    b: float
    theta: float=0
    m: int = 6
    
    def __post_init__(self) : assert not(self.m%2), '`m` parameter of level set should be even.'

def construct_geometry(
    mmcs: List[MMC], 
    symmetry_y: bool=True, 
    no_elements_x: int=100,
    no_elements_y: int=50,
    n_samples: int=1_000
) -> MultiPolygon :
    lcs: List[LameCurveConfig] = [LameCurveConfig(mmc.pos_X, mmc.pos_Y, mmc.length, mmc.thickness, mmc.angle, m=6) for mmc in mmcs]
    if (symmetry_y) : 
        lcs += [replace(lc, y=no_elements_y-lc.y, theta=np.pi - lc.theta) for lc in lcs]

    t = np.linspace(0, 2*np.pi, n_samples)
    geo: MultiPolygon = MultiPolygon()
    for lc in lcs : 
        geo = geo.union(Polygon(compute_curve(lc, t)))

    domain: Polygon = Polygon([(0,0), (no_elements_x,0), (no_elements_x,no_elements_y),(0,no_elements_y)])
    geo = geo.intersection(domain)

    return geo if isinstance(geo, MultiPolygon) else MultiPolygon([geo]) 


# regular 2D rotation matrix
R2D = lambda phi : np.array([
    [np.cos(phi), -np.sin(phi)],
    [np.sin(phi),  np.cos(phi)]
])


def compute_curve(
    lc: LameCurveConfig,
    t: float, 
) -> Tuple[float, float] :
    X = np.c_[
        (lc.a/2) * np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/lc.m), 
        (lc.b/2) * np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/lc.m)
    ].T

    return (R2D(lc.theta) @ X).T + (lc.x, lc.y)


# def geo_from_binary_image(I: List[List[bool]]) -> MultiPolygon :
#     label_pos = measure.label(np.pad( I, 1), connectivity=1)
#     label_neg = measure.label(np.pad(~I, 1), connectivity=1)

#     geo: MultiPolygon = MultiPolygon()
#     for lab in range(1,label_pos.max()+1) : 
#         geo = geo.union(Polygon(
#             measure.find_contours(label_pos == lab)[0][:,[1,0]]
#         ))
#     for lab in range(1,label_neg.max()+1) : 
#         geo = geo.difference(Polygon(
#             measure.find_contours(label_neg == lab)[0][:,[1,0]]
#         ))

#     return geo if isinstance(geo, MultiPolygon) else MultiPolygon([geo]) 

def geo_from_binary_image(mask: List[List[bool]]) -> MultiPolygon :
    polygons = [
        (Polygon(poly['coordinates'][0]), bool(is_positive)) for (poly, is_positive) in 
        shapes(mask.astype(np.uint8))
    ]

    geo: MultiPolygon = MultiPolygon((poly for (poly, is_positive) in polygons if is_positive))
    for (poly, is_positive) in polygons:
        if not(is_positive) and geo.contains(poly) : geo = geo.difference(poly)

    return geo if isinstance(geo, MultiPolygon) else MultiPolygon([geo])