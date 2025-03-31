from geometry_parameterizations.MMC import MMC

import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import minimize

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import List, Tuple, Union

from shapely.geometry import Polygon

@dataclass
class LameCurveConfig:
    x: float
    y: float
    a: float
    b: float
    theta: float=0
    m: int = 6
    
    def __post_init__(self) : assert not(self.m%2), '`m` parameter of level set should be even.'

@dataclass
class Point: 
    x: float
    y: float
    # Point's local coordinate geo orientation, only to make sure `guess_minimum_distance_points` works
    theta: float=0


def construct_lame_configs(mmcs: List[MMC], symmetry_y:bool=True, no_elements_y: int=50) -> List[LameCurveConfig] :
    lcs: List[LameCurveConfig] = [LameCurveConfig(mmc.pos_X, mmc.pos_Y, mmc.length, mmc.thickness, mmc.angle, m=6) for mmc in mmcs]
    if (symmetry_y) : 
        lcs += [replace(lc, y=no_elements_y-lc.y, theta=np.pi - lc.theta) for lc in lcs]
    return lcs

# regular 2D rotation matrix
R2D = lambda phi : np.array([
    [np.cos(phi), -np.sin(phi)],
    [np.sin(phi),  np.cos(phi)]
])

# get quadrant domain $ Q_k: {1,2,3,4} \to [0, 2\pi) $
Q_k = lambda k : ((k-1) * np.pi/2, k * np.pi/2)

# quadrant enumeration $ q: [0, 2\pi) \to {1,2,3,4} $
q = lambda phi : (2*phi)//np.pi % 4 + 1

GeometricObject = Union[LameCurveConfig, Point]


def compute_shape(
    lc: LameCurveConfig,
    x: float,
    y: float
) -> bool :
    return -((
        ((x - lc.x) * np.cos(lc.theta) + (y - lc.y)*np.sin(lc.theta)) / 
        (lc.a / 2)
    )**lc.m + (
        ((x - lc.x) *-np.sin(lc.theta) + (y - lc.y)*np.cos(lc.theta)) / 
        (lc.b / 2)
    )**lc.m - 1
    ) >= 0


def compute_curve(
    lc: LameCurveConfig,
    t: float, 
) -> Tuple[float, float] :
    X = np.c_[
        (lc.a/2) * np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/lc.m), 
        (lc.b/2) * np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/lc.m)
    ].T

    return (R2D(lc.theta) @ X).T + (lc.x, lc.y)


def compute_area(lcs: List[LameCurveConfig], xaxis_bounds: Tuple[float,float], yaxis_bounds: Tuple[float, float], n_samples: int=100):
    geo: Polygon = Polygon()
    dom: Polygon = Polygon([
        (xaxis_bounds[0],yaxis_bounds[0]),
        (xaxis_bounds[1],yaxis_bounds[0]),
        (xaxis_bounds[1],yaxis_bounds[1]),
        (xaxis_bounds[0],yaxis_bounds[1]),
    ])

    t = np.linspace(0, 2*np.pi, n_samples)
    for lc in lcs :
        (x, y) = compute_curve(lc, t).T
        geo = geo.union(Polygon(np.c_[x, y]))

    return geo.intersection(dom).area


def get_local_angle(
    geo1: GeometricObject, geo2: GeometricObject
) -> Tuple[float, float]:
    t_est = lambda geo_i, geo_j : (
        np.arctan2(geo_j.y - geo_i.y, geo_j.x - geo_i.x) - geo_i.theta
    ) % (2*np.pi)

    return (t_est(geo1, geo2), t_est(geo2, geo1))


def distance_to_point(
    lc: LameCurveConfig,
    t : float, 
    pt: Point
) -> float :
    return np.linalg.norm(compute_curve(lc, t) - (pt.x, pt.y), axis=1)


def solve_distance_to_point(lc: LameCurveConfig, pt: Point, n_samples: int=100) -> Tuple[float, float] :
    if compute_shape(lc, pt.x, pt.y) : return (float('nan'), 0)
    (t_guess, _) = get_local_angle(lc, pt)
    t_bounds = (t_lower, t_upper) = Q_k(q(t_guess))
    t = np.linspace(*t_bounds, n_samples)
    d = distance_to_point(lc, t, pt)
    return (t[d.argmin()], d.min()) if d.min() > 0 else (float('nan'), 0)


def distance_to_yaxis(
    lc: LameCurveConfig,
    t : float,
) -> float :
    return compute_curve(lc, t)[:,0]


def solve_distance_to_yaxis(lc: LameCurveConfig, n_samples: int=100) -> Tuple[float, float] :
    t_bounds = (t_lower, t_upper) = (
        {Q_k(1), Q_k(4)} if (np.cos(lc.theta) < 0) else {Q_k(2), Q_k(3)}
    ).intersection(
        {Q_k(1), Q_k(2)} if (np.sin(lc.theta) > 0) else {Q_k(3), Q_k(4)}
    ).pop()
    t = np.linspace(*t_bounds, n_samples)
    d = distance_to_yaxis(lc, t)
    return (t[d.argmin()], d.min()) if d.min() > 0 else (float('nan'), 0)


def solve_distance_to_bounded_yaxis(lc: LameCurveConfig, yaxis_bounds=(0,5), n_samples: int=100) -> Tuple[float, float] :
    # TODO : use some guarantees like if shape is far outside the bounds
    (t, d) = solve_distance_to_yaxis(lc)
    yaxis_pt = Point(x=0, y=compute_curve(lc, t)[0,1])
    if (yaxis_bounds[0] < yaxis_pt.y < yaxis_bounds[1]) :
        return (t, d)
    return solve_distance_to_point(lc, Point(0, yaxis_bounds[0] if yaxis_pt.y < yaxis_bounds[0] else yaxis_bounds[1]))


def distance_between_curves(
    lc1: LameCurveConfig, 
    t1 : float, 
    lc2: LameCurveConfig,
    t2 : float
) -> float :
    return np.hypot(
        *(compute_curve(lc1, t1) - compute_curve(lc2, t2)).T
    )


def bounds_quadrants(t) :
    center = np.pi/2*(2*(t + np.pi/4)//np.pi)
    return np.c_[center-np.pi/2, center+np.pi/2]


def bounds_simple(t) : return np.c_[t - np.pi/2, t + np.pi/2]


def solve_distance_between_curves(
    lc1: LameCurveConfig,
    lc2: LameCurveConfig,
    bounds_method: Callable=bounds_quadrants,
    n_samples: int=100
) -> Tuple[float, float, float] :
    t = np.linspace(0, 2*np.pi, n_samples)
    if compute_shape(lc2, *compute_curve(lc1, t).T).any() : return (float('nan'), float('nan'), 0)

    (t_est, s_est) = get_local_angle(lc1, lc2)

    bounds = (
        bounds_method(t_est)[0], 
        bounds_method(s_est)[0]
    )

    result = minimize(
        fun=lambda x : distance_between_curves(lc1, x[0], lc2, x[1]),
        x0=(t_est, s_est),
        bounds=bounds,
        method='L-BFGS-B',
    )
    # TODO : check convergence status
    return (*result['x'], result['fun'])


def calculate_disconnection(lcs: List[LameCurveConfig], pts: List[Point], yaxis_bounds: Tuple[float, float]) -> float :
    n = len(lcs)
    D = np.zeros((n,n))
    for i in range(n) :
        for j in range(i) :
            (_, _, d) = solve_distance_between_curves(lcs[i], lcs[j])
            # making sure the MST understands there is an edge, for 0 it thinks it's not connected,
            # also if the values get too low (<1e-8) this can happen
            D[i,j] = D[j,i] = d if (d > 1e-6) else -1

    D:np.ndarray = np.maximum( # using the max to filter out the -1
        minimum_spanning_tree(D).toarray(), 0
    )
    D_MST:np.ndarray = minimum_spanning_tree(D).toarray()
    D_MST[D_MST < 1/2] = 0
    d_MST = D_MST.sum()

    if (d_yaxis := min(solve_distance_to_bounded_yaxis(lc, yaxis_bounds)[1] for lc in lcs)) > 1/2 : 
        d_MST += d_yaxis
    for pt in pts : 
        if (d_pt := min(solve_distance_to_point(lc, pt)[1] for lc in lcs)) > 1/2 : 
            d_MST += d_pt

    return d_MST