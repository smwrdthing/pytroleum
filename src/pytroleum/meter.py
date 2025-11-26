import numpy as np
from typing import Callable, Iterable
# Apply vectorization with decorators. Either enforce input arguments type conversion
# or straight up use np.vectorize; second approach destroysdocstrings

# TODO :
#  1. docstrings
#  2. functions for U-shaped tube
#  3. some more convenience functions
#  4. functions for other geometries (long-term goal)

# !DISCLAIMER! : All functions work with SI units!


# -----------------------------------------------------------------------------
# CROSS-SECTIONAL AREA COMPUTATIONS
# -----------------------------------------------------------------------------


def area_cs_circle_trunc(D: float, h: float) -> float:
    """This function computes area of circle which is truncated with horizontal
    straight line. Formula obtained by the integration of the following function
    from zero to h:

                        2*sqrt((D/2)**2 - (x-D/2)**2)

    Parameters
    ----------
    D
        Circle diameter.

    h
        Level of truncation.

    Returns
    -------
    A
        Area of truncated circle.
    """

    # This one does not require decorator for vectorization, as it is relatively simple,
    # regular numpy broadcasting applies

    y = h/D  # convenience variable, dimensionless level
    A = D**2/2*(np.arcsin(np.sqrt(y))-(1-2*y)*np.sqrt(y*(1-y)))

    return A


def area_cs_cover_ellipse(H: float, x: float, D: float, h: float) -> float:
    """_summary_

    Parameters
    ----------
    H
        _description_
    D
        _description_
    x
        _description_
    h
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    raise NotImplementedError(
        "Cross-sectional area for this type of covers is not supported yet")


def area_cs_cover_circle(x: float, D: float, h: float) -> float:
    """Computations for special case of elliptic cover with H = D/2

    Parameters
    ----------
    D
        _description_
    x
        _description_
    h
        _description_

    Returns
    -------
        _description_
    """
    return area_cs_cover_ellipse(D/2, D, x, h)


def area_cs_cover_cone(H: float, x: float, D: float, h: float) -> float:
    """_summary_

    Parameters
    ----------
    H
        _description_
    D
        _description_
    x
        _description_
    h
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    raise NotImplementedError(
        "Cross-sectional area for this type of covers is not supported yet")


def area_cs_cover_torus(H: float, D: float, x: float, h: float) -> float:
    """_summary_

    Parameters
    ----------
    H
        _description_
    D
        _description_
    x
        _description_
    h
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    raise NotImplementedError(
        "Cross-sectional area for this type of covers is not supported yet")


def area_cs_Utube_trunc(D_t: float) -> float:
    """This function computes cross-sectional area of the U-shaped tube in the middle
    region far apart from start/end of tube.

    Parameters
    ----------
    D_t
        U-tube representative diameter.

    Returns
    -------
        Cross-sectional area of the U-shaped tube (far apart from start/end of tube).
    """
    return 2*np.pi*(D_t/2)**2/4  # explicit formula for clarity


# -----------------------------------------------------------------------------
# PLANE CUT AREA COMPUTATIONS
# -----------------------------------------------------------------------------


def area_planecut_cylinder(L: float, D: float, h: float) -> float:
    """ This function computes area of a top surface formed by truncation of
    horizontal cylinder by a horizontal plane.

    Parameters
    ----------
    L
        Length of cylinder.

    D
        Diameter of cylinder.

    h
        Level of truncation.

    Returns
    -------
        Area of surface obtained with truncation of horizontal cylinder by
        horizontal plane.
    """
    return 2*L*np.sqrt(h*(D-h))


def area_planecut_cover_ellipse(H: float, D: float, h: float) -> float:
    """This function computes area of a top surface formed by truncation of
    elliptic cover by horizontal plane.

    Parameters
    ----------
    H
        Height of cover.
    D
        Diameter of cover's base.
    h
        Level of truncation.

    Returns
    -------
        Area of a surface formed by truncation of elliptic cover by horizontal plane.
    """
    return np.pi*H/D*(h*D-h**2)


def area_planecut_cover_circle(D: float, h: float) -> float:
    """This function computes area of a top surface formed by truncation of
    circular cover by horizontal plane. This is considered as a special case
    of elliptic cover wiht H = D/2.

    Parameters
    ----------
    D
        Diameter of cover's base.
    h
        Level of truncation.

    Returns
    -------
        Area of a surface formed by truncation of circular cover and horizontal plane
    """
    return area_planecut_cover_ellipse(D/2, D, h)


def area_planecut_cover_cone(H: float, D: float, h: float) -> float:
    """_summary_

    Parameters
    ----------
    H
        _description_
    D
        _description_
    h
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    raise NotImplementedError(
        "Plane cut area for this type of covers is not supported yet")


def area_planecut_cover_torus(H: float, D: float, h: float) -> float:
    """_summary_

    Parameters
    ----------
    H
        _description_
    D
        _description_
    h
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    raise NotImplementedError(
        "Plane cut area for this type of covers is not supported yet")


def area_planecut_section_horiz_general(
        H_left: float, L: float, H_right: float, D: float, h: float,
        left_cov: Callable, right_cov: Callable) -> float:
    """This function performs computations of area for top surface formed by truncation of
    horizontal section with two covers by horizontal plane. Computations are performed in
    a general manner.

    Parameters
    ----------
    H_left
        Left cover length.
    L
        Cylindrical part length.
    H_right
        Right cover length.
    D
        Diameter of section.
    h
        Truncation level.
    left_cov
        Function for planecut area of left cover with (H,D,h) signature.
    right_cov
        Function for planecut area of right cover with (H,D,h) signature.

    Returns
    -------
    A_planecut_section
        Area formed by horizontal truncation of horizontal section with two specified
        covers.
    """

    A_planecut_left_cover = left_cov(H_left, D, h)
    A_planecut_cylinder = area_planecut_cylinder(L, D, h)
    A_planecut_right_cover = left_cov(H_left, D, h)

    A_planecut_section = (
        A_planecut_left_cover+A_planecut_cylinder+A_planecut_right_cover)

    return A_planecut_section


def area_planecut_section_horiz_ellipses(
        H_left: float, L: float, H_right: float, D: float, h: float) -> float:
    """This function performs computations of area formed by horizontal truncation of
    horizontal section with two elliptic covers. Considered as special case.

    Parameters
    ----------
    H_left
        Left cover length.
    L
        Cylindrical part length.
    H_right
        Right cover length.
    D
        Diameter of section.
    h
        Truncation level.

    Returns
    -------
    A_planecut_section
        Area formed by horizontal truncation of horizontal section with two elliptic
        covers.
    """
    A_planecut_section = area_planecut_section_horiz_general(
        H_left, L, H_right, D, h,
        left_cov=area_planecut_cover_ellipse,
        right_cov=area_planecut_cover_ellipse)
    return A_planecut_section


# -----------------------------------------------------------------------------
# VOLUME COMPUTATIONS
# -----------------------------------------------------------------------------


def volume_cylinder_trunc(L: float, D: float, h: float) -> float:
    """This function computes volume of horizontal cylinder truncated by a horizontal
    plane.

    Parameters
    ----------
    L
        Length of cylinder.
    D
        Diameter of cylinder.
    h
        Truncation level.

    Returns
    -------
        Volume of truncated horizontal cylinder.
    """
    A = area_cs_circle_trunc(D, h)
    return L*A


def volume_cover_elliptic_trunc(H: float, D: float, h: float) -> float:
    """This function computes volume of semi-ellipsoid truncated by a horizontal
    plane.

    Parameters
    ----------
    H
        Length of semi-ellipsoid axis.
    D
        Base diameter.
    h
        Truncation level.

    Returns
    -------
        Volume of semi-ellipsoid truncated by a horizontal plane.
    """
    return np.pi*H/D*(D*h**2/2-h**3/3)


def volume_cover_circle_trunc(D: float, h: float) -> float:
    """This function computes volume of semi-sphere truncated by a horizontal
    plane. This is considered as a special case of semi-ellipsoid with H = D/2.

    Parameters
    ----------
    D
        Base diameter.
    h
        Truncation level.

    Returns
    -------
        Volume of semi-sphere truncated by a horizontal plane.
    """
    return volume_cover_elliptic_trunc(D/2, D, h)


def volume_cover_cone_trunc(H: float, D: float, h: float):
    """_summary_

    Parameters
    ----------
    H
        _description_
    D
        _description_
    h
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    raise NotImplementedError(
        "Volume for this type of covers is not supported yet")


def volume_cover_torus_trunc(H: float, D: float, h: float):
    """_summary_

    Parameters
    ----------
    H
        _description_
    D
        _description_
    h
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    raise NotImplementedError(
        "Volume for this type of covers is not supported yet")


def volume_Utube_trunc():
    pass


def volume_section_horiz_general(H_left: float, L: float, H_right: float, D: float,
                                 h: float, left_cov: Callable, right_cov: Callable
                                 ) -> float:
    """This function computes volume of horizontal section with two covers truncated by
    horizontal plane. Computations are performed in a general manner.

    Parameters
    ----------
    H_left
        Left cover length.
    L
        Cylindrical part length.
    H_right
        Right cover length.
    D
        Diameter of section.
    h
        Truncation level.
    left_cov
        Function for volume of left cover with (H,D,h) signature.
    right_cov
        Function for volume of right cover with (H,D,h) signature.

    Returns
    -------
        Volume of horizontal section with specified covers truncated by horizontal plane.
    """
    V_cylinder_trunc = volume_cylinder_trunc(L, D, h)
    V_left_cover_trunc = left_cov(H_left, D, h)
    V_right_cover_trunc = right_cov(H_right, D, h)

    V = V_cylinder_trunc + V_left_cover_trunc + V_right_cover_trunc

    return V


def volume_section_horiz_ellipses(H_left: float, L: float, H_right: float,
                                  D: float, h: float) -> float:
    """This function computes volume of horizontal section with two elliptic covers
    truncated by horizontal plane.

    Parameters
    ----------
    H_left
        Left cover length.
    L
        Cylindrical part length.
    H_right
        Right cover length.
    D
        Diameter of section.
    h
        Truncation level.

    Returns
    -------
    V
        Volume of horizontal section with elliptic covers truncated by horizontal plane.
    """
    V = volume_section_horiz_general(H_left, L, H_right, D, h,
                                     left_cov=volume_cover_elliptic_trunc,
                                     right_cov=volume_cover_elliptic_trunc)
    return V


# -----------------------------------------------------------------------------
# GENERIC PROCEDURES INVOLVING GEOMETRY-RELATED COMPUTATIONS
# -----------------------------------------------------------------------------


def graduate(
        h_min: float, h_max: float, vol_of_lvl: Callable, N: int = 50) -> tuple[Iterable]:
    """This function computes volume values that correspond to level values ranging from
    min to max possible value for element with provided volume dependency on level.

    Parameters
    ----------
    h_min
        Minimal possible level.
    h_max
        Maximal possible level.
    vol_of_lvl
        Function that relates level and volume in the element.
    N, optional
        Number of point between min and max level, by default 50.

    Returns
    -------
        Tuple of arrays, first array correspond to generated levels, second to computed
        volumes.
    """
    h = np.linspace(h_min, h_max, N)
    V = vol_of_lvl(h)
    return h, V  # type: ignore


def inverse_graduate(V_target: float, h: Iterable[float], V: Iterable[float]) -> float:
    """This function performs computations to detemine level value that corresponds to
    target volume. Computations are performed by means of linear interpolation over
    arrays of levels and volumes (see function for graduation).

    Parameters
    ----------
    V_target
        Target value of volume for which level should be determined.
    h
        Array of levels ranging from min to max possible value of level.
    V
        Array of volumes corresponding to provided array of levels.

    Returns
    -------
        Interpolated value of level that corresponds to V_target
    """
    # pyrightcries anout types, but everything should be fine with iterables
    return np.interp(V_target, V, h)  # type: ignore


if __name__ == '__main__':
    print(volume_section_horiz_ellipses(0.8, 1, 0.8, 1, 0.5))
    # yildes same value as function in legacy, additional tests should be conducted in
    # dedicated directory
