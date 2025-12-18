import numpy as np

from typing import Any, NoReturn, Callable, overload
from numpy.typing import NDArray
from numpy import float64

# All type hints should be implemented with @overload decorator!

# !DISCLAIMER! : All functions work with SI units!


# -----------------------------------------------------------------------------
# CROSS-SECTIONAL AREA COMPUTATIONS
# -----------------------------------------------------------------------------

@overload
def area_cs_circle_trunc(
        diameter: float | float64, level: float | float64) -> float | float64:
    ...


@overload
def area_cs_circle_trunc(
        diameter: float | float64, level: NDArray[float64]) -> NDArray[float64]:
    ...


def area_cs_circle_trunc(diameter, level):
    """This function computes area of circle which is truncated with horizontal
    straight line. Formula obtained by the integration of the following function
    from zero to h:

                        2*sqrt((D/2)**2 - (x-D/2)**2)

    Parameters
    ----------
    diameter
        Circle diameter.

    level
        Level of truncation.

    Returns
    -------
    area
        Area of truncated circle.
    """

    # This one does not require decorator for vectorization, as it is relatively simple,
    # regular numpy broadcasting applies

    y = level/diameter  # convenience variable, dimensionless level
    area = diameter**2/2*(np.arcsin(np.sqrt(y))-(1-2*y)*np.sqrt(y*(1-y)))

    return area


def area_cs_cover_ellipse(
        length_semiaxis,
        axial_position,
        diameter,
        level):
    """_summary_

    Parameters
    ----------
    length_semiaxis
        _description_
    diameter
        _description_
    axial_position
        _description_
    level
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


def area_cs_cover_circle(
        axial_position,
        diameter,
        level):
    """Computations for special case of elliptic cover with H = D/2

    Parameters
    ----------
    axial_position
        _description_
    diameter
        _description_
    level
        _description_

    Returns
    -------
        _description_
    """
    raise NotImplementedError(
        "Cross-sectional area for this type of covers is not supported yet")


def area_cs_cover_cone(
        length_semiaxis,
        axial_position,
        diameter,
        level):
    """_summary_

    Parameters
    ----------
    length_semiaxis
        _description_
    axial_position
        _description_
    diameter
        _description_
    level
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


def area_cs_cover_torus(
        length_semiaxis,
        axial_position,
        diameter,
        level):
    """_summary_

    Parameters
    ----------
    length_semiaxis
        _description_
    axial_position
        _description_
    diameter
        _description_
    level
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


def area_cs_Utube_trunc(diameter: float | float64):
    """This function computes cross-sectional area of the U-shaped tube in the middle
    region far apart from start/end of tube.

    Parameters
    ----------
    diameter
        U-tube representative diameter.

    Returns
    -------
        Cross-sectional area of the U-shaped tube (far apart from start/end of tube).
    """
    return 2*np.pi*diameter**2/4  # explicit formula for clarity


# -----------------------------------------------------------------------------
# PLANE CUT AREA COMPUTATIONS
# -----------------------------------------------------------------------------

@overload
def area_planecut_cylinder(
        length: float | float64,
        diameter: float | float64,
        level: float | float64) -> float | float64:
    ...


@overload
def area_planecut_cylinder(
        length: float | float64,
        diameter: float | float64,
        level: NDArray[float64]) -> NDArray[float64]:
    ...


def area_planecut_cylinder(length, diameter, level):
    """ This function computes area of a top surface formed by truncation of
    horizontal cylinder by a horizontal plane.

    Parameters
    ----------
    length
        Length of cylinder.

    diameter
        Diameter of cylinder.

    level
        Level of truncation.

    Returns
    -------
        Area of surface obtained with truncation of horizontal cylinder by
        horizontal plane.
    """
    return 2*length*np.sqrt(level*(diameter-level))


@overload
def area_planecut_cover_ellipse(
        length_semiaxis: float | float64,
        diameter: float | float64,
        level: float | float64) -> float | float64:
    ...


@overload
def area_planecut_cover_ellipse(
        length_semiaxis: float | float64,
        diameter: float | float64,
        level: NDArray[float64]) -> NDArray[float64]:
    ...


def area_planecut_cover_ellipse(length_semiaxis, diameter, level):
    """This function computes area of a top surface formed by truncation of
    elliptic cover by horizontal plane.

    Parameters
    ----------
    length_semiaxis
        Height of cover.
    diameter
        Diameter of cover's base.
    level
        Level of truncation.

    Returns
    -------
        Area of a surface formed by truncation of elliptic cover by horizontal plane.
    """
    return np.pi*length_semiaxis/diameter*(level*diameter-level**2)


@overload
def area_planecut_cover_circle(
        diameter: float | float64, level: float | float64) -> float | float64:
    ...


@overload
def area_planecut_cover_circle(
        diameter: float | float64, level: NDArray[float64]) -> NDArray[float64]:
    ...


def area_planecut_cover_circle(diameter, level):
    """This function computes area of a top surface formed by truncation of
    circular cover by horizontal plane. This is considered as a special case
    of elliptic cover wiht H = D/2.

    Parameters
    ----------
    diameter
        Diameter of cover's base.
    level
        Level of truncation.

    Returns
    -------
        Area of a surface formed by truncation of circular cover and horizontal plane
    """
    return area_planecut_cover_ellipse(diameter/2, diameter, level)


def area_planecut_cover_cone(
        length_semiaxis,
        diameter,
        level):
    """_summary_

    Parameters
    ----------
    length_semiaxis
        _description_
    diameter
        _description_
    level
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


def area_planecut_cover_torus(
        length_semiaxis,
        diameter,
        level):
    """_summary_

    Parameters
    ----------
    length_semiaxis
        _description_
    diameter
        _description_
    level
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


@overload
def area_planecut_section_horiz_general(
    length_semiaxis_left: float | float64,
    length_cylinder: float | float64,
    length_semiaxis_right: float | float64,
    diameter: float | float64,
    level: float | float64,
    area_cover_left_fn: Callable[
        [float | float64, float | float64, float | float64], float | float64],
    area_cover_right_fn: Callable[
        [float | float64, float | float64, float | float64], float | float64]
) -> float | float64:
    ...


@overload
def area_planecut_section_horiz_general(
    length_semiaxis_left: float | float64,
    length_cylinder: float | float64,
    length_semiaxis_right: float | float64,
    diameter: float | float64,
    level: NDArray[float64],
    area_cover_left_fn: Callable[
        [float | float64, float | float64, NDArray[float64]], NDArray[float64]],
    area_cover_right_fn: Callable[
        [float | float64, float | float64, NDArray[float64]], NDArray[float64]]
) -> NDArray[float64]:
    ...


def area_planecut_section_horiz_general(
        length_semiaxis_left, length_cylinder, length_semiaxis_right,
        diameter, level, area_cover_left_fn, area_cover_right_fn):
    """This function performs computations of area for top surface formed by truncation of
    horizontal section with two covers by horizontal plane. Computations are performed in
    a general manner.

    Parameters
    ----------
    length_semiaxis_left
        Left cover length.
    length_cylinder
        Cylindrical part length.
    length_semiaxis_right
        Right cover length.
    diameter
        Diameter of section.
    level
        Truncation level.
    left_cover_area
        Function for planecut area of left cover with (H,D,h) signature.
    right_cover_area
        Function for planecut area of right cover with (H,D,h) signature.

    Returns
    -------
    area_section
        Area formed by horizontal truncation of horizontal section with two specified
        covers.
    """

    A_cover_left = area_cover_left_fn(length_semiaxis_left, diameter, level)
    A_cover_right = area_cover_right_fn(length_semiaxis_right, diameter, level)
    A_cylinder = area_planecut_cylinder(length_cylinder, diameter, level)

    area_section = A_cover_left+A_cylinder+A_cover_right

    return area_section


@overload
def area_planecut_section_horiz_ellipses(
        length_semiaxis_left: float | float64,
        length_cylinder: float | float64,
        length_semiaxis_right: float | float64,
        diameter: float | float64,
        level: float | float64) -> float | float64:
    ...


@overload
def area_planecut_section_horiz_ellipses(
        length_semiaxis_left: float | float64,
        length_cylinder: float | float64,
        length_semiaxis_right: float | float64,
        diameter: float | float64,
        level: NDArray[float64]) -> NDArray[float64]:
    ...


def area_planecut_section_horiz_ellipses(
        length_semiaxis_left, length_cylinder, length_semiaxis_right,
        diameter, level):
    """This function performs computations of area formed by horizontal truncation of
    horizontal section with two elliptic covers. Considered as special case.

    Parameters
    ----------
    length_semiaxis_left
        Left cover length.
    length_cylinder
        Cylindrical part length.
    length_semiaxis_right
        Right cover length.
    diameter
        Diameter of section.
    level
        Truncation level.

    Returns
    -------
    A_planecut_section
        Area formed by horizontal truncation of horizontal section with two elliptic
        covers.
    """

    # NOTE : failed test for this function

    A_planecut_section = area_planecut_section_horiz_general(
        length_semiaxis_left,
        length_cylinder,
        length_semiaxis_right,
        diameter,
        level,
        area_cover_left_fn=area_planecut_cover_ellipse,
        area_cover_right_fn=area_planecut_cover_ellipse)
    return A_planecut_section


# -----------------------------------------------------------------------------
# VOLUME COMPUTATIONS
# -----------------------------------------------------------------------------

@overload
def volume_cylinder_trunc(length: float | float64, diameter: float | float64,
                          level: float | float64) -> float | float64:
    ...


@overload
def volume_cylinder_trunc(length: float | float64, diameter: float | float64,
                          level: NDArray[float64]) -> NDArray[float64]:
    ...


def volume_cylinder_trunc(length, diameter, level):
    """This function computes volume of horizontal cylinder truncated by a horizontal
    plane.

    Parameters
    ----------
    length
        Length of cylinder.
    diameter
        Diameter of cylinder.
    level
        Truncation level.

    Returns
    -------
        Volume of truncated horizontal cylinder.
    """
    return length*area_cs_circle_trunc(diameter, level)


@overload
def volume_cover_elliptic_trunc(
        length_semiaxis: float | float64,
        diameter: float | float64,
        level: float | float64) -> float | float64:
    ...


@overload
def volume_cover_elliptic_trunc(
        length_semiaxis: float | float64,
        diameter: float | float64,
        level: NDArray[float64]) -> NDArray[float64]:
    ...


def volume_cover_elliptic_trunc(length_semiaxis, diameter, level):
    """This function computes volume of semi-ellipsoid truncated by a horizontal
    plane.

    Parameters
    ----------
    length_semiaxis
        Length of semi-ellipsoid axis.
    diameter
        Base diameter.
    level
        Truncation level.

    Returns
    -------
        Volume of semi-ellipsoid truncated by a horizontal plane.
    """
    return np.pi*length_semiaxis/diameter*(diameter*level**2/2-level**3/3)


@overload
def volume_cover_circle_trunc(
        diameter: float | float64, level: float | float64) -> float | float64:
    ...


@overload
def volume_cover_circle_trunc(
        diameter: float | float64, level: NDArray[float64]) -> NDArray[float64]:
    ...


def volume_cover_circle_trunc(diameter, level):
    """This function computes volume of semi-sphere truncated by a horizontal
    plane. This is considered as a special case of semi-ellipsoid with H = D/2.

    Parameters
    ----------
    diameter
        Base diameter.
    level
        Truncation level.

    Returns
    -------
        Volume of semi-sphere truncated by a horizontal plane.
    """
    return volume_cover_elliptic_trunc(diameter/2, diameter, level)


def volume_cover_cone_trunc(
        length_semiaxis,
        diameter,
        level):
    """_summary_

    Parameters
    ----------
    length_semiaxis
        _description_
    diameter
        _description_
    level
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
        "Volume for this type of covers is not supported yet")


def volume_cover_torus_trunc(
        length_semiaxis,
        diameter,
        level):
    """_summary_

    Parameters
    ----------
    length_semiaxis
        _description_
    diameter
        _description_
    level
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    raise NotImplementedError(
        "Volume for this type of covers is not supported yet")


def volume_Utube_trunc():
    raise NotImplementedError(
        "Volume for this element is not supported yet")


@overload
def volume_section_horiz_general(
    length_semiaxis_left: float | float64,
    length_cylinder: float | float64,
    length_semiaxis_right: float | float64,
    diameter: float | float64,
    level: float | float64,
    volume_cover_left_fn: Callable[
        [float | float64, float | float64, float | float64], float | float64],
    volume_cover_right_fn: Callable[
        [float | float64, float | float64, float | float64], float | float64]
):
    ...


@overload
def volume_section_horiz_general(
    length_semiaxis_left: float | float64,
    length_cylinder: float | float64,
    length_semiaxis_right: float | float64,
    diameter: float | float64,
    level: float | float64,
    volume_cover_left_fn: Callable[
        [float | float64, float | float64, NDArray[float64]], NDArray[float64]],
    volume_cover_right_fn: Callable[
        [float | float64, float | float64, NDArray[float64]], NDArray[float64]]
):
    ...


def volume_section_horiz_general(
    length_semiaxis_left, length_cylinder, length_semiaxis_right,
    diameter, level, volume_cover_left_fn, volume_cover_right_fn
):
    """This function computes volume of horizontal section with two covers truncated by
    horizontal plane. Computations are performed in a general manner.

    Parameters
    ----------
    length_semiaxis_left
        Left cover length.
    length_cylinder
        Cylindrical part length.
    length_semiaxis_right
        Right cover length.
    diameter
        Diameter of section.
    level
        Truncation level.
    volume_cover_left_fn
        Function for volume of left cover with (H,D,h) signature.
    volume_cover_right_fn
        Function for volume of right cover with (H,D,h) signature.

    Returns
    -------
    volume_section
        Volume of horizontal section with specified covers truncated by horizontal plane.
    """
    V_cylinder = volume_cylinder_trunc(length_cylinder, diameter, level)
    V_left_cover = volume_cover_left_fn(length_semiaxis_left, diameter, level)
    V_right_cover = volume_cover_right_fn(
        length_semiaxis_right, diameter, level)

    volume_section = V_cylinder + V_left_cover + V_right_cover

    return volume_section


@overload
def volume_section_horiz_ellipses(
        length_semiaxis_left: float | float64,
        length_cylinder: float | float64,
        length_semiaxis_right: float | float64,
        diameter: float | float64,
        level: float | float64) -> float | float64:
    ...


@overload
def volume_section_horiz_ellipses(
        length_semiaxis_left: float | float64,
        length_cylinder: float | float64,
        length_semiaxis_right: float | float64,
        diameter: float | float64,
        level: NDArray[float64]) -> NDArray[float64]:
    ...


def volume_section_horiz_ellipses(
        length_semiaxis_left,
        length_cylinder,
        length_semiaxis_right,
        diameter,
        level):
    """This function computes volume of horizontal section with two elliptic covers
    truncated by horizontal plane.

    Parameters
    ----------
    length_semiaxis_left
        Left cover length.
    length_cylinder
        Cylindrical part length.
    length_semiaxis_right
        Right cover length.
    diameter
        Diameter of section.
    level
        Truncation level.

    Returns
    -------
    V
        Volume of horizontal section with elliptic covers truncated by horizontal plane.
    """
    volume_section = volume_section_horiz_general(
        length_semiaxis_left,
        length_cylinder,
        length_semiaxis_right,
        diameter,
        level,
        volume_cover_left_fn=volume_cover_elliptic_trunc,
        volume_cover_right_fn=volume_cover_elliptic_trunc)
    return volume_section


# -----------------------------------------------------------------------------
# GENERIC PROCEDURES INVOLVING GEOMETRY-RELATED COMPUTATIONS
# -----------------------------------------------------------------------------


def graduate(
        level_min: float | float64,
        level_max: float | float64,
        volume_fn: Callable[[NDArray[float64]], NDArray[float64]],
        number_of_points: int = 50
) -> tuple[NDArray[float64], NDArray[float64]]:
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
    (level, volume)
        Tuple of arrays, first array correspond to generated levels, second to computed
        volumes.
    """
    level = np.linspace(level_min, level_max, number_of_points)
    volume = volume_fn(level)
    return level, volume


@overload
def inverse_graduate(
        volume_target: float | float64,
        level_graduated: NDArray[float64],
        volume_graduated: NDArray[float64]) -> float | float64:
    ...


@overload
def inverse_graduate(
        volume_target: NDArray[float64],
        level_graduated: NDArray[float64],
        volume_graduated: NDArray[float64]) -> NDArray[float64]:
    ...


def inverse_graduate(volume_target, level_graduated, volume_graduated):
    """This function performs computations to detemine level value that corresponds to
    target volume. Computations are performed by means of linear interpolation over
    arrays of levels and volumes (see function for graduation).

    Parameters
    ----------
    volume_target
        Target value of volume for which level should be determined.
    level_graduated
        Array of levels ranging from min to max possible value of level.
    volume_graduated
        Array of volumes corresponding to provided array of levels.

    Returns
    -------
        Interpolated value of level that corresponds to V_target
    """
    return np.interp(volume_target, volume_graduated, level_graduated)


if __name__ == '__main__':
    print(volume_section_horiz_ellipses(0.8, 1, 0.8, 1, 0.5))
    # yildes same value as function in legacy, additional tests should be conducted in
    # dedicated directory
