import numpy as np
from typing import Callable, Iterable, overload
from numpy.typing import NDArray
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

# types generally should be defined somewhere else, should fix later
type Numeric = float | NDArray


def area_cs_circle_trunc(diameter: Numeric, level: Numeric) -> Numeric:
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
        length_semiaxis: Numeric,
        axial_position: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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
        axial_position: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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
    return area_cs_cover_ellipse(diameter/2, axial_position, diameter, level)


def area_cs_cover_cone(
        length_semiaxis: Numeric,
        axial_position: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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
        length_semiaxis: Numeric,
        axial_position: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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


def area_cs_Utube_trunc(diameter: Numeric) -> Numeric:
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
    return 2*np.pi*(diameter/2)**2/4  # explicit formula for clarity


# -----------------------------------------------------------------------------
# PLANE CUT AREA COMPUTATIONS
# -----------------------------------------------------------------------------


def area_planecut_cylinder(length: Numeric, diameter: Numeric, level: Numeric) -> Numeric:
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


def area_planecut_cover_ellipse(
        length_semiaxis: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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


def area_planecut_cover_circle(diameter: Numeric, level: Numeric) -> Numeric:
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
        length_semiaxis: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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
        length_semiaxis: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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


def area_planecut_section_horiz_general(
        length_semiaxis_left: Numeric,
        length_cylinder: Numeric,
        length_semiaxis_right: Numeric,
        diameter: Numeric,
        level: Numeric,
        area_cover_left_fn: Callable[[Numeric, Numeric, Numeric], Numeric],
        area_cover_right_fn: Callable[[Numeric, Numeric, Numeric], Numeric]) -> Numeric:
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


def area_planecut_section_horiz_ellipses(
        length_semiaxis_left: Numeric,
        length_cylinder: Numeric,
        length_semiaxis_right: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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


def volume_cylinder_trunc(
        length: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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


def volume_cover_elliptic_trunc(
        length_semiaxis: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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


def volume_cover_circle_trunc(diameter: Numeric, level: Numeric) -> Numeric:
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
        length_semiaxis: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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
        length_semiaxis: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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


def volume_section_horiz_general(
    length_semiaxis_left: Numeric,
    length_cylinder: Numeric,
    length_semiaxis_right: Numeric,
    diameter: Numeric,
    level: Numeric,
    volume_cover_left_fn: Callable[
        [Numeric, Numeric, Numeric], Numeric],
    volume_cover_right_fn: Callable[
        [Numeric, Numeric, Numeric], Numeric]
) -> Numeric:
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


def volume_section_horiz_ellipses(
        length_semiaxis_left: Numeric,
        length_cylinder: Numeric,
        length_semiaxis_right: Numeric,
        diameter: Numeric,
        level: Numeric) -> Numeric:
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
        level_min: Numeric,
        level_max: Numeric,
        volume_fn: Callable[[Numeric], Numeric],
        number_of_points: int = 50
) -> tuple[Numeric, Numeric]:
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


def inverse_graduate(
        volume_target: Numeric,
        level_graduated: Numeric,
        volume_graduated: Numeric) -> Numeric:
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
    # pyright cries anout types, but everything should be fine with iterables
    # type: ignore
    return np.interp(volume_target, volume_graduated, level_graduated)


if __name__ == '__main__':
    print(volume_section_horiz_ellipses(0.8, 1, 0.8, 1, 0.5))
    # yildes same value as function in legacy, additional tests should be conducted in
    # dedicated directory
