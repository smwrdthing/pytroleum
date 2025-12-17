import pytest
from pytroleum import meter


class TestAreaCalculations:
    """"Class for testing area calculation functions"""

    @pytest.mark.parametrize("diameter, level, expected_area",
                             [
                                 (0.092, 0.062, 0.004155),
                                 (0.092, 0.05, 0.003324),
                                 (0.092, 0.08, 0.005841)
                             ]
                             )
    def test_area_cs_circle_trunc(self, diameter, level, expected_area):
        """This function tests a function that computes the area of a circle
        which is truncated by a horizontal straight line.

        Parameters
        ----------
        diameter
            Circle diameter.

        level
            Level of truncation.

        expected_area
            Expected area of truncated circle.
        """
        result = meter.area_cs_circle_trunc(diameter, level)
        assert abs(result - expected_area) < 1e-3

    @pytest.mark.parametrize("diameter, expected_area",
                             [
                                 (0.3, 0.141372),
                                 (0.18, 0.050894),
                                 (0.382, 0.229217),
                             ]
                             )
    def test_area_cs_Utube_trunc(self, diameter, expected_area):
        """This function tests a function that computes cross-sectional area of the
        U-shaped tube in the middle region far apart from start/end of tube.

        Parameters
        ----------
        diameter
            U-tube representative diameter.

        expected_area
            Expected cross-sectional area of the U-shaped tube (far apart from start/end
            of tube).
        """
        result = meter.area_cs_Utube_trunc(diameter)
        assert abs(result - expected_area) < 1e-3

    @pytest.mark.parametrize("length, diameter, level, expected_area",
                             [
                                 (9.5, 2, 1.5, 16.45448)
                             ]
                             )
    def test_area_planecut_cylinder(self, length, diameter, level, expected_area):
        """ This function tests a function that computes area of a top surface formed by
        truncation of horizontal cylinder by a horizontal plane.

        Parameters
        ----------
        length
            Length of cylinder.

        diameter
            Diameter of cylinder.

        level
            Level of truncation.

        expected_area
            Expected area of surface obtained with truncation of horizontal cylinder by
            horizontal plane.
        """
        result = meter.area_planecut_cylinder(length, diameter, level)
        assert abs(result - expected_area) < 1e-3

    @pytest.mark.parametrize("length_semiaxis, diameter, level, expected_area",
                             [
                                 (0.05499, 0.2, 0.14, 0.007917)
                             ]
                             )
    def test_area_planecut_cover_ellipse(
            self, length_semiaxis, diameter, level, expected_area):
        """This function tests a function that computes area of a top surface formed
        by truncation of elliptic cover by horizontal plane.

        Parameters
        ----------
        length_semiaxis
            Height of cover.

        diameter
            Diameter of cover's base.

        level
            Level of truncation.

        expected_area
            Expected area of a surface formed by truncation of elliptic cover by
            horizontal plane.
        """
        result = meter.area_planecut_cover_ellipse(
            length_semiaxis, diameter, level)
        assert abs(result - expected_area) < 1e-3

    @pytest.mark.parametrize("diameter, level, expected_area",
                             [
                                 (0.204, 0.14314, 0.013684)
                             ]
                             )
    def test_area_planecut_cover_circle(self, diameter, level, expected_area):
        """This function tests a function that computes area of a top surface formed
        by truncation of circular cover by horizontal plane. This is considered as
        a special case of elliptic cover wiht H = D/2.

        Parameters
        ----------
        diameter
            Diameter of cover's base.

        level
            Level of truncation.

        expected_area
            Expected area of a surface formed by truncation of circular cover and
            horizontal plane.
        """
        result = meter.area_planecut_cover_circle(diameter, level)
        assert abs(result - expected_area) < 1e-3

    @pytest.mark.skip(reason="Test data to be added later")
    def test_area_planecut_section_horiz_ellipses(
            self, length_semiaxis_left, length_cylinder, length_semiaxis_right,
            diameter, level, expected_area):
        """This function tests a function performs computations of area formed by
        horizontal truncation of horizontal section with two elliptic covers. Considered
        as special case.

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

        expected_area
            Expected area formed by horizontal truncation of horizontal section with two
            elliptic covers.
        """
        result = meter.area_planecut_section_horiz_ellipses(
            length_semiaxis_left, length_cylinder, length_semiaxis_right,
            diameter, level)
        assert abs(result - expected_area) < 1e-3


class TestVolumeCalculations:
    """"Class for testing volume calculation functions"""

    @pytest.mark.parametrize("length, diameter, level, expected_volume",
                             [
                                 (0.076, 0.05628, 0.0331, 0.000073)
                             ]
                             )
    def test_volume_cylinder_trunc(
            self, length, diameter, level, expected_volume):
        """This function tests a function computes volume of horizontal cylinder truncated
        by a horizontal plane.

        Parameters
        ----------
        length
            Length of cylinder.

        diameter
            Diameter of cylinder.

        level
            Truncation level.

        expected_volume
            Expected volume of truncated horizontal cylinder.
        """
        result = meter.volume_cylinder_trunc(length, diameter, level)
        assert abs(result - expected_volume) < 1e-3

    @pytest.mark.parametrize("length_semiaxis, diameter, level, expected_volume",
                             [
                                 (0.05499, 0.2, 0.14, 0.000991)
                             ]
                             )
    def test_volume_cover_elliptic_trunc(
            self, length_semiaxis, diameter, level, expected_volume):
        """This function tests a function computes volume of semi-ellipsoid truncated by a
        horizontal plane.

        Parameters
        ----------
        length_semiaxis
            Length of semi-ellipsoid axis.

        diameter
            Base diameter.

        level
            Truncation level.

        expected_volume
            Expected volume of semi-ellipsoid truncated by a horizontal plane.
        """
        result = meter.volume_cover_elliptic_trunc(
            length_semiaxis, diameter, level)
        assert abs(result - expected_volume) < 1e-3

    @pytest.mark.parametrize("diameter, level, expected_volume",
                             [
                                 (0.2, 0.14, 0.001747)
                             ]
                             )
    def test_volume_cover_circle_trunc(self, diameter, level, expected_volume):
        """This function tests a function computes volume of semi-sphere truncated
        by a horizontal plane. This is considered as a special case of semi-ellipsoid
        with H = D/2.

        Parameters
        ----------
        diameter
            Base diameter.

        level
            Truncation level.

        expected_volume
            Expected volume of semi-sphere truncated by a horizontal plane.
        """
        result = meter.volume_cover_circle_trunc(diameter, level)
        assert abs(result - expected_volume) < 1e-3

    @pytest.mark.skip(reason="Test data to be added later")
    def test_volume_section_horiz_ellipses(
            self, length_semiaxis_left, length_cylinder, length_semiaxis_right,
            diameter, level, expected_volume):
        """This function tests a function computes volume of horizontal section with two
        elliptic covers truncated by horizontal plane.

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

        expected_volume
            Expected volume of horizontal section with elliptic covers truncated by
            horizontal plane.
        """
        result = meter.volume_section_horiz_ellipses(
            length_semiaxis_left, length_cylinder, length_semiaxis_right,
            diameter, level)
        assert abs(result - expected_volume) < 1e-3
