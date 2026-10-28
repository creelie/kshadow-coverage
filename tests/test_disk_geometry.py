"""Geometric regression cases with known exact answers and topological effects."""
import itertools
import math
import unittest

import numpy as np

from kshadow import _mec_radius, disks_intersect, nerve_of_disks, shadow_betti


class DiskGeometryTests(unittest.TestCase):
    def test_acute_triangle_has_a_hole_at_every_binary_scale(self):
        # The circumradius is exactly 65/14, whereas r=4.4 is smaller.
        # All pairs intersect. Filling the triangle would erase a real H_1.
        points = np.array([[0., 0.], [8., 0.], [4., 7.]])
        for power in [-900, -40, -30, 0, 30, 900]:
            with self.subTest(power=power):
                scale = math.ldexp(1., power)
                p = points * scale
                r = 4.4 * scale
                self.assertFalse(disks_intersect((0, 1, 2), p, r))
                nerve = nerve_of_disks(p, r)
                self.assertEqual({len(f) for f in nerve}, {1, 2})
                self.assertEqual(shadow_betti(nerve, 1)[:2], (1, 1))
                self.assertAlmostEqual(_mec_radius(p) / scale, 65 / 14)

    def test_translation_rotation_and_order_do_not_change_membership(self):
        points = np.array([[0., 0.], [8., 0.], [4., 7.]])
        for p in [points, points + 2. ** 40, points[:, ::-1], -points]:
            for order in itertools.permutations(range(3)):
                self.assertFalse(disks_intersect(order, p, 4.4))
                self.assertTrue(disks_intersect(order, p, 4.7))

    def test_pair_tangency(self):
        p = np.array([[0., 0.], [2., 0.]])
        self.assertFalse(disks_intersect((0, 1), p, 1.))
        self.assertTrue(disks_intersect((0, 1), p, 1., strict=False))
        self.assertNotIn(frozenset([0, 1]), nerve_of_disks(p, 1.))

    def test_triple_tangency_and_adjacent_floating_radii(self):
        # Three acute-triangle vertices exactly on the radius-five circle.
        p = np.array([[5., 0.], [-3., 4.], [-3., -4.]])
        self.assertEqual(_mec_radius(p), 5.)
        self.assertFalse(disks_intersect((0, 1, 2), p, 5.))
        self.assertTrue(disks_intersect((0, 1, 2), p, 5., strict=False))
        self.assertFalse(disks_intersect((0, 1, 2), p, math.nextafter(5., 0.), strict=False))
        self.assertTrue(disks_intersect((0, 1, 2), p, math.nextafter(5., math.inf)))

    def test_collinear_and_coincident_points(self):
        p = np.array([[0., 0.], [0., 0.], [2., 0.]])
        self.assertEqual(_mec_radius(p), 1.)
        self.assertFalse(disks_intersect((0, 1, 2), p, 1.))
        self.assertTrue(disks_intersect((0, 1, 2), p, 1., strict=False))
        self.assertTrue(disks_intersect((0, 1), p, 0., strict=False))
        self.assertFalse(disks_intersect((0, 1), p, 0.))

    def test_helly_for_more_than_three_disks(self):
        square = np.array([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]])
        self.assertFalse(disks_intersect(range(4), square, 1.4))
        self.assertTrue(disks_intersect(range(4), square, 1.5))
        nerve = nerve_of_disks(square, 1.5, max_size=3)
        self.assertIn(frozenset(range(4)), nerve)

    def test_empty_family_and_invalid_geometry(self):
        self.assertTrue(disks_intersect([], [], 0.))
        self.assertEqual(len(nerve_of_disks([], 1.)), 0)
        for bad in [math.nan, math.inf, -1., 0.]:
            with self.subTest(radius=bad), self.assertRaises(ValueError):
                nerve_of_disks([[0., 0.]], bad)
        for bad in [[[math.nan, 0.]], [[0., math.inf]], [[0., 0., 0.]], np.empty((0, 3))]:
            with self.subTest(points=bad), self.assertRaises(ValueError):
                nerve_of_disks(bad, 1.)


if __name__ == '__main__':
    unittest.main()
