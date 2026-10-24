import itertools
import math
import unittest

import numpy as np

from mesh_coverage import certify_mesh_k_coverage


class MeshCoverageTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0.], [3., 0.], [0., 4.]])
        self.faces = [[0, 1, 2]]
        # Exact Euclidean distances to contacts at the three vertices.
        self.upper = np.array([[0., 3., 4.], [3., 0., 5.], [4., 5., 0.]])

    def test_changing_witnesses_and_k_fold_coverage(self):
        for k, radius in [(1, 3.), (2, 7.), (3, 8.)]:
            result = certify_mesh_k_coverage(
                self.vertices, self.faces, self.upper, radius, k=k)
            self.assertTrue(result.all_faces_pass)
            self.assertEqual(result.passing_faces, (0,))
            for i in range(21):
                for j in range(21-i):
                    x = (i*self.vertices[0] + j*self.vertices[1]
                         + (20-i-j)*self.vertices[2]) / 20
                    distances = np.linalg.norm(self.vertices-x, axis=1)
                    self.assertGreaterEqual(np.count_nonzero(distances < radius), k)
                    for failed in itertools.combinations(range(3), k-1):
                        survivors = [d for c, d in enumerate(distances) if c not in failed]
                        self.assertLess(min(survivors), radius)

    def test_exact_open_closed_boundary(self):
        vertices = [[0., 0., 0.], [1., 1., 1.], [.5, 0., 0.]]
        # Genuine upper bounds; the nearest bound is zero at each vertex.
        upper = [[0., 2., 1.], [2., 0., 2.], [1., 2., 0.]]
        closed = certify_mesh_k_coverage(vertices, self.faces, upper, 1., open_sets=False)
        opened = certify_mesh_k_coverage(vertices, self.faces, upper, 1.)
        self.assertTrue(closed.all_faces_pass)  # D^2 = 3, (r-m)^2 = 1.
        self.assertFalse(opened.all_faces_pass)
        self.assertTrue(certify_mesh_k_coverage(
            vertices, self.faces, upper, np.nextafter(1., np.inf)).all_faces_pass)
        self.assertFalse(certify_mesh_k_coverage(
            vertices, self.faces, upper, np.nextafter(1., 0.),
            open_sets=False).all_faces_pass)

    def test_scale_invariance_without_square_overflow(self):
        for exponent in [-900, -500, 0, 500, 900]:
            scale = 2. ** exponent
            result = certify_mesh_k_coverage(
                self.vertices*scale, self.faces, self.upper*scale, 3.*scale)
            self.assertTrue(result.all_faces_pass)
            unresolved = certify_mesh_k_coverage(
                self.vertices*scale, self.faces, self.upper*scale, 2.6*scale)
            self.assertEqual(unresolved.unresolved_faces, (0,))

    def test_unresolved_is_not_an_uncovered_claim(self):
        result = certify_mesh_k_coverage(self.vertices, self.faces, self.upper, 2.6)
        self.assertFalse(result.all_faces_pass)
        self.assertEqual(result.unresolved_faces, (0,))
        # Exact dyadic mesh, with rigorously rounded-up distance bounds.
        n = 32
        lattice = [(i, j) for i in range(n+1) for j in range(n+1-i)]
        index = {point: v for v, point in enumerate(lattice)}
        vertices = [[3*i/n, 4*j/n] for i, j in lattice]
        upper = []
        for ci, cj in [(0, 0), (n, 0), (0, n)]:
            row = []
            for i, j in lattice:
                square = ((3*(i-ci))**2 + (4*(j-cj))**2) * 1024**2
                integer_root = math.isqrt(square)
                ceiling = integer_root + (integer_root**2 != square)
                row.append(ceiling/(n*1024))
            upper.append(row)
        faces = []
        for i in range(n):
            for j in range(n-i):
                faces.append([index[i, j], index[i+1, j], index[i, j+1]])
                if i+j <= n-2:
                    faces.append([index[i+1, j], index[i+1, j+1], index[i, j+1]])
        refined = certify_mesh_k_coverage(vertices, faces, upper, 2.6)
        self.assertEqual(len(faces), n*n)
        self.assertTrue(refined.all_faces_pass)
        self.assertEqual(len(refined.passing_faces), n*n)

    def test_face_indices_and_degeneracies(self):
        for faces in [[], [[-1, 1, 2]], [[0, 1, 3]], [[0., 1., 2.]],
                      [[0, 0, 2]], [[0, 1]], [[0, 1, 2], [0, 0, 0]]]:
            with self.assertRaises(ValueError):
                certify_mesh_k_coverage(self.vertices, faces, self.upper, 3.)
        with self.assertRaises(ValueError):
            certify_mesh_k_coverage([[0., 0.], [1., 0.], [2., 0.]],
                                   self.faces, self.upper, 3.)

    def test_invalid_bounds_and_parameters(self):
        for k in [0, 4, 1.5, True]:
            with self.assertRaises(ValueError):
                certify_mesh_k_coverage(self.vertices, self.faces, self.upper, 3., k=k)
        for radius in [-1., np.inf, np.nan]:
            with self.assertRaises(ValueError):
                certify_mesh_k_coverage(self.vertices, self.faces, self.upper, radius)
        for upper in [np.ones((3, 2)), np.ones(3), np.empty((0, 3)),
                      [[-1., 0., 0.]], [[np.nan, 0., 0.]], [[np.inf, 0., 0.]]]:
            with self.assertRaises(ValueError):
                certify_mesh_k_coverage(self.vertices, self.faces, upper, 3.)
        with self.assertRaises(ValueError):
            certify_mesh_k_coverage([[np.nan, 0.], [3., 0.], [0., 4.]],
                                   self.faces, self.upper, 3.)
        with self.assertRaises(ValueError):
            certify_mesh_k_coverage(self.vertices, self.faces, self.upper, 3., open_sets='yes')


if __name__ == '__main__':
    unittest.main()
