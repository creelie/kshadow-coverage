import math
from fractions import Fraction
import unittest
import numpy as np
from certified_geodesic import certified_graph_vertex_bounds
from mesh_coverage import certify_mesh_k_coverage


class CertifiedPathTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0.,0.],[3.,0.],[0.,4.]])
        self.faces = [[0,1,2]]

    def test_exact_paths_and_truncation(self):
        full = certified_graph_vertex_bounds(self.vertices, self.faces, [0,1,2], quantum=2**-10)
        np.testing.assert_array_equal(full.upper, [[0,3,4],[3,0,5],[4,5,0]])
        short = certified_graph_vertex_bounds(self.vertices, self.faces, [0], search_limit=3)
        self.assertEqual(short.upper[0,1], 3.)
        self.assertGreaterEqual(short.upper[0,2], 4.)

    def test_irrational_edges_are_outward(self):
        v = [[0.,0.],[1.,1.],[2.,0.]]
        for quantum in [2**-4, 2**-20, 2**-40]:
            out = certified_graph_vertex_bounds(v, self.faces, [0], quantum=quantum)
            self.assertGreaterEqual(Fraction(out.upper[0,1])**2, 2)
            self.assertLess(out.upper[0,1], math.sqrt(2)+2*quantum)

    def test_scale_and_coverage_contract(self):
        for exponent in [-500,0,500]:
            scale = 2.**exponent
            out = certified_graph_vertex_bounds(self.vertices*scale,self.faces,[0,1,2],quantum=2.**(exponent-20))
            self.assertTrue(certify_mesh_k_coverage(self.vertices*scale,self.faces,out.upper,3*scale).all_faces_pass)

    def test_disconnected_invalid_and_targets(self):
        out = certified_graph_vertex_bounds(self.vertices,self.faces,[1],targets=[0,2])
        np.testing.assert_array_equal(out.upper, [[3,5]])
        with self.assertRaises(ValueError):
            certified_graph_vertex_bounds([[0,0],[3,0],[0,4],[9,9]],self.faces,[0])
        for kwargs in [{'quantum':0},{'quantum':.1},{'quantum':np.inf},{'search_limit':-1},{'targets':[3]}]:
            with self.assertRaises(ValueError):
                certified_graph_vertex_bounds(self.vertices,self.faces,[0],**kwargs)


if __name__ == '__main__':
    unittest.main()
