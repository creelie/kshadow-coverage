# Exact disk predicates and a scaling counterexample

The planar predicates interpret coordinates and radii **after NumPy binary64
conversion**. Each resulting floating-point value is a rational number, which
`Fraction` represents exactly. The membership decision uses squared radii and
rational comparisons, including the strict distinction between open and closed
disks. Measurement uncertainty and the quality of a cortical distance model are
separate from arithmetic exactness.

## Counterexample to the previous absolute tolerances

Take centers `(0,0), (8,0), (4,7)` and common radius `4.4`. The triangle is acute.
Its circumcenter is `(4,33/14)` and its minimum enclosing radius is `65/14`,
strictly larger than `4.4`. Thus all three disks have empty common intersection,
although every pair intersects. Their union has one component and one hole.

The previous implementation returned this answer at unit scale. After scaling
both centers and radius by `2^-30`, its absolute determinant threshold treated
the triangle as collinear and substituted half the longest side. It incorrectly
inserted the triangular nerve face and erased the hole. At still smaller scales,
the absolute diameter-containment tolerance could also alter the decision.

The new tests check this topological consequence across binary scales, along with
translations, coordinate symmetries, input permutations and exact tangencies.

## Squared-radius formula

For two points with squared separation `a`, the minimum enclosing squared radius
is `a/4`. For three points, let `a,b,c` be the three squared side lengths and let
`D` be the determinant of two side vectors from one vertex.

If `2 max(a,b,c) >= a+b+c`, the triangle is right, obtuse or degenerate. Its
longest-side diameter disk contains the third point, and the squared radius is
`max(a,b,c)/4`. Indeed, writing the farthest endpoints as p,q and the other point
as z, this inequality is equivalent to `(z-p) dot (z-q) <= 0`, exactly the
condition that z lies in that diameter disk. No disk containing p and q can have
a smaller radius.

Otherwise the triangle is acute and D is nonzero. A smallest enclosing disk
exists. If its boundary has one support point, its radius is zero. If it has two
support points, they must be opposite ends of a diameter: otherwise shifting its
center slightly toward their chord reduces the maximal distance. This is
impossible for an acute triangle, since its third vertex lies outside every
side-diameter disk. Hence all three points support the smallest disk, which is
their circumcircle. Its squared radius is `a*b*c/(4*D^2)`, by the area formula
`R = (side1*side2*side3)/(4*area)` and `area=abs(D)/2`.

Every expression is rational. No square root or numerical tolerance is needed
to decide intersection: compare this squared radius with `r^2`, strictly for
open disks and non-strictly for closed disks. For more than three centers, the
maximum of these radii over triples is the enclosing radius, by planar Helly's
theorem. The two-point and empty-family cases are handled separately.

`_mec_radius` still returns a rounded scalar for callers that need a displayed
radius. Its decimal square root is not used by either membership predicate.
`nerve_of_disks` caches exact center coordinates and uses the same predicates
for both edges and triangles, so approximate edge tests cannot contradict exact
triangle tests. It requires finite planar coordinates and a finite positive
radius, as appropriate for its open-disk nerve.

This correction concerns the planar equal-disk oracle. It does not make
geodesic footprints a good cover, establish target inclusion from Betti numbers,
or turn geometric coverage into a clinical outcome guarantee. Existing author
attribution and cached experimental data are preserved.
