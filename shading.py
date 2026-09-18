"""
shading.py -- surface shading for the cortical figures.

The earlier figures used a flat grey with a single Lambertian term, which
renders the hemisphere as a smooth blob: the folds are visible only through
the silhouette.  Three changes put the anatomy back.

  1. Sulcal depth.  The FreeSurfer reconstruction carries a per-vertex
     sulcal-depth field (negative on gyral crowns, positive in fundi).
     Binarising it the way FreeSurfer's own curvature display does, and
     using it to set the base tone, makes gyri light and sulci dark, which
     is what the eye reads as a folded cortex rather than a grey surface.
  2. A specular term.  A Blinn-Phong highlight on top of the diffuse term
     gives the pial surface the slight sheen a fixed brain has, and, more
     usefully, picks out the curvature of each gyral crown.
  3. A depth cue.  Faces further from the camera are blended slightly
     toward the background, which separates the near bank of a sulcus from
     the far one.

The functions take triangle arrays and return per-face RGB, so they drop
into the existing Poly3DCollection calls unchanged.
"""
import numpy as np

# light from the upper left front, the convention of every anatomical figure
LIGHT = np.array([-0.55, 0.35, 0.75]); LIGHT /= np.linalg.norm(LIGHT)
VIEW = np.array([-1.0, 0.0, 0.12]); VIEW /= np.linalg.norm(VIEW)

GYRUS = np.array([0.86, 0.83, 0.81])     # warm light grey, gyral crown
SULCUS = np.array([0.40, 0.385, 0.40])   # cool dark grey, sulcal fundus


def face_normals(pos, tri):
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    n = n / np.maximum(nn, 1e-12)[:, None]
    # the mesh has consistent winding, so orientation is fixed once for the
    # whole patch, not per face: flipping face by face against the patch
    # centroid would invert every triangle inside a sulcus.
    ctr = pos[tri].mean(axis=1) - pos[tri].reshape(-1, 3).mean(axis=0)
    if float((n * ctr).sum(axis=1).mean()) < 0.0:
        n = -n
    return n


def curvature_tone(sulc, tri, sharpness=3.2):
    """per-face base colour from binarised sulcal depth"""
    s = sulc[tri].mean(axis=1)
    t = 1.0 / (1.0 + np.exp(-sharpness * s))          # 0 on crowns, 1 in fundi
    return GYRUS[None, :] * (1 - t)[:, None] + SULCUS[None, :] * t[:, None]


def shade_surface(pos, tri, sulc=None, base=None, ambient=0.42,
                  specular=0.28, shininess=24.0, depth_cue=0.18,
                  light=LIGHT, view=VIEW, bg=np.array([1.0, 1.0, 1.0])):
    """
    Diffuse + specular shading of a triangle set, with an optional
    sulcal-depth base tone and a distance-to-camera cue.
    """
    n = face_normals(pos, tri)
    if base is None:
        base = curvature_tone(sulc, tri) if sulc is not None else np.tile(GYRUS, (len(tri), 1))
    base = np.asarray(base, float)
    if base.ndim == 1:
        base = np.tile(base, (len(tri), 1))

    lam = np.clip(n @ light, 0.0, 1.0)
    rgb = base * (ambient + (1.0 - ambient) * lam)[:, None]

    if specular > 0.0:
        h = light + view
        h = h / np.linalg.norm(h)
        spec = np.clip(n @ h, 0.0, 1.0) ** shininess
        rgb = rgb + specular * spec[:, None]

    if depth_cue > 0.0:
        d = pos[tri].mean(axis=1) @ view
        rng = float(d.max() - d.min())
        if rng > 1e-9:
            far = (d - d.min()) / rng             # 1 = furthest from camera
            w = depth_cue * (1.0 - far)
            rgb = rgb * (1.0 - w)[:, None] + bg[None, :] * w[:, None]

    return np.clip(rgb, 0.0, 1.0)


def shade_overlay(pos, tri, rgb_flat, ambient=0.52, specular=0.14,
                  shininess=20.0, light=LIGHT, view=VIEW):
    """
    Shading for coloured overlays (multiplicity, regions): keep the colour
    readable, so more ambient and less specular than the bare surface.
    """
    n = face_normals(pos, tri)
    lam = np.clip(n @ light, 0.0, 1.0)
    rgb = np.asarray(rgb_flat, float)[:, :3] * (ambient + (1.0 - ambient) * lam)[:, None]
    if specular > 0.0:
        h = light + view; h = h / np.linalg.norm(h)
        rgb = rgb + specular * (np.clip(n @ h, 0.0, 1.0) ** shininess)[:, None]
    return np.clip(rgb, 0.0, 1.0)
