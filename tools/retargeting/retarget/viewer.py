import mujoco
import numpy as np


def update_overlay(viewer, number, sec=0.0):
    viewer.user_scn.ngeom = 0
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LABEL,
        size=np.array([0.2, 0.2, 0.2]),
        pos=np.array([0.0, 0.2, 1.3]),
        mat=np.eye(3).flatten(),
        rgba=np.array([0, 0, 0, 0]),
    )
    geom.label = f"Frame {number}, {sec:.2f}s"
    viewer.user_scn.ngeom += 1
