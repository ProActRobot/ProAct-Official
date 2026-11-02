# Bundled dependencies

`unitree_sdk2py/` contains the retained Unitree SDK used by physical G1 control.
The robot adapter adds this directory to its import path. CycloneDDS and the
robot network interface still need deployment-specific configuration; see the
[client deployment notes](../client/README.md).

The patched SDK and its platform-specific CRC libraries are preserved. License
and upstream attribution are in [THIRD_PARTY_NOTICES.md](../../docs/THIRD_PARTY_NOTICES.md).

PyMO remains in `src/motion/data_process/pymo/` to preserve the module names stored
in historical preprocessing files. Its location is intentional.
