.. _sec-eradiate-phase:

Phase functions
===============

The Eradiate kernel provides phase function plugins suited for atmospheric
radiative transfer:

- ``rayleigh_polarized`` — Rayleigh scattering phase matrix with full
  polarimetric support, appropriate for molecular scattering in the atmosphere.
- ``tabphase_irregular`` — tabulated phase function defined on an
  irregular angular grid, suitable for aerosol or cloud particle scattering.
- ``tabphase_polarized`` — tabulated polarimetric phase matrix (Müller
  matrix form) for polarized light transport through anisotropic media.
- ``multiphase`` — linearly combines multiple phase functions, enabling
  composite media (*e.g.* a mixture of molecular and aerosol scattering).
- ``particlefieldphase`` — polarized phase function tabulated on a grid of
  effective radius and effective variance, read from two spatial fields at
  the interaction point.
