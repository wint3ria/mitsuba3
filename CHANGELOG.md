# Eradiate Mitsuba — Changelog

Similar to the upstream Mitsuba 3 codebase, this fork does not strictly follow
the [Semantic Versioning](https://semver.org/) convention. However, we also
strive to document breaking API changes in the release notes below.

---

## v0.6.0 (upcoming release)

### New features

- **`eoheterogeneous` medium**: A new heterogeneous that replicates and extends
  the upstream `heterogeneous` medium. Introduces a separate bounding box
  parametrization that allows the medium to have a different bounding box to
  the underlying `sigma_t` volume bbox. In turns allows to exploit the wrap
  mechanism of the volume and implement periodic boundaries
  (PR [#32](https://github.com/eradiate/mitsuba3/pull/32)).

### Bug fixes

- **`multiphase` phase**: Guard the plugin from evaluating phase function
  with a weight equal to zero at the interaction point. Handle the case where
  all phase weights are zero at the interaction point. Use the blendphase
  parameter format convention.

### Improvements

- Improve the *DDIS* (Detector Directional Importance Sampling) variance
  reduction technique by generating DDIS phase functions, envelopes of the
  max phase function used in a medium. Implemented by `piecewise` and
  `eoheterogeneous` media
  (PR [#32](https://github.com/eradiate/mitsuba3/pull/32)).
- Implement prediction-based splitting (PBS) and N-tuple local estimate (NLE)
  variance reduction methods, completing work on VROOM method support in the
  `eovolpath` integrator
  (PR [#33](https://github.com/eradiate/mitsuba3/pull/33)).
- Add an experimental implementation for periodic boundaries of extremum grid
  structures
  (PR [#34](https://github.com/eradiate/mitsuba3/pull/34)).
- Create a new custom scene traversal function which traverses all parents of
  a node, not just the first one. Combine it with Eradiate's traversal function
  (PR [#37](https://github.com/eradiate/mitsuba3/pull/37)).

## v0.5.0 (9th June 2026)

This release is the first to feature a full list of changes. It also contains
significant upgrades on our specific tooling and documentation, including a new,
Eradiate-specific docs site deployed to Read The Docs.

We introduce a first, experimental implementation of variance reduction methods
targeted toward cloudy scenes. All of these changes are subject to change in the
few next versions.

### New features

- **`eovolpath` integrator**: A new volumetric path tracer that replicates and
  extends the upstream `volpath` integrator. Introduces the *DDIS* (Detector
  Directional Importance Sampling) method for improved variance reduction in
  participating media, as well as residual ratio tracking for transmittance
  estimation during emitter sampling.
  (PR [#25](https://github.com/eradiate/mitsuba3/pull/25),
  [#27](https://github.com/eradiate/mitsuba3/pull/27))

- **Extremum plugin family**: This new plugin family that provides
  majorant/minorant bounds for a medium, enabling distance sampling using local
  majorants rather than a global majorant. Three implementations are provided:

  - `extremum_global` — spatially invariant extremum.
  - `extremum_grid` — grid-based extremum tracking.
  - `extremum_spherical` — spherical-coordinates-based extremum tracking.

  (PR [#24](https://github.com/eradiate/mitsuba3/pull/24))

- **Stokes-moment integrator**: This new integrator plugin combines the `moment`
  and `stokes` integrators into a single implementation which retrieves both the
  second moment and Stokes vector components of radiance samples simultaneously.
  This addresses a limitation that would constrain the nesting `moment` and
  `stokes` to a configuration that would not allow getting the second moment of
  all Stokes vector components (only the intensity).
  (PR [#29](https://github.com/eradiate/mitsuba3/pull/29))

### Improvements

- **`Volume` interface**: Added a `min()` method to the `Volume` interface,
  propagated to `VolumeGrid` and `Texture`, as required by the extremum plugins.
  (PR [#27](https://github.com/eradiate/mitsuba3/pull/27))

- **`Medium` analytical transmittance API**: Renamed `sample_interaction_real`
  to `sample_interaction_analytical` and `eval_transmittance_pdf_real` to
  `eval_analytical_transmittance`, which now return the full unpolarized
  spectrum. Reworked the `piecewise` medium to compute transmittance and PDF
  correctly in spectral mode, with new spectral unit tests.
  (PR [#31](https://github.com/eradiate/mitsuba3/pull/31))

### Bug fixes

- **`tabphase_polarized`**: Fixed out-of-sync node handling that could produce
  incorrect results for polarized tabulated phase functions.
  (PR [#26](https://github.com/eradiate/mitsuba3/pull/26))

- **`piecewise_volpath`**: Fixed the integrator to work with non-spectral media.
  (PR [#31](https://github.com/eradiate/mitsuba3/pull/31))
