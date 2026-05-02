#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/volume.h>
#include <mitsuba/render/volumegrid.h>
#include <mitsuba/render/eradiate/phase_utils.h>
#include <iomanip>
#include <sstream>


NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-particlefieldphase:

Particle field phase function (:monosp:`particlefieldphase`)
--------------------------------------------------------------

.. pluginparameters::

 * - r_eff_volume
   - |volume|
   - The medium's effective radius field.
   - |exposed|

 * - v_eff_volume
   - |volume|
   - The medium's effective variance field.
   - |exposed|

 * - r_eff_grid
   - |float array|
   - The regularly-spaced ``r_eff`` values the tabulation grid is defined
     at. Its length is the tabulation grid's node count along the ``r_eff``
     axis.
   - |exposed|

 * - v_eff_grid
   - |float array|
   - The regularly-spaced ``v_eff`` values the tabulation grid is defined
     at. Its length is the tabulation grid's node count along the ``v_eff``
     axis.
   - |exposed|

 * - nodes
   - |float array|
   - The per-entry :math:`\cos\theta` discretizations, concatenated back to
     back (one sub-range per ``(r_eff, v_eff)`` grid entry, in row-major
     ``r_eff * n_v + v_eff`` order) — not a single shared grid or union.
     Each entry's own sub-range must start at -1, end at 1, and be strictly
     increasing; see ``grid_start``/``grid_len``.
   - |exposed|

 * - phase_mueller
   - |float array|
   - Flat, 6-components-per-node Mueller matrix table, laid out the same way
     as :ref:`phase-tabphase_polarized`'s ``m11``/``m12``/``m22``/``m33``/
     ``m34``/``m44`` columns, one row per entry of ``nodes``.
   - |exposed|

 * - grid_start
   - |uint array|
   - Offset of each ``(r_eff, v_eff)`` grid entry's sub-range within the flat
     ``nodes``/``phase_mueller`` arrays.
   - |exposed|

 * - grid_len
   - |uint array|
   - Length of each ``(r_eff, v_eff)`` grid entry's sub-range within the flat
     ``nodes``/``phase_mueller`` arrays.
   - |exposed|

 * - sigma_s_weight
   - |float array|
   - Per-entry scattering-probability weight, used to rescale the bilinear
     blend of the 4 corners cornering a query point.
   - |exposed|

 * - blending_method
   - |string|
   - How the 4 cornering entries are combined when sampling: ``"search"``
     (default) blends the 4 corners' CDFs and inverts the result via
     bisection; ``"stochastic"`` picks one cornering entry at random,
     weighted by its blend weight, and samples it directly; ``"tabulate"``
     merges the 4 corners onto one common grid on the host before sampling
     and is scalar-only (construction throws under any JIT variant).

 * - cdf, norm
   - |float array|
   - Optional, precomputed replacement for the CDF/normalization this plugin
     would otherwise build itself from ``nodes``/``phase_mueller`` at
     construction. Must be provided together. Supplying these bypasses the
     node-layout validation ``nodes``/``phase_mueller`` otherwise go through.

This plugin implements a phase function tabulated on a 2D grid of effective
radius (``r_eff``) and effective variance (``v_eff``), each grid entry
carrying its own tabulated, polarized phase matrix over an irregular
:math:`\cos\theta` discretization. At a given interaction point,
``r_eff_volume``/``v_eff_volume`` are evaluated to find the point's
``(r_eff, v_eff)``, and the 4 grid entries cornering that value are
bilinearly blended (further rescaled by ``sigma_s_weight``) according to
``blending_method``.
*/

template <typename Float, typename Spectrum>
class ParticleFieldPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext, Volume, VolumeGrid)

    using FloatStorage  = DynamicBuffer<Float>;
    using UInt32Storage = DynamicBuffer<UInt32>;
    using Vector6u      = dr::Array<UInt32, 6>;
    using Vector6f      = dr::Array<Float, 6>;

    enum class BlendingMethod { Stochastic, Tabulate, Search };

    static BlendingMethod method_from_string(const std::string &method) {
        if (method == "stochastic")  return BlendingMethod::Stochastic;
        if (method == "tabulate")    return BlendingMethod::Tabulate;
        if (method == "search")      return BlendingMethod::Search;
        Throw("ParticleFieldPhaseFunction: unknown blending_method '%s'", method);
    }

    static std::string method_to_string(BlendingMethod method) {
        switch (method) {
            case BlendingMethod::Stochastic: return "stochastic";
            case BlendingMethod::Tabulate:   return "tabulate";
            case BlendingMethod::Search:     return "search";
            default: Throw("ParticleFieldPhaseFunction: unknown blending method");
        }
    }

    struct BilinearWeights {
        Vector4u idx;
        Vector4f w;
    };

    struct BisectionBounds {
        Vector4u starts, lens;
        Vector4f norm;
        Float lo, hi, target, tol;
        Vector4u lo4_at_hi;
    };

    template <size_t N>
    struct ColumnEntry {
        std::vector<ScalarFloat> x;
        std::array<std::vector<ScalarFloat>, N> y;
        ScalarFloat weight = 0.f;
    };

    template<typename data_t, typename storage_t>
    storage_t load_grid(const Properties &props, const char *name) {
        if (props.type(name) == Properties::Type::Object) {
            ref<Object> obj = props.get<ref<Object>>(name);
            auto *vg = dynamic_cast<VolumeGrid *>(obj.get());
            if (!vg)
                Throw("ParticleFieldPhaseFunction: property \"%s\" must be a VolumeGrid.", name);

            size_t n = (size_t) vg->size().x();
            const ScalarFloat *buf = vg->data();
            storage_t out = dr::zeros<storage_t>(n);
            for (size_t i = 0; i < n; ++i)
                dr::scatter(out, data_t(buf[i]), UInt32(i));
            return out;
        }

        return props.get_any<storage_t>(name);
    }

    explicit ParticleFieldPhaseFunction(const Properties &props) : Base(props) {
        m_r_eff_volume = props.get_volume<Volume>("r_eff_volume");
        m_v_eff_volume = props.get_volume<Volume>("v_eff_volume");

        m_r_eff_grid    = load_grid<Float, FloatStorage>(props, "r_eff_grid");
        m_v_eff_grid    = load_grid<Float, FloatStorage>(props, "v_eff_grid");
        m_n_r = (int) dr::width(m_r_eff_grid);
        m_n_v = (int) dr::width(m_v_eff_grid);
        m_nodes         = props.get_any<FloatStorage>("nodes");
        m_mueller       = props.get_any<FloatStorage>("phase_mueller");
        m_grid_start    = load_grid<UInt32, UInt32Storage>(props, "grid_start");
        m_grid_len      = load_grid<UInt32, UInt32Storage>(props, "grid_len");
        m_sigma_s_weight = load_grid<Float, FloatStorage>(props, "sigma_s_weight");

        m_blending_method = method_from_string(props.get<std::string>("blending_method", "search"));

        if (m_blending_method == BlendingMethod::Tabulate && dr::is_jit_v<Float>)
            Throw("ParticleFieldPhaseFunction: blending_method \"tabulate\" builds its "
                  "merged CDF on the host from a single lane (dr::slice(u, 0)) and "
                  "is only correct for the scalar variant; use \"search\" with the "
                  "llvm/cuda variants.");

        if (dr::width(m_mueller) != dr::width(m_nodes) * 6u)
            Throw("ParticleFieldPhaseFunction: phase_mueller must have 6 * len(nodes) elements");
        if (dr::width(m_sigma_s_weight) < (size_t)(m_n_r * m_n_v))
            Throw("ParticleFieldPhaseFunction: sigma_s_weight must have at least n_r * n_v = %d elements", m_n_r * m_n_v);

        if (props.has_property("cdf") && props.has_property("norm")) {
            m_cdf  = props.get_any<FloatStorage>("cdf");
            m_norm = props.get_any<FloatStorage>("norm");
        } else if (props.has_property("cdf") || props.has_property("norm")) {
            Throw("ParticleFieldPhaseFunction: cdf and norm must be provided together");
        } else {
            precompute_cdf();
        }

        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.clear();
        m_components.push_back(m_flags);
    }

private:
    /// Validate the node layout, then build the per-entry CDFs and their
    /// normalization factors (\ref m_cdf, \ref m_norm), vectorized over one
    /// lane per (r_eff, v_eff) entry.
    void precompute_cdf() {
        // Validate the node layout (scalar/host-side: needs concrete
        // per-entry values to produce meaningful Throw() messages, so this
        // cannot run as a vectorized JIT loop).
        size_t total_pts = dr::width(m_nodes);
        size_t n = (size_t)(m_n_r * m_n_v);

        size_t expected_start = 0;

        for (size_t i = 0; i < n; ++i) {
            size_t start = dr::slice(m_grid_start, i);
            size_t len   = dr::slice(m_grid_len,   i);

            if (len < 2)
                Throw("ParticleFieldPhaseFunction: entry %zu has grid_len=%zu (need >= 2).", i, len);
            if (start + len > total_pts)
                Throw("ParticleFieldPhaseFunction: entry %zu slice [%zu, %zu) exceeds nodes size %zu.",
                      i, start, start + len, total_pts);
            if (start != expected_start)
                Throw("ParticleFieldPhaseFunction: entry %zu grid_start=%zu, expected %zu "
                      "(start/len must form a contiguous concatenation; is 'nodes' a "
                      "global union instead of the per-entry concatenation?).",
                      i, start, expected_start);
            expected_start = start + len;

            bool invalid_s = dr::slice(m_nodes, start) != -1.f;
            bool invalid_e = dr::slice(m_nodes, start + len - 1) != 1.f;
            if (invalid_s || invalid_e)
                Throw(
                    "ParticleFieldPhaseFunction: invalid phase bound for entry %zu. Start is 1.0: %b, end is 1.0: %b",
                    i, invalid_s, invalid_e
                );

            for (size_t k = 1; k < len; ++k) {
                ScalarFloat prev = dr::slice(m_nodes, start + k - 1);
                ScalarFloat cur  = dr::slice(m_nodes, start + k);
                if (cur <= prev)
                    Throw("ParticleFieldPhaseFunction: entry %zu not strictly increasing: "
                          "nodes[%zu]=%.17g >= nodes[%zu]=%.17g (%s).",
                          i, start + k - 1, prev, start + k, cur,
                          cur == prev ? "duplicate" : "decreasing - descending cos(theta)?");
            }
        }

        if (expected_start != total_pts)
            Throw("ParticleFieldPhaseFunction: entries cover %zu nodes, 'nodes' has %zu.",
                  expected_start, total_pts);

        // Vectorized CDF construction: two passes on all entries, since the rescale factor
        // for an entry isn't known until its own accumulation has fully
        // finished.
        using MaskStorage = dr::mask_t<FloatStorage>;

        m_cdf  = dr::zeros<FloatStorage>(total_pts);
        m_norm = dr::zeros<FloatStorage>(n);

        UInt32Storage entry = dr::arange<UInt32Storage>((uint32_t) n);
        UInt32Storage start = dr::gather<UInt32Storage>(m_grid_start, entry);
        UInt32Storage len   = dr::gather<UInt32Storage>(m_grid_len,   entry);

        dr::scatter(m_cdf, dr::zeros<FloatStorage>(n), start);

        UInt32Storage k = dr::zeros<UInt32Storage>(n);
        FloatStorage running = dr::zeros<FloatStorage>(n);
        MaskStorage active_loop = k < (len - 1u);

        std::tie(k, running, active_loop) = dr::while_loop(
            std::make_tuple(k, running, active_loop),
            [](const UInt32Storage &, const FloatStorage &, const MaskStorage &active_loop) {
                return dr::any(active_loop);
            },
            [this, start, len](UInt32Storage &k, FloatStorage &running, MaskStorage &active_loop) {
                UInt32Storage idx0 = start + k, idx1 = idx0 + 1u;

                FloatStorage x0 = dr::gather<FloatStorage>(m_nodes,   idx0,      active_loop);
                FloatStorage x1 = dr::gather<FloatStorage>(m_nodes,   idx1,      active_loop);
                FloatStorage y0 = dr::gather<FloatStorage>(m_mueller, idx0 * 6u, active_loop);
                FloatStorage y1 = dr::gather<FloatStorage>(m_mueller, idx1 * 6u, active_loop);

                running = dr::select(active_loop,
                                      running + 0.5f * (y0 + y1) * (x1 - x0),
                                      running);
                dr::scatter(m_cdf, running, idx1, active_loop);

                k += 1u;
                active_loop &= (k < (len - 1u));

                // See Dr.Jit's docs, "Evaluation" > "Caching, continued".
                dr::eval(k, running, active_loop, m_cdf);
            },
            "ParticleFieldPhaseFunction precompute_cdf (accumulate)");

        // `running` now holds each entry's raw (unnormalized) total integral
        MaskStorage has_mass = running > 0.f;
        FloatStorage norm_per_entry = dr::select(has_mass, dr::rcp(running), FloatStorage(0.f));
        dr::scatter(m_norm, norm_per_entry, entry);

        UInt32Storage k2 = dr::zeros<UInt32Storage>(n);
        MaskStorage active_loop2 = has_mass && (k2 < len);

        std::tie(k2, active_loop2) = dr::while_loop(
            std::make_tuple(k2, active_loop2),
            [](const UInt32Storage &, const MaskStorage &active_loop2) {
                return dr::any(active_loop2);
            },
            [this, start, len, norm_per_entry](UInt32Storage &k2, MaskStorage &active_loop2) {
                UInt32Storage idx = start + k2;
                FloatStorage val = dr::gather<FloatStorage>(m_cdf, idx, active_loop2);
                dr::scatter(m_cdf, val * norm_per_entry, idx, active_loop2);

                k2 += 1u;
                active_loop2 &= (k2 < len);

                // See Dr.Jit's docs, "Evaluation" > "Caching, continued".
                dr::eval(k2, active_loop2, m_cdf);
            },
            "ParticleFieldPhaseFunction precompute_cdf (rescale)");
    }

    /// Locate the 4 entry indices in the r_eff, v_eff grids cornering the
    /// (r_eff, v_eff) value at the current interaction point, and compute
    /// the associated 4 interpolation weights. Assumes the r_eff and v_eff
    /// grids are regular.
    BilinearWeights get_bilinear_weights(Float r_eff, Float v_eff,
                                         Float r_min, Float r_max,
                                         Float v_min, Float v_max,
                                         Mask active) const {
        // r_eff dimension
        UInt32 ir = 0u;
        Float tr = Float(0.f);
        if (m_n_r > 1) {
            Float dr_   = (r_max - r_min) / Float(m_n_r - 1);
            ir  = dr::clip(UInt32((r_eff - r_min) / dr_), 0u, UInt32(m_n_r - 2));
            Float r0 = dr::fmadd(Float(ir),      dr_, r_min);
            Float r1 = dr::fmadd(Float(ir + 1u), dr_, r_min);
            Float dri = r1 - r0;
            Float r_scale = dr::maximum(dr::abs(r_max - r_min), Float(1.f));
            tr = dr::select(dr::abs(dri) > dr::Epsilon<Float> * r_scale, (r_eff - r0) / dri, Float(0.f));
        }

        // v_eff dimension
        UInt32 iv = 0u;
        Float tv = Float(0.f);
        if (m_n_v > 1) {
            Float dv_   = (v_max - v_min) / Float(m_n_v - 1);
            iv  = dr::clip(UInt32((v_eff - v_min) / dv_), 0u, UInt32(m_n_v - 2));
            Float v0 = dr::fmadd(Float(iv),      dv_, v_min);
            Float v1 = dr::fmadd(Float(iv + 1u), dv_, v_min);
            Float dvi = v1 - v0;
            Float v_scale = dr::maximum(dr::abs(v_max - v_min), Float(1.f));
            tv = dr::select(dr::abs(dvi) > dr::Epsilon<Float> * v_scale, (v_eff - v0) / dvi, Float(0.f));
        }

        // Calculate the geometric weight for each corner and store them in bw
        Float _tr = 1.f - tr, _tv = 1.f - tv;
        UInt32 n_v   = m_n_v;
        UInt32 ir_hi = m_n_r > 1 ? ir + 1u : ir;
        UInt32 iv_hi = m_n_v > 1 ? iv + 1u : iv;
        BilinearWeights bw;
        bw.idx = Vector4u(ir    * n_v + iv,
                          ir    * n_v + iv_hi,
                          ir_hi * n_v + iv,
                          ir_hi * n_v + iv_hi);
        bw.w   = Vector4f(_tr * _tv, _tr * tv, tr * _tv, tr * tv);

        // Rescale each corner according to its scattering probability
        Vector4f sca  = dr::gather<Vector4f>(m_sigma_s_weight, bw.idx, active);
        bw.w         *= sca;
        Float sca_sum = dr::sum(bw.w);
        bw.w          = dr::select(sca_sum > dr::Epsilon<Float>, bw.w / sca_sum, bw.w);

        return bw;
    }

private:
    // =========================================================================
    //  Helper methods
    // =========================================================================

    static constexpr Vector6u k_mueller_offsets() { return Vector6u(0u, 1u, 2u, 3u, 4u, 5u); }

    /// Gather the 6 Mueller matrix components stored at `point`
    Vector6f gather_mueller(UInt32 point, Mask active) const {
        return dr::gather<Vector6f>(m_mueller, point * 6u + k_mueller_offsets(), active);
    }

private:
    // =========================================================================
    //  Search methods
    // =========================================================================

    /// Perform a binary search on 4 contiguous sub-sections of `arr`
    /// independently. The 4 sub-sections are maintained using 4-value
    /// vectors of indices (`Vector4u`) and values (`Vector4f`).
    Vector4u ragged_searchsorted4(const FloatStorage &arr,
                                   Vector4u start, Vector4u len,
                                   Float query, Mask active) const {
        Vector4u lo = start;
        Vector4u hi = start + len;
        dr::mask_t<Vector4u> running = dr::mask_t<Vector4u>(active) && (lo < hi);

        std::tie(lo, hi, running) = dr::while_loop(
            std::make_tuple(lo, hi, running),
            [](const Vector4u &, const Vector4u &,
               const dr::mask_t<Vector4u> &running) {
                return dr::any(running);
            },
            [&arr, query](Vector4u &lo, Vector4u &hi,
                          dr::mask_t<Vector4u> &running) {
                Vector4u mid = (lo + hi) >> 1u;
                Vector4f val = dr::gather<Vector4f>(arr, mid, running);
                dr::mask_t<Vector4u> go_right = running && (val < Vector4f(query));
                lo = dr::select(go_right, mid + 1u, lo);
                hi = dr::select(running && !go_right, mid, hi);
                running = running && (lo < hi);
            },
            "ParticleFieldPhaseFunction ragged_searchsorted4");

        return dr::clip(lo, start + 1u, start + len - 1u) - 1u;
    }

    /// Perform a searchsorted operation on a single sub-section of `arr`.
    /// Offloaded to dr::binary_search.
    UInt32 searchsorted1(const FloatStorage &arr, UInt32 start, UInt32 len,
                         Float query, Mask active) const {
        UInt32 lo = dr::binary_search<UInt32>(
            start, start + len,
            [&arr, active, query](UInt32 i) DRJIT_INLINE_LAMBDA {
                return active && (dr::gather<Float>(arr, i, active) < query);
            });
        UInt32 result = dr::clip(lo, start + 1u, start + len - 1u) - 1u;
        return result;
    }

    /// Locate sample `u` in a CDF sub-section and perform the quadratic
    /// inversion of its cosine.
    Float sample_cos_theta_entry(Float u, UInt32 start, UInt32 len,
                                 Float norm, Mask active) const {
        UInt32 lo = searchsorted1(m_cdf, start, len, u, active);
        UInt32 hi = lo + 1u;

        Float x0 = dr::gather<Float>(m_nodes,   lo,      active);
        Float x1 = dr::gather<Float>(m_nodes,   hi,      active);
        Float c0 = dr::gather<Float>(m_cdf,     lo,      active);
        Float y0 = dr::gather<Float>(m_mueller, lo * 6u, active) * norm;
        Float y1 = dr::gather<Float>(m_mueller, hi * 6u, active) * norm;

        Float h = x1 - x0;
        Float s = (u - c0) / h;

        // Numerically stable form of the quadratic root. The classical
        // form (y0 - sqrt(y0^2 + 2*s*(y1-y0))) / (y0 - y1) subtracts two
        // nearly-equal values whenever y0 ~ y1. This case is likely when 
        // y0 and y1 are taken from a smoothly-varying phase function.
        Float disc = dr::fmadd(y0, y0, 2.f*s*(y1-y0));
        Float t    = 2.f*s * dr::rcp(y0 + dr::safe_sqrt(disc));
        return dr::fmadd(dr::clip(t, 0.f, 1.f), h, x0);
    }


    /// Blend the 4 cornering CDFs at the given cosine x.
    Float eval_blended_cdf_at(Float x, const BilinearWeights &bw, Vector4u lo4,
                               Vector4f norm, Mask active) const {
        Vector4u hi4   = lo4 + 1u;
        Vector4f x0_4  = dr::gather<Vector4f>(m_nodes,   lo4,      active);
        Vector4f x1_4  = dr::gather<Vector4f>(m_nodes,   hi4,      active);
        Vector4f y0_4  = dr::gather<Vector4f>(m_mueller, lo4 * 6u, active);
        Vector4f y1_4  = dr::gather<Vector4f>(m_mueller, hi4 * 6u, active);
        Vector4f cdf0  = dr::gather<Vector4f>(m_cdf,     lo4,      active);
        Vector4f dx    = x1_4 - x0_4;
        Vector4f t4    = dr::clip(dr::select(dx > dr::Epsilon<Float>,
                                             (Vector4f(x) - x0_4) / dx, 0.f), 0.f, 1.f);
        Vector4f cdf_c = cdf0 + norm * dx * t4 * dr::fmadd(0.5f * t4, y1_4 - y0_4, y0_4);
        return dr::dot(bw.w, cdf_c);
    }

    /// Locate cosine x in the 4 cornering CDFs and blend them using
    /// \ref eval_blended_cdf_at.
    Float eval_blended_cdf(Float x, const BilinearWeights &bw,
                            Vector4u starts, Vector4u lens, Vector4f norm, Mask active) const {
        Vector4u lo4 = ragged_searchsorted4(m_nodes, starts, lens, x, active);
        return eval_blended_cdf_at(x, bw, lo4, norm, active);
    }

    /// Define the CDF bisection boundaries for a given set of 4 cornering
    /// CDFs and their associated weights.
    BisectionBounds
    bisection_bounds(Float u, const BilinearWeights &bw, Mask active) const {
        Vector4u starts = dr::gather<Vector4u>(m_grid_start, bw.idx, active);
        Vector4u lens   = dr::gather<Vector4u>(m_grid_len,   bw.idx, active);
        Vector4f norm   = dr::gather<Vector4f>(m_norm,       bw.idx, active);

        Float lo = dr::max(dr::gather<Vector4f>(m_nodes, starts,             active));
        Float hi = dr::min(dr::gather<Vector4f>(m_nodes, starts + lens - 1u, active));

        Vector4u lo4_at_hi = ragged_searchsorted4(m_nodes, starts, lens, hi, active);
        Float target = u * eval_blended_cdf_at(hi, bw, lo4_at_hi, norm, active);
        Float tol    = 16 * dr::Epsilon<Float> * dr::maximum(dr::abs(hi), Float(1.f));

        return { starts, lens, norm, lo, hi, target, tol, lo4_at_hi };
    }

private:

    // =========================================================================
    //  Blend methods implementations
    // =========================================================================

    /// Merge the 4 cornering phase-value discretizations onto a common grid,
    /// then build and invert a CDF from the blend.
    Float sample_cos_theta_tabulate(Float u, const BilinearWeights &bw, Mask active) const {
        ScalarFloat lo_bound = 0.f, hi_bound = 0.f;
        auto entries = gather_column_entries<1>(bw, active, lo_bound, hi_bound);

        std::vector<ScalarFloat> merged;
        std::array<std::vector<ScalarFloat>, 1> Y_cols;
        merge_and_interpolate<1>(entries, true, lo_bound, hi_bound, merged, Y_cols);

        const std::vector<ScalarFloat> &Y = Y_cols[0];
        size_t n = merged.size();
        std::vector<ScalarFloat> cdf(n, 0.0);
        for (size_t k = 1; k < n; ++k)
            cdf[k] = cdf[k - 1] + 0.5 * (Y[k - 1] + Y[k]) * (merged[k] - merged[k - 1]);

        ScalarFloat total = cdf.back();
        if (n < 2 || total <= 0.f)
            return Float(lo_bound);

        ScalarFloat target = dr::slice(u, 0) * total;

        size_t hi_idx = (size_t)(std::lower_bound(cdf.begin(), cdf.end(), target) - cdf.begin());
        hi_idx = std::min(std::max(hi_idx, (size_t) 1), n - 1);
        size_t lo_idx = hi_idx - 1;

        ScalarFloat x0 = merged[lo_idx], x1 = merged[hi_idx];
        ScalarFloat y0 = Y[lo_idx],      y1 = Y[hi_idx];
        ScalarFloat c0 = cdf[lo_idx];

        ScalarFloat h = x1 - x0;
        ScalarFloat s = h > 0.f ? (target - c0) / h : 0.f;

        // Same rationalized form as sample_cos_theta_entry()
        ScalarFloat disc = std::max(ScalarFloat(0.f), y0 * y0 + ScalarFloat(2.f) * s * (y1 - y0));
        ScalarFloat t = ScalarFloat(2.f) * s / (y0 + std::sqrt(disc));
        t = std::min(std::max(t, ScalarFloat(0.f)), ScalarFloat(1.f));

        ScalarFloat cos_t = x0 + t * h;
        return Float(cos_t);
    }

    /// Bisect the 4 cornering CDF entries for the sample until they have all
    /// converged on their respective segment. Invert the cosine using the
    /// quadratic formula on the blended CDF segment.
    std::tuple<Float, Vector4u, Vector4u, Vector4u, Mask>
    sample_cos_theta_search(Float u, const BilinearWeights &bw, Mask active) const {
        auto [starts, lens, norm, lo, hi, target, tol, lo4_at_hi] = bisection_bounds(u, bw, active);

        Vector4u lo4_at_lo = ragged_searchsorted4(m_nodes, starts, lens, lo, active);

        UInt32 iter_count = 0u;
        Mask active_loop = active && !dr::all(lo4_at_lo == lo4_at_hi) && ((hi - lo) > tol);

        // We want to find the angle whose blended CDF equals the sample.
        // The blended CDF where we search from is not expressed anywhere, so
        // we bisect it to find the cosine segment in each corner's tabulated
        // CDF where the sample is expressed first.
        //
        // This loop maintains a search window on the angle axis. At each
        // step we compare the blended CDF at the window midpoint to the 
        // sample. The search window on cosine is then updated accordingly.
        //
        // When it has shrunk enough that each corner found a matching cosine
        // segment at the low and high bounds of the window, the loop stops
        // and these segments are the ones to blend and invert.
        std::tie(lo, hi, lo4_at_lo, lo4_at_hi, iter_count, active_loop) = dr::while_loop(
            std::make_tuple(lo, hi, lo4_at_lo, lo4_at_hi, iter_count, active_loop),
            [](const Float &, const Float &, const Vector4u &, const Vector4u &,
               const UInt32 &, const Mask &active_loop) {
                return active_loop;
            },
            [this, bw, starts, lens, norm, target, tol](
                Float &lo, Float &hi, Vector4u &lo4_at_lo, Vector4u &lo4_at_hi,
                UInt32 &iter_count, Mask &active_loop) {
                Float mid = 0.5f * (lo + hi);
                // Look for the segment in each corner containing mid
                Vector4u lo4_at_mid =
                    ragged_searchsorted4(m_nodes, starts, lens, mid, active_loop);
                // Evaluate the blended CDF according to bw, and compare it to
                // the sample
                Mask go_right = active_loop &&
                    (eval_blended_cdf_at(mid, bw, lo4_at_mid, norm, active_loop) < target);

                Mask update_lo = active_loop && go_right;
                Mask update_hi = active_loop && !go_right;
                lo        = dr::select(update_lo, mid, lo);
                hi        = dr::select(update_hi, mid, hi);
                lo4_at_lo = dr::select(update_lo, lo4_at_mid, lo4_at_lo);
                lo4_at_hi = dr::select(update_hi, lo4_at_mid, lo4_at_hi);

                iter_count += 1u;
                Mask stable = dr::all(lo4_at_lo == lo4_at_hi);
                active_loop &= (iter_count < 30u) && !stable && ((hi - lo) > tol);
            },
            "ParticleFieldPhaseFunction search bisection");

        // Now perform the blend and inversion
        Mask stable = active && dr::all(lo4_at_lo == lo4_at_hi);

        Vector4u lo4 = lo4_at_lo;
        Vector4u hi4 = lo4 + 1u;
        Vector4f x0_4   = dr::gather<Vector4f>(m_nodes,   lo4,      active);
        Vector4f x1_4   = dr::gather<Vector4f>(m_nodes,   hi4,      active);
        Vector4f y0_4   = dr::gather<Vector4f>(m_mueller, lo4 * 6u, active);
        Vector4f y1_4   = dr::gather<Vector4f>(m_mueller, hi4 * 6u, active);
        Vector4f cdf0_4 = dr::gather<Vector4f>(m_cdf,     lo4,      active);
        Vector4f dx4    = x1_4 - x0_4;

        Vector4f wn = bw.w * norm;
        Vector4f k2 = dr::select(dx4 > dr::Epsilon<Float>, 0.5f * wn * (y1_4 - y0_4) / dx4, Vector4f(0.f));
        Vector4f k1 = wn * y0_4;

        Float A = dr::sum(k2);
        Float B = dr::sum(k1 - 2.f * k2 * x0_4);
        Float C = dr::sum(bw.w * cdf0_4 - k1 * x0_4 + k2 * x0_4 * x0_4) - target;

        // Numerically stable quadratic formula: the naïve (-B ± sqrt(disc)) / (2*A)
        // cancels whenever sqrt(disc) and B are close in value.
        // Rewriting via q = -0.5*(B + sign(B)*sqrt(disc)) and roots {q/A, C/q} this.
        Float disc  = dr::maximum(dr::fmadd(B, B, -4.f * A * C), 0.f);
        Float sq    = dr::sqrt(disc);
        Float q     = -0.5f * (B + dr::sign(B) * sq);
        Float coeff_scale = dr::maximum(dr::maximum(dr::abs(A), dr::abs(B)), dr::maximum(dr::abs(C), Float(1.f)));
        Float eps_scaled  = dr::Epsilon<Float> * coeff_scale;
        Float root1 = dr::select(dr::abs(A) > eps_scaled, q / A, 0.f);
        Float root2 = dr::select(dr::abs(q) > eps_scaled, C / q, 0.f);
        Float root_lin = dr::select(dr::abs(B) > eps_scaled, -C / B, 0.5f * (lo + hi));

        // Pick the root that needs the least clipping to land in [lo, hi],
        // rather than a naïve inclusion test: the valid root can round
        // outside [lo, hi] in single precision.
        Float root1_clipped = dr::clip(root1, lo, hi);
        Float root2_clipped = dr::clip(root2, lo, hi);
        Mask  root1_closer   = dr::abs(root1 - root1_clipped) <= dr::abs(root2 - root2_clipped);
        Float x_quad  = dr::select(root1_closer, root1_clipped, root2_clipped);
        Float cos_t   = dr::select(dr::abs(A) > eps_scaled, x_quad, root_lin);
        cos_t = dr::select(stable, dr::clip(cos_t, lo, hi), 0.5f * (lo + hi));

        return { cos_t, starts, lens, lo4, stable };
    }


    /// Select one of the 4 cornering CDFs using \c u_select as a weight and
    /// invert it.
    std::tuple<Vector3f, Spectrum, Float>
    sample_stochastic(const PhaseFunctionContext &ctx,
                         const MediumInteraction3f &mei,
                         Float u_select, const Point2f &sample2,
                         const BilinearWeights &bw, Mask active) const {
        Vector4f cumw(bw.w[0],
                      bw.w[0] + bw.w[1],
                      bw.w[0] + bw.w[1] + bw.w[2],
                      bw.w[0] + bw.w[1] + bw.w[2] + bw.w[3]);

        UInt32 idx = bw.idx[0];
        idx = dr::select(u_select > cumw[0], bw.idx[1], idx);
        idx = dr::select(u_select > cumw[1], bw.idx[2], idx);
        idx = dr::select(u_select > cumw[2], bw.idx[3], idx);

        // Weight of whichever corner ends up selected above (same comparison
        // chain as idx). The corner *choice* stays detached below -- s/l/norm/
        // cos_t never see a gradient from bw.w, matching how dielectric.cpp's
        // reflection/transmission lobe choice is detached to avoid bias. What
        // needs correcting is the returned *weight*: as bw.w varies, the
        // probability of having landed on this corner varies too, which is a
        // score-function (REINFORCE) contribution, not a pathwise one.
        Float w_sel = bw.w[0];
        w_sel = dr::select(u_select > cumw[0], bw.w[1], w_sel);
        w_sel = dr::select(u_select > cumw[1], bw.w[2], w_sel);
        w_sel = dr::select(u_select > cumw[2], bw.w[3], w_sel);

        UInt32 s = dr::gather<UInt32>(m_grid_start, idx, active);
        UInt32 l = dr::gather<UInt32>(m_grid_len,   idx, active);
        Float  norm = dr::gather<Float>(m_norm, idx, active);

        Float cos_t = sample_cos_theta_entry(sample2.x(), s, l, norm, active);
        Float sin_t = dr::safe_sqrt(1.f - cos_t * cos_t);
        auto [sin_phi, cos_phi] = dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());
        Vector3f wo = -mei.to_world(Vector3f(sin_t * cos_phi, sin_t * sin_phi, cos_t));
        wo = dr::select(active, wo, Vector3f(0.f));

        auto [phase_val, pdf] = eval_pdf_flat(ctx, mei, wo, bw, active);

        Spectrum weight = phase_val * dr::rcp(pdf);

        // REINFORCE-style correction for the discrete corner choice above
        // (same dr::replace_grad idiom as dielectric.cpp's lobe selection):
        // numerically a no-op (w_sel / detach(w_sel) == 1), but injects
        // d(log w_sel)/d(theta) into weight's gradient.
        if constexpr (dr::is_diff_v<Float>) {
            if (dr::grad_enabled(w_sel)) {
                Float score = dr::replace_grad(Float(1.f), w_sel / dr::detach(w_sel));
                weight *= score;
            }
        }

        weight = dr::select(active, weight, Spectrum(0.f));
        pdf    = dr::select(active, pdf, Float(0.f));

        return { wo, weight, pdf };
    }

    /// Get the bilinear weights for r_eff, v_eff at the interaction location.
    std::pair<BilinearWeights, Mask>
    get_weights(const MediumInteraction3f &mei, Mask active) const {
        Float r_eff = m_r_eff_volume->eval_1(mei, active);
        Float v_eff = m_v_eff_volume->eval_1(mei, active);

        // These sanity checks can only be done in scalar variants.
        if constexpr (!dr::is_jit_v<Float>) {
            Mask nan_mask = dr::isnan(r_eff) || dr::isnan(v_eff);
            if (dr::any(active && nan_mask)) {
                std::ostringstream oss;
                oss << "ParticleFieldPhaseFunction: NaN r_eff or v_eff at interaction point.\n";
                oss << "  mei.p       = " << mei.p << "\n";
                oss << "  r_eff value = " << r_eff << "\n";
                oss << "  v_eff value = " << v_eff << "\n";
                oss << "  r_eff is NaN: " << dr::isnan(r_eff) << "\n";
                oss << "  v_eff is NaN: " << dr::isnan(v_eff) << "\n";
                oss << "  r_eff_volume to_local:\n" << m_r_eff_volume->bbox() << "\n";
                oss << "  v_eff_volume to_local:\n" << m_v_eff_volume->bbox() << "\n";
                Throw("%s", oss.str());
            }
        }

        Float r_min = 0.f, r_max = 0.f;
        if (m_n_r > 1) {
            r_min = dr::gather<Float>(m_r_eff_grid, UInt32(0u), active);
            r_max = dr::gather<Float>(m_r_eff_grid, UInt32(m_n_r - 1), active);
            if constexpr (!dr::is_jit_v<Float>) {
                if (dr::any(active && (r_eff < r_min || r_eff > r_max)))
                    Throw("ParticleFieldPhaseFunction: r_eff out of dataset range. %s out of [%s; %s]", r_eff, r_min, r_max);
            }
        }

        Float v_min = 0.f, v_max = 0.f;
        if (m_n_v > 1) {
            v_min = dr::gather<Float>(m_v_eff_grid, UInt32(0u), active);
            v_max = dr::gather<Float>(m_v_eff_grid, UInt32(m_n_v - 1), active);
            if constexpr (!dr::is_jit_v<Float>) {
                if (dr::any(active && (v_eff < v_min || v_eff > v_max)))
                    Throw("ParticleFieldPhaseFunction: v_eff out of dataset range.");
            }
        }

        auto weights = get_bilinear_weights(r_eff, v_eff, r_min, r_max, v_min, v_max, active);

        return { weights, active };
    }



    /// Scalar retrieval of the 4 cornering entries' phase values.
    template <size_t N>
    std::array<ColumnEntry<N>, 4>
    gather_column_entries(const BilinearWeights &bw, Mask active,
                           ScalarFloat &lo_bound, ScalarFloat &hi_bound) const {
        Vector4u starts = dr::gather<Vector4u>(m_grid_start, bw.idx, active);
        Vector4u lens   = dr::gather<Vector4u>(m_grid_len,   bw.idx, active);
        Vector4f norms  = dr::gather<Vector4f>(m_norm,       bw.idx, active);

        std::array<ColumnEntry<N>, 4> entries;
        for (size_t i = 0; i < 4; ++i) {
            size_t s = dr::slice(starts[i], 0);
            size_t l = dr::slice(lens[i],   0);

            ColumnEntry<N> &e = entries[i];
            e.weight = dr::slice(bw.w[i], 0) * dr::slice(norms[i], 0);
            e.x.resize(l);
            for (auto &col : e.y)
                col.resize(l);
            for (size_t k = 0; k < l; ++k) {
                e.x[k] = dr::slice(m_nodes, s + k);
                for (size_t c = 0; c < N; ++c)
                    e.y[c][k] = dr::slice(m_mueller, (s + k) * 6u + c);
            }

            ScalarFloat first = e.x.front(), last = e.x.back();
            if (i == 0) { lo_bound = first; hi_bound = last; }
            else        { lo_bound = std::max(lo_bound, first); hi_bound = std::min(hi_bound, last); }
        }
        return entries;
    }

    /// Scalar merge of the 4 cornering entries' nodes, interpolating and
    /// accumulating their phase values onto the merged grid.
    template <size_t N>
    static void merge_and_interpolate(const std::array<ColumnEntry<N>, 4> &entries,
                                       bool restrict_to_intersection,
                                       ScalarFloat lo_bound, ScalarFloat hi_bound,
                                       std::vector<ScalarFloat> &merged,
                                       std::array<std::vector<ScalarFloat>, N> &Y) {
        merged.clear();
        if (restrict_to_intersection)
            merged = { lo_bound, hi_bound };
        for (const ColumnEntry<N> &e : entries)
            for (ScalarFloat x : e.x)
                if (!restrict_to_intersection || (x >= lo_bound && x <= hi_bound))
                    merged.push_back(x);

        std::sort(merged.begin(), merged.end());
        merged.erase(std::unique(merged.begin(), merged.end()), merged.end());

        size_t n = merged.size();
        for (auto &col : Y)
            col.assign(n, ScalarFloat(0.f));

        for (const ColumnEntry<N> &e : entries) {
            if (e.weight == 0.f || e.x.empty())
                continue;

            size_t l  = e.x.size();
            size_t hi = 1;
            for (size_t k = 0; k < n; ++k) {
                ScalarFloat x = merged[k];
                while (hi < l - 1 && e.x[hi] < x)
                    ++hi;
                size_t lo = hi - 1;

                ScalarFloat h = e.x[hi] - e.x[lo];
                ScalarFloat t = h > 0.f ? (x - e.x[lo]) / h : ScalarFloat(0.f);
                t = std::clamp(t, ScalarFloat(0.f), ScalarFloat(1.f));

                for (size_t c = 0; c < N; ++c)
                    Y[c][k] += e.weight * (e.y[c][lo] + t * (e.y[c][hi] - e.y[c][lo]));
            }
        }
    }

private:

    // =========================================================================
    //  Post-processing
    // =========================================================================

    /// Blend and evaluate the PDF given a cosine angle and the associated
    /// bilinear weights.
    std::pair<Spectrum, Float>
    eval_pdf_flat_at(const PhaseFunctionContext &ctx,
                      const MediumInteraction3f &mei,
                      const Vector3f &wo,
                      const BilinearWeights &bw,
                      Float cos_t,
                      Vector4u starts, Vector4u lens, Vector4u lo4,
                      Mask active) const {
        Vector4u hi4        = lo4 + 1u;
        Vector4f x0_4       = dr::gather<Vector4f>(m_nodes, lo4, active);
        Vector4f x1_4       = dr::gather<Vector4f>(m_nodes, hi4, active);
        Vector4f t4         = (Vector4f(cos_t) - x0_4) / (x1_4 - x0_4);
        Vector4f norms      = dr::gather<Vector4f>(m_norm, bw.idx, active);

        Vector4f domain_lo  = dr::gather<Vector4f>(m_nodes, starts,             active);
        Vector4f domain_hi  = dr::gather<Vector4f>(m_nodes, starts + lens - 1u, active);
        Vector4f w          = dr::select(Vector4f(cos_t) >= domain_lo && Vector4f(cos_t) <= domain_hi,
                                          bw.w, Vector4f(0.f));

        Spectrum phase_val(0.f);
        Float pdf;

        if constexpr (is_polarized_v<Spectrum>) {
            Vector4f m11_4;
            for (size_t c = 0; c < 4; ++c) {
                Vector6f v0  = gather_mueller(lo4[c], active);
                Vector6f v1  = gather_mueller(hi4[c], active);
                Vector6f mi  = dr::fmadd(t4[c], v1 - v0, v0);
                m11_4[c]     = mi[0];
                Float scale  = norms[c] * dr::InvTwoPi<ScalarFloat> * w[c];
                phase_val += MuellerMatrix<Float>(
                    mi[0]*scale,  mi[1]*scale, 0.f,            0.f,
                    mi[1]*scale,  mi[2]*scale, 0.f,            0.f,
                    0.f,            0.f,           mi[3]*scale,  mi[4]*scale,
                    0.f,            0.f,          -mi[4]*scale,  mi[5]*scale);
            }
            pdf = dr::dot(w * norms, m11_4) * dr::InvTwoPi<ScalarFloat>;

            Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? wo : mei.wi,
                     wi_hat = ctx.mode == TransportMode::Radiance ? mei.wi : wo;
            Vector3f x_hat      = dr::cross(-wo_hat, wi_hat),
                     p_axis_in  = dr::normalize(dr::cross(x_hat, -wo_hat)),
                     p_axis_out = dr::normalize(dr::cross(x_hat,  wi_hat));
            phase_val = mueller::rotate_mueller_basis(
                phase_val,
                -wo_hat, p_axis_in,  mueller::stokes_basis(-wo_hat),
                 wi_hat, p_axis_out, mueller::stokes_basis( wi_hat));
            dr::masked(phase_val, dr::isnan(phase_val)) = depolarizer<Spectrum>(0.f);
        } else {
            Vector4f y0_m11 = dr::gather<Vector4f>(m_mueller, lo4 * 6u, active);
            Vector4f y1_m11 = dr::gather<Vector4f>(m_mueller, hi4 * 6u, active);
            Vector4f m11_4  = dr::fmadd(t4, y1_m11 - y0_m11, y0_m11);
            pdf = dr::dot(w * norms, m11_4) * dr::InvTwoPi<ScalarFloat>;
            phase_val = Spectrum(pdf);
        }

        return { phase_val, pdf };
    }

    /// Locate a cosine in the 4 cornering PDF sub-sections and evaluate
    /// its blended PDF.
    std::pair<Spectrum, Float>
    eval_pdf_flat(const PhaseFunctionContext &ctx,
                  const MediumInteraction3f &mei,
                  const Vector3f &wo,
                  const BilinearWeights &bw,
                  Mask active) const {
        Float cos_t     = -dr::dot(wo, mei.wi);
        Vector4u starts = dr::gather<Vector4u>(m_grid_start, bw.idx, active);
        Vector4u lens   = dr::gather<Vector4u>(m_grid_len,   bw.idx, active);
        Vector4u lo4    = ragged_searchsorted4(m_nodes, starts, lens, cos_t, active);
        return eval_pdf_flat_at(ctx, mei, wo, bw, cos_t, starts, lens, lo4, active);
    }


public:

    std::tuple<Vector3f, Spectrum, Float>
    sample(const PhaseFunctionContext &ctx,
           const MediumInteraction3f &mei,
           Float sample1,
           const Point2f &sample2,
           Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        auto [bw, valid] = get_weights(mei, active);
        active = active && valid;

        Float cos_t;
        switch (m_blending_method) {
            case BlendingMethod::Stochastic:
                return sample_stochastic(ctx, mei, sample1, sample2, bw, active);

            case BlendingMethod::Tabulate:
                cos_t = sample_cos_theta_tabulate(sample2.x(), bw, active);
                break;

            case BlendingMethod::Search: {
                auto [cos_t_s, starts, lens, lo4, stable] = sample_cos_theta_search(sample2.x(), bw, active);
                Float sin_t = dr::safe_sqrt(1.f - cos_t_s * cos_t_s);
                auto [sin_phi, cos_phi] = dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());
                Vector3f wo{ sin_t * cos_phi, sin_t * sin_phi, cos_t_s };
                wo = -mei.to_world(wo);

                Vector4u lo4_fresh = ragged_searchsorted4(m_nodes, starts, lens, cos_t_s, active && !stable);
                Vector4u lo4_final = dr::select(stable, lo4, lo4_fresh);

                auto [phase_val, pdf] = eval_pdf_flat_at(ctx, mei, wo, bw, cos_t_s, starts, lens, lo4_final, active);
                Spectrum weight = phase_val * dr::rcp(pdf);

                wo     = dr::select(active, wo,     Vector3f(0.f));
                weight = dr::select(active, weight, Spectrum(0.f));
                pdf    = dr::select(active, pdf,    Float(0.f));
                return { wo, weight, pdf };
            }
        }
        Float sin_t = dr::safe_sqrt(1.f - cos_t * cos_t);
        auto [sin_phi, cos_phi] =
            dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());

        Vector3f wo{ sin_t * cos_phi, sin_t * sin_phi, cos_t };
        wo = -mei.to_world(wo);

        auto [phase_val, pdf] = eval_pdf_flat(ctx, mei, wo, bw, active);
        Spectrum weight = phase_val * dr::rcp(pdf);

        wo     = dr::select(active, wo,     Vector3f(0.f));
        weight = dr::select(active, weight, Spectrum(0.f));
        pdf    = dr::select(active, pdf,    Float(0.f));

        return { wo, weight, pdf };
    }

    std::pair<Spectrum, Float>
    eval_pdf(const PhaseFunctionContext &ctx,
             const MediumInteraction3f &mei,
             const Vector3f &wo,
             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        auto [bw, valid] = get_weights(mei, active);
        active = active && valid;
        auto [val, pdf] = eval_pdf_flat(ctx, mei, wo, bw, active);

        val = dr::select(active, val, Spectrum(0.f));
        pdf = dr::select(active, pdf, Float(0.f));

        return { val, pdf };
    }

    void parameters_changed(const std::vector<std::string> &keys = {}) override {
        // precompute_cdf() only depends on nodes/phase_mueller/grid_start/grid_len
        // (r_eff_volume, v_eff_volume, r_eff_grid, v_eff_grid, sigma_s_weight are
        // read later, at query time, so they don't need it re-run).
        if (keys.empty() || string::contains(keys, "nodes") ||
            string::contains(keys, "phase_mueller") ||
            string::contains(keys, "grid_start") ||
            string::contains(keys, "grid_len"))
            precompute_cdf();
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("r_eff_volume",  m_r_eff_volume.get(), ParamFlags::Differentiable);
        cb->put("v_eff_volume",  m_v_eff_volume.get(), ParamFlags::Differentiable);
        cb->put("r_eff_grid",    m_r_eff_grid,         ParamFlags::NonDifferentiable);
        cb->put("v_eff_grid",    m_v_eff_grid,         ParamFlags::NonDifferentiable);
        cb->put("nodes",           m_nodes,              ParamFlags::NonDifferentiable);
        cb->put("phase_mueller",   m_mueller,            ParamFlags::Differentiable | ParamFlags::Discontinuous);
        cb->put("grid_start",      m_grid_start,         ParamFlags::NonDifferentiable);
        cb->put("grid_len",        m_grid_len,           ParamFlags::NonDifferentiable);
        cb->put("sigma_s_weight",  m_sigma_s_weight,     ParamFlags::Differentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ParticleFieldPhaseFunction[" << std::endl
            << "  n_r = " << m_n_r << "," << std::endl
            << "  n_v = " << m_n_v << "," << std::endl
            << "  total_pts = " << dr::width(m_nodes) << "," << std::endl
            << "  blending_method = " << method_to_string(m_blending_method) << std::endl
            << "]";
        return oss.str();
    }

    FloatStorage get_envelope_nodes() const override {
        size_t n = dr::width(m_nodes);
        std::vector<ScalarFloat> tmp(n);
        for (size_t i = 0; i < n; ++i)
            tmp[i] = dr::slice(m_nodes, i);

        std::sort(tmp.begin(), tmp.end());
        auto last = std::unique(tmp.begin(), tmp.end());
        tmp.erase(last, tmp.end());

        return dr::load<FloatStorage>(tmp.data(), tmp.size());
    }


    void accumulate_envelope(const FloatStorage &nodes, FloatStorage &values) const override {
        size_t n = dr::width(nodes);
        std::vector<ScalarFloat> tmp_values(n, ScalarFloat(0));

        for (int ir = 0; ir < m_n_r; ++ir) {
            for (int iv = 0; iv < m_n_v; ++iv) {
                size_t entry = ir * m_n_v + iv;
                size_t start = dr::slice(m_grid_start, entry);
                size_t len   = dr::slice(m_grid_len,   entry);
                ScalarFloat norm = dr::slice(m_norm,      entry);

                size_t lo = start;

                for (size_t i = 0; i < n; ++i) {
                    ScalarFloat cos_t = dr::slice(nodes, i);

                    while (lo < start + len - 2 && dr::slice(m_nodes, lo + 1) < cos_t)
                        ++lo;

                    size_t hi     = lo + 1;
                    ScalarFloat x0  = dr::slice(m_nodes, lo);
                    ScalarFloat x1  = dr::slice(m_nodes, hi);
                    ScalarFloat t   = dr::clip((cos_t - x0) / (x1 - x0), ScalarFloat(0.f), ScalarFloat(1.f));
                    ScalarFloat y0  = dr::slice(m_mueller, lo * 6u);
                    ScalarFloat y1  = dr::slice(m_mueller, hi * 6u);
                    ScalarFloat m11 = dr::fmadd(t, y1 - y0, y0) * norm * dr::InvTwoPi<ScalarFloat>;
                    tmp_values[i] = dr::maximum(tmp_values[i], m11);
                }
            }
        }

        values = dr::maximum(
	        dr::load<FloatStorage>(tmp_values.data(), tmp_values.size()),
	        values
        );
    }

    MI_DECLARE_CLASS(ParticleFieldPhaseFunction)

private:
    ref<Volume> m_r_eff_volume;
    ref<Volume> m_v_eff_volume;

    FloatStorage  m_r_eff_grid;
    FloatStorage  m_v_eff_grid;
    FloatStorage  m_nodes;
    FloatStorage  m_mueller;
    UInt32Storage m_grid_start;
    UInt32Storage m_grid_len;
    FloatStorage  m_cdf;
    FloatStorage  m_norm;
    FloatStorage  m_sigma_s_weight;

    int m_n_r = 0;
    int m_n_v = 0;

    BlendingMethod m_blending_method;
};

MI_EXPORT_PLUGIN(ParticleFieldPhaseFunction)

NAMESPACE_END(mitsuba)
