#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/volume.h>
#include <mitsuba/render/volumegrid.h>
#include <mitsuba/render/interaction.h>
#include <iomanip>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class ParticlePhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext, Volume, VolumeGrid)

    using FloatStorage  = DynamicBuffer<Float>;
    using UInt32Storage = DynamicBuffer<UInt32>;
    using Vector6u      = dr::Array<UInt32, 6>;
    using Vector6f      = dr::Array<Float, 6>;

    enum class BlendingMethod { BlendedCDF, Stochastic };

    struct BilinearWeights {
        Vector4u idx;
        Vector4f w;
    };

    explicit ParticlePhaseFunction(const Properties &props) : Base(props) {
        m_r_eff_volume = props.get_volume<Volume>("r_eff_volume");
        m_v_eff_volume = props.get_volume<Volume>("v_eff_volume");

        m_n_r = props.get<int>("n_r");
        m_n_v = props.get<int>("n_v");

        auto load_float_grid = [&](const char *name) -> FloatStorage {
            try {
                ref<Object> obj = props.get<ref<Object>>(name);
                auto *vg = dynamic_cast<VolumeGrid *>(obj.get());
                if (!vg)
                    Throw("ParticlePhaseFunction: property \"%s\" must be a VolumeGrid.", name);
                size_t n = (size_t) vg->size().x();
                const ScalarFloat *buf = vg->data();
                FloatStorage out = dr::zeros<FloatStorage>(n);
                for (size_t i = 0; i < n; ++i)
                    dr::scatter(out, Float(buf[i]), UInt32((uint32_t) i));
                return out;
            } catch (...) {}
            return props.get_any<FloatStorage>(name);
        };

        auto load_uint_grid = [&](const char *name) -> UInt32Storage {
            try {
                ref<Object> obj = props.get<ref<Object>>(name);
                auto *vg = dynamic_cast<VolumeGrid *>(obj.get());
                if (!vg)
                    Throw("ParticlePhaseFunction: property \"%s\" must be a VolumeGrid.", name);
                size_t n = (size_t) vg->size().x();
                const ScalarFloat *buf = vg->data();
                UInt32Storage out = dr::zeros<UInt32Storage>(n);
                for (size_t i = 0; i < n; ++i)
                    dr::scatter(out, UInt32((uint32_t) buf[i]), UInt32((uint32_t) i));
                return out;
            } catch (...) {}
            return props.get_any<UInt32Storage>(name);
        };

        m_r_eff_grid    = load_float_grid("r_eff_grid");
        m_v_eff_grid    = load_float_grid("v_eff_grid");
        m_nodes         = props.get_any<FloatStorage>("nodes");
        m_mueller       = props.get_any<FloatStorage>("phase_mueller");
        m_grid_start    = load_uint_grid("grid_start");
        m_grid_len      = load_uint_grid("grid_len");
        m_sigma_s_weight = props.get_any<FloatStorage>("sigma_s_weight");

        std::string method = props.get<std::string>("blending_method", "blended_cdf");
        if (method == "blended_cdf")
            m_blending_method = BlendingMethod::BlendedCDF;
        else if (method == "stochastic")
            m_blending_method = BlendingMethod::Stochastic;
        else
            Throw("ParticlePhaseFunction: unknown blending_method '%s'", method);

        if ((size_t) dr::width(m_mueller) != dr::width(m_nodes) * 6u)
            Throw("ParticlePhaseFunction: phase_mueller must have 6 * len(nodes) elements");
        if ((size_t) dr::width(m_r_eff_grid) != (size_t) m_n_r)
            Throw("ParticlePhaseFunction: r_eff_grid must have n_r elements");
        if ((size_t) dr::width(m_v_eff_grid) != (size_t) m_n_v)
            Throw("ParticlePhaseFunction: v_eff_grid must have n_v elements");
        if ((size_t) dr::width(m_sigma_s_weight) != (size_t)(m_n_r * m_n_v))
            Throw("ParticlePhaseFunction: sigma_s_weight must have n_r * n_v = %d elements", m_n_r * m_n_v);

        if (props.has_property("cdf") && props.has_property("norm")) {
            m_cdf  = props.get_any<FloatStorage>("cdf");
            m_norm = props.get_any<FloatStorage>("norm");
        } else if (props.has_property("cdf") || props.has_property("norm")) {
            Throw("ParticlePhaseFunction: cdf and norm must be provided together");
        } else {
            precompute_cdf();
        }

        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.clear();
        m_components.push_back(m_flags);

        uint32_t start0 = dr::slice(m_grid_start, 0);
        uint32_t len0   = dr::slice(m_grid_len,   0);

        std::ostringstream nodes_ss, m11_ss, m12_ss, m22_ss, m33_ss, m34_ss, m44_ss;
        nodes_ss << std::scientific << std::setprecision(17);
        m11_ss   << std::scientific << std::setprecision(17);
        m12_ss   << std::scientific << std::setprecision(17);
        m22_ss   << std::scientific << std::setprecision(17);
        m33_ss   << std::scientific << std::setprecision(17);
        m34_ss   << std::scientific << std::setprecision(17);
        m44_ss   << std::scientific << std::setprecision(17);
        for (uint32_t k = 0; k < len0; ++k) {
            if (k) { nodes_ss<<","; m11_ss<<","; m12_ss<<","; m22_ss<<",";
                     m33_ss<<","; m34_ss<<","; m44_ss<<","; }
            nodes_ss << (double) dr::slice(m_nodes,   start0 + k);
            m11_ss   << (double) dr::slice(m_mueller, (start0 + k) * 6u + 0u);
            m12_ss   << (double) dr::slice(m_mueller, (start0 + k) * 6u + 1u);
            m22_ss   << (double) dr::slice(m_mueller, (start0 + k) * 6u + 2u);
            m33_ss   << (double) dr::slice(m_mueller, (start0 + k) * 6u + 3u);
            m34_ss   << (double) dr::slice(m_mueller, (start0 + k) * 6u + 4u);
            m44_ss   << (double) dr::slice(m_mueller, (start0 + k) * 6u + 5u);
        }

        std::ostringstream nodes_debug;
        nodes_debug << std::scientific << std::setprecision(17);
        nodes_debug << "nodes[0][0.." << len0 << "]: ";
        bool has_duplicates = false;
        for (uint32_t k = 0; k < len0; ++k) {
            if (k) nodes_debug << ", ";
            double v = (double) dr::slice(m_nodes, start0 + k);
            nodes_debug << v;
            if (k > 0 && v <= (double) dr::slice(m_nodes, start0 + k - 1))
                has_duplicates = true;
        }
        Log(Debug, "%s", nodes_debug.str());

        if (!has_duplicates) {
            Properties ref_props("tabphase_polarized");
            ref_props.set<std::string>("nodes", nodes_ss.str());
            ref_props.set<std::string>("m11",   m11_ss.str());
            ref_props.set<std::string>("m12",   m12_ss.str());
            ref_props.set<std::string>("m22",   m22_ss.str());
            ref_props.set<std::string>("m33",   m33_ss.str());
            ref_props.set<std::string>("m34",   m34_ss.str());
            ref_props.set<std::string>("m44",   m44_ss.str());
            m_ref_phase = PluginManager::instance()->create_object<PhaseFunction<Float,Spectrum>>(ref_props);
        } else {
            Log(Debug, "Skipping reference phase: entry 0 has duplicate/non-increasing nodes");
        }
        Log(Debug, "ParticlePhaseFunction constructed: %s", to_string());
        Log(Debug, "  n_nodes=%zu  n_entries=%d  grid_start[0]=%u  grid_len[0]=%u",
            dr::width(m_nodes), m_n_r * m_n_v, start0, len0);
        Log(Debug, "  m_bisection_steps=%u", m_bisection_steps);
        Log(Debug, "  cdf[start0]=%f  cdf[start0+len0-1]=%f",
            (double)dr::slice(m_cdf, start0), (double)dr::slice(m_cdf, start0 + len0 - 1));
    }

    void precompute_cdf() {
        size_t total_pts = dr::width(m_nodes);
        size_t n = (size_t)(m_n_r * m_n_v);

        m_cdf  = dr::zeros<FloatStorage>(total_pts);
        m_norm = dr::zeros<FloatStorage>(n);

        double min_spacing = dr::Infinity<double>;

        for (size_t i = 0; i < n; ++i) {
            uint32_t start = dr::slice(m_grid_start, i);
            uint32_t len   = dr::slice(m_grid_len,   i);

            if (len < 2)
                Throw("ParticlePhaseFunction: entry %zu has fewer than 2 nodes!", i);

            double running = 0.0;
            dr::scatter(m_cdf, Float(0.f), UInt32(start));
            for (uint32_t k = 0; k < len - 1; ++k) {
                double x0 = (double) dr::slice(m_nodes,   start + k);
                double x1 = (double) dr::slice(m_nodes,   start + k + 1);
                double y0 = (double) dr::slice(m_mueller, (start + k)     * 6u);
                double y1 = (double) dr::slice(m_mueller, (start + k + 1) * 6u);
                running += 0.5 * (y0 + y1) * (x1 - x0);
                dr::scatter(m_cdf, Float((float) running), UInt32(start + k + 1));
                min_spacing = std::min(min_spacing, x1 - x0);
            }

            double total = running;
            if (total > 0.0) {
                dr::scatter(m_norm, Float((float)(1.0 / total)), UInt32((uint32_t) i));
                for (uint32_t k = 0; k < len; ++k) {
                    double val = (double) dr::slice(m_cdf, start + k);
                    dr::scatter(m_cdf, Float((float)(val / total)), UInt32(start + k));
                }
            }
        }

        double full_range = (double) dr::slice(m_nodes, dr::width(m_nodes) - 1)
                          - (double) dr::slice(m_nodes, 0);
        uint32_t steps = (uint32_t) std::ceil(full_range / min_spacing);
        m_bisection_steps = 1u;
        while (m_bisection_steps < steps)
            m_bisection_steps <<= 1u;
    }

    BilinearWeights get_bilinear_weights(Float r_eff, Float v_eff, Mask active) const {
        UInt32 ir = 0u;
        Float tr = Float(0.f);
        if (m_n_r > 1) {
            Float r_min = dr::gather<Float>(m_r_eff_grid, UInt32(0u), active);
            Float r_max = dr::gather<Float>(m_r_eff_grid, UInt32(m_n_r - 1), active);
            Float dr_   = (r_max - r_min) / Float(m_n_r - 1);
            ir  = dr::clip(UInt32((r_eff - r_min) / dr_), 0u, (uint32_t)(m_n_r - 2));
            Float r0 = dr::fmadd(Float(ir),      dr_, r_min);
            Float r1 = dr::fmadd(Float(ir + 1u), dr_, r_min);
            Float dri = r1 - r0;
            tr = dr::select(dr::abs(dri) > dr::Epsilon<Float>, (r_eff - r0) / dri, Float(0.f));
        }

        UInt32 iv = 0u;
        Float tv = Float(0.f);
        if (m_n_v > 1) {
            Float v_min = dr::gather<Float>(m_v_eff_grid, UInt32(0u), active);
            Float v_max = dr::gather<Float>(m_v_eff_grid, UInt32(m_n_v - 1), active);
            Float dv_   = (v_max - v_min) / Float(m_n_v - 1);
            iv  = dr::clip(UInt32((v_eff - v_min) / dv_), 0u, (uint32_t)(m_n_v - 2));
            Float v0 = dr::fmadd(Float(iv),      dv_, v_min);
            Float v1 = dr::fmadd(Float(iv + 1u), dv_, v_min);
            Float dvi = v1 - v0;
            tv = dr::select(dr::abs(dvi) > dr::Epsilon<Float>, (v_eff - v0) / dvi, Float(0.f));
        }

        Float _tr = 1.f - tr, _tv = 1.f - tv;
        UInt32 n_v   = (uint32_t) m_n_v;
        UInt32 ir_hi = m_n_r > 1 ? ir + 1u : ir;
        UInt32 iv_hi = m_n_v > 1 ? iv + 1u : iv;

        BilinearWeights bw;
        bw.idx = Vector4u(ir    * n_v + iv,
                          ir    * n_v + iv_hi,
                          ir_hi * n_v + iv,
                          ir_hi * n_v + iv_hi);
        bw.w   = Vector4f(_tr * _tv, _tr * tv, tr * _tv, tr * tv);

        // Scale by per-entry scattering coefficient and renormalise.
        Vector4f sca  = dr::gather<Vector4f>(m_sigma_s_weight, bw.idx, active);
        bw.w         *= sca;
        Float sca_sum = dr::sum(bw.w);
        bw.w          = dr::select(sca_sum > dr::Epsilon<Float>, bw.w / sca_sum, bw.w);

        return bw;
    }

    static constexpr Vector6u k_mueller_offsets() { return Vector6u(0u, 1u, 2u, 3u, 4u, 5u); }

    Vector6f gather_mueller(UInt32 point, Mask active) const {
        return dr::gather<Vector6f>(m_mueller, point * 6u + k_mueller_offsets(), active);
    }

    Vector4u ragged_searchsorted4(const FloatStorage &arr,
                                   Vector4u start, Vector4u len,
                                   Float query, Mask active) const {
        Vector4u lo = start;
        Vector4u hi = start + len;
        dr::mask_t<Vector4u> running = dr::mask_t<Vector4u>(active) && (lo < hi);

        uint32_t max_len = dr::slice(dr::max(len), 0);
        uint32_t iters   = 0u;
        while ((1u << iters) < max_len)
            ++iters;

        for (uint32_t it = 0u; it < iters && dr::any_nested(running); ++it) {
            Vector4u mid  = (lo + hi) >> 1u;
            Vector4f val  = dr::gather<Vector4f>(arr, mid, running);
            dr::mask_t<Vector4u> go_right = running && (val < Vector4f(query));
            lo = dr::select(go_right, mid + 1u, lo);
            hi = dr::select(running && !go_right, mid, hi);
            running = running && (lo < hi);
        }

        Vector4u result = dr::clip(lo, start + 1u, start + len - 1u) - 1u;
        Log(Debug, "ragged_searchsorted4: start=[%u,%u,%u,%u] len=[%u,%u,%u,%u] query=%s -> lo=[%u,%u,%u,%u] hi=[%u,%u,%u,%u]",
            dr::slice(start[0],0), dr::slice(start[1],0), dr::slice(start[2],0), dr::slice(start[3],0),
            dr::slice(len[0],0),   dr::slice(len[1],0),   dr::slice(len[2],0),   dr::slice(len[3],0),
            std::to_string(dr::slice(query,0)).c_str(),
            dr::slice(result[0],0), dr::slice(result[1],0), dr::slice(result[2],0), dr::slice(result[3],0),
            dr::slice(result[0]+1u,0), dr::slice(result[1]+1u,0), dr::slice(result[2]+1u,0), dr::slice(result[3]+1u,0));
        return result;
    }

    UInt32 searchsorted1(const FloatStorage &arr, UInt32 start, UInt32 len,
                         Float query, Mask active, const char *tag = "?") const {
        UInt32 lo = dr::binary_search<UInt32>(
            start, start + len,
            [&](UInt32 i) DRJIT_INLINE_LAMBDA {
                return active && (dr::gather<Float>(arr, i, active) < query);
            });
        UInt32 result = dr::clip(lo, start + 1u, start + len - 1u) - 1u;
        Log(Debug, "searchsorted1[%s]: start=%u len=%u query=%s -> lo=%s hi=%s",
            tag, dr::slice(start,0), dr::slice(len,0),
            std::to_string(dr::slice(query,0)).c_str(),
            std::to_string(dr::slice(result,0)).c_str(),
            std::to_string(dr::slice(result+1u,0)).c_str());
        return result;
    }

    Vector6f interp_mueller_entry(UInt32 start, UInt32 len,
                                  Float cos_t, Mask active) const {
        UInt32 lo = searchsorted1(m_nodes, start, len, cos_t, active, "interp_mueller_nodes");
        UInt32 hi = lo + 1u;
        Float x0  = dr::gather<Float>(m_nodes, lo, active);
        Float x1  = dr::gather<Float>(m_nodes, hi, active);
        Float t   = (cos_t - x0) / (x1 - x0);
        Vector6f v0 = gather_mueller(lo, active);
        Vector6f v1 = gather_mueller(hi, active);
        return dr::select(active, dr::fmadd(t, v1 - v0, v0), dr::zeros<Vector6f>());
    }

    Float sample_cos_theta_entry(Float u, UInt32 start, UInt32 len,
                                 Mask active) const {
        UInt32 lo = searchsorted1(m_cdf, start, len, u, active, "sample_cdf");
        UInt32 hi = lo + 1u;

        Float x0 = dr::gather<Float>(m_nodes,   lo,      active);
        Float x1 = dr::gather<Float>(m_nodes,   hi,      active);
        Float y0 = dr::gather<Float>(m_mueller, lo * 6u, active);
        Float y1 = dr::gather<Float>(m_mueller, hi * 6u, active);
        Float c0 = dr::gather<Float>(m_cdf,     lo,      active);

        Float w = x1 - x0;
        Float s = (u - c0) / w;

        Float t_linear = (y0 - dr::safe_sqrt(dr::square(y0) + 2.f * s * (y1 - y0))) / (y0 - y1);
        Float t_const  = s / y0;
        Float t        = dr::select(dr::abs(y1 - y0) < dr::Epsilon<Float>, t_const, t_linear);
        return dr::fmadd(dr::clip(t, 0.f, 1.f), w, x0);
    }

    Float eval_blended_cdf(Float x, const BilinearWeights &bw,
                            Vector4u starts, Vector4u lens, Mask active) const {
        Vector4u lo4   = ragged_searchsorted4(m_nodes, starts, lens, x, active);
        Vector4u hi4   = lo4 + 1u;
        Vector4f x0_4  = dr::gather<Vector4f>(m_nodes,   lo4,      active);
        Vector4f x1_4  = dr::gather<Vector4f>(m_nodes,   hi4,      active);
        Vector4f y0_4  = dr::gather<Vector4f>(m_mueller, lo4 * 6u, active);
        Vector4f y1_4  = dr::gather<Vector4f>(m_mueller, hi4 * 6u, active);
        Vector4f cdf0  = dr::gather<Vector4f>(m_cdf,     lo4,      active);
        Vector4f dx    = x1_4 - x0_4;
        Vector4f t4    = dr::clip(dr::select(dx > dr::Epsilon<Float>,
                                             (Vector4f(x) - x0_4) / dx, 0.f), 0.f, 1.f);
        Vector4f cdf_c = cdf0 + dx * t4 * dr::fmadd(0.5f * t4, y1_4 - y0_4, y0_4);
        return dr::dot(bw.w, cdf_c);
    }

    Float eval_blended_pdf(Float x, const BilinearWeights &bw,
                            Vector4u starts, Vector4u lens, Mask active) const {
        Vector4u lo4   = ragged_searchsorted4(m_nodes, starts, lens, x, active);
        Vector4u hi4   = lo4 + 1u;
        Vector4f x0_4  = dr::gather<Vector4f>(m_nodes,   lo4,      active);
        Vector4f x1_4  = dr::gather<Vector4f>(m_nodes,   hi4,      active);
        Vector4f y0_4  = dr::gather<Vector4f>(m_mueller, lo4 * 6u, active);
        Vector4f y1_4  = dr::gather<Vector4f>(m_mueller, hi4 * 6u, active);
        Vector4f dx    = x1_4 - x0_4;
        Vector4f t4    = dr::clip(dr::select(dx > dr::Epsilon<Float>,
                                             (Vector4f(x) - x0_4) / dx, 0.f), 0.f, 1.f);
        Vector4f norm  = dr::gather<Vector4f>(m_norm, bw.idx, active);
        return dr::dot(bw.w * norm, dr::fmadd(t4, y1_4 - y0_4, y0_4)) * dr::InvTwoPi<ScalarFloat>;
    }

    Float sample_cos_theta_blended_cdf(Float u, const BilinearWeights &bw,
                                       Mask active) const {
        Vector4u starts = dr::gather<Vector4u>(m_grid_start, bw.idx, active);
        Vector4u lens   = dr::gather<Vector4u>(m_grid_len,   bw.idx, active);

        Float x_min = dr::max(dr::gather<Vector4f>(m_nodes, starts,            active));
        Float x_max = dr::min(dr::gather<Vector4f>(m_nodes, starts + lens - 1u, active));

        Float range = x_max - x_min;
        uint32_t N  = m_bisection_steps;

        UInt32 lo = dr::binary_search<UInt32>(
            UInt32(0u), UInt32(N),
            [&](UInt32 i) DRJIT_INLINE_LAMBDA {
                Float x = dr::fmadd(Float(i) / Float(N), range, x_min);
                return eval_blended_cdf(x, bw, starts, lens, active) < u;
            });

        Float x_lo = dr::fmadd(Float(dr::maximum(lo,      UInt32(0u))) / Float(N), range, x_min);
        Float x_hi = dr::fmadd(Float(dr::minimum(lo + 1u, UInt32(N) )) / Float(N), range, x_min);

        Float y0_b = eval_blended_pdf(x_lo, bw, starts, lens, active);
        Float y1_b = eval_blended_pdf(x_hi, bw, starts, lens, active);
        Float c0_b = eval_blended_cdf(x_lo, bw, starts, lens, active);

        Float w = x_hi - x_lo;
        Float s = dr::select(w > dr::Epsilon<Float>, (u - c0_b) / w, 0.f);

        Float t_linear = (y0_b - dr::safe_sqrt(dr::square(y0_b) + 2.f * s * (y1_b - y0_b)))
                         / (y0_b - y1_b);
        Float t_const  = dr::select(y0_b > dr::Epsilon<Float>, s / y0_b, 0.f);
        Float t        = dr::select(dr::abs(y1_b - y0_b) < dr::Epsilon<Float>, t_const, t_linear);
        return dr::fmadd(dr::clip(t, 0.f, 1.f), w, x_lo);
    }

    std::tuple<Vector3f, Spectrum, Float>
    sample_stochastic(const PhaseFunctionContext &ctx,
                      const MediumInteraction3f &mei,
                      Float u_select, const Point2f &sample2,
                      const BilinearWeights &bw, Mask active) const {
        Vector4f cumw = dr::prefix_sum(bw.w, false);
        UInt32 idx = bw.idx[0];
        idx = dr::select(u_select > cumw[0], bw.idx[1], idx);
        idx = dr::select(u_select > cumw[1], bw.idx[2], idx);
        idx = dr::select(u_select > cumw[2], bw.idx[3], idx);

        UInt32 s = dr::gather<UInt32>(m_grid_start, idx, active);
        UInt32 l = dr::gather<UInt32>(m_grid_len,   idx, active);
        Float  norm = dr::gather<Float>(m_norm, idx, active);

        Float cos_t = sample_cos_theta_entry(sample2.x(), s, l, active);
        Float sin_t = dr::safe_sqrt(1.f - cos_t * cos_t);
        auto [sin_phi, cos_phi] = dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());
        Vector3f wo = -mei.to_world(Vector3f(sin_t * cos_phi, sin_t * sin_phi, cos_t));

        UInt32 lo = searchsorted1(m_nodes, s, l, cos_t, active, "stochastic_nodes");
        UInt32 hi = lo + 1u;
        Float x0  = dr::gather<Float>(m_nodes, lo, active);
        Float x1  = dr::gather<Float>(m_nodes, hi, active);
        Float t   = (cos_t - x0) / (x1 - x0);

        Vector6f v0 = gather_mueller(lo, active);
        Vector6f v1 = gather_mueller(hi, active);
        Vector6f mi = dr::fmadd(t, v1 - v0, v0);
        Float pdf   = mi[0] * norm * dr::InvTwoPi<ScalarFloat>;

        Spectrum weight(0.f);
        if constexpr (is_polarized_v<Spectrum>) {
            Float scale = norm * dr::InvTwoPi<ScalarFloat>;
            Spectrum phase_val = MuellerMatrix<Float>(
                mi[0]*scale,  mi[1]*scale, 0,            0,
                mi[1]*scale,  mi[2]*scale, 0,            0,
                0,            0,           mi[3]*scale,  mi[4]*scale,
                0,            0,          -mi[4]*scale,  mi[5]*scale);

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
            weight = phase_val * dr::rcp(pdf);
        } else {
            weight = Spectrum(1.f);
        }
        wo     = dr::select(active, wo,     Vector3f(0.f));
        weight = dr::select(active, weight, Spectrum(0.f));
        pdf    = dr::select(active, pdf,    Float(0.f));
        return { wo, weight, pdf };
    }

    std::pair<BilinearWeights, Mask>
    get_weights(const MediumInteraction3f &mei, Mask active) const {
        Float r_eff = m_r_eff_volume->eval_1(mei, active);
        Float v_eff = m_v_eff_volume->eval_1(mei, active);

        Mask nan_mask = dr::isnan(r_eff) || dr::isnan(v_eff);
        if (dr::any(active && nan_mask)) {
            std::ostringstream oss;
            oss << "ParticlePhaseFunction: NaN r_eff or v_eff at interaction point.\n";
            oss << "  mei.p       = " << mei.p << "\n";
            oss << "  r_eff value = " << r_eff << "\n";
            oss << "  v_eff value = " << v_eff << "\n";
            oss << "  r_eff is NaN: " << dr::isnan(r_eff) << "\n";
            oss << "  v_eff is NaN: " << dr::isnan(v_eff) << "\n";
            oss << "  r_eff_volume to_local:\n" << m_r_eff_volume->bbox() << "\n";
            oss << "  v_eff_volume to_local:\n" << m_v_eff_volume->bbox() << "\n";
            Throw("%s", oss.str());
        }

        if (m_n_r > 1) {
            Float r_min = dr::gather<Float>(m_r_eff_grid, UInt32(0u), active);
            Float r_max = dr::gather<Float>(m_r_eff_grid, UInt32(m_n_r - 1), active);
            if (dr::any(active && (r_eff < r_min || r_eff > r_max)))
                Throw("ParticlePhaseFunction: r_eff out of dataset range.");
        }

        if (m_n_v > 1) {
            Float v_min = dr::gather<Float>(m_v_eff_grid, UInt32(0u), active);
            Float v_max = dr::gather<Float>(m_v_eff_grid, UInt32(m_n_v - 1), active);
            if (dr::any(active && (v_eff < v_min || v_eff > v_max)))
                Throw("ParticlePhaseFunction: v_eff out of dataset range.");
        }

        return { get_bilinear_weights(r_eff, v_eff, active), active };
    }

    std::pair<Spectrum, Float>
    eval_pdf_flat(const PhaseFunctionContext &ctx,
                  const MediumInteraction3f &mei,
                  const Vector3f &wo,
                  const BilinearWeights &bw,
                  Mask active) const {
        Float cos_t         = -dot(wo, mei.wi);
        Vector4u starts     = dr::gather<Vector4u>(m_grid_start, bw.idx, active);
        Vector4u lens       = dr::gather<Vector4u>(m_grid_len,   bw.idx, active);
        Vector4u lo4        = ragged_searchsorted4(m_nodes, starts, lens, cos_t, active);
        Vector4u hi4        = lo4 + 1u;
        Vector4f x0_4       = dr::gather<Vector4f>(m_nodes, lo4, active);
        Vector4f x1_4       = dr::gather<Vector4f>(m_nodes, hi4, active);
        Vector4f t4         = (Vector4f(cos_t) - x0_4) / (x1_4 - x0_4);
        Vector4f norms      = dr::gather<Vector4f>(m_norm, bw.idx, active);
        Vector4f y0_m11     = dr::gather<Vector4f>(m_mueller, lo4 * 6u, active);
        Vector4f y1_m11     = dr::gather<Vector4f>(m_mueller, hi4 * 6u, active);
        Float pdf           = dr::dot(bw.w * norms,
                                      dr::fmadd(t4, y1_m11 - y0_m11, y0_m11))
                              * dr::InvTwoPi<ScalarFloat>;

        Spectrum phase_val(0.f);

        if constexpr (is_polarized_v<Spectrum>) {
            for (uint32_t c = 0u; c < 4u; ++c) {
                Vector6f v0  = gather_mueller(lo4[c], active);
                Vector6f v1  = gather_mueller(hi4[c], active);
                Vector6f mi  = dr::fmadd(t4[c], v1 - v0, v0);
                Float scale  = norms[c] * dr::InvTwoPi<ScalarFloat> * bw.w[c];
                phase_val += MuellerMatrix<Float>(
                    mi[0]*scale,  mi[1]*scale, 0,            0,
                    mi[1]*scale,  mi[2]*scale, 0,            0,
                    0,            0,           mi[3]*scale,  mi[4]*scale,
                    0,            0,          -mi[4]*scale,  mi[5]*scale);
            }

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
            Vector4f m11_4 = dr::fmadd(t4, y1_m11 - y0_m11, y0_m11);
            phase_val = Spectrum(dr::dot(bw.w * norms, m11_4) * dr::InvTwoPi<ScalarFloat>);
        }

        return { phase_val, pdf };
    }

    std::tuple<Vector3f, Spectrum, Float>
    sample(const PhaseFunctionContext &ctx,
           const MediumInteraction3f &mei,
           Float sample1,
           const Point2f &sample2,
           Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        auto [bw, valid] = get_weights(mei, active);
        active = active && valid;

        if (m_blending_method == BlendingMethod::Stochastic)
            return sample_stochastic(ctx, mei, sample1, sample2, bw, active);

        Float cos_t = sample_cos_theta_blended_cdf(sample2.x(), bw, active);
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

        if (m_ref_phase) {
            auto [ref_wo, ref_weight, ref_pdf] = m_ref_phase->sample(ctx, mei, sample1, sample2, active);
            Log(Debug, "sample: our pdf=%s ref_pdf=%s our_wo=%s ref_wo=%s",
                std::to_string(dr::slice(pdf,0)).c_str(),
                std::to_string(dr::slice(ref_pdf,0)).c_str(),
                std::to_string(dr::slice(wo.x(),0)).c_str(),
                std::to_string(dr::slice(ref_wo.x(),0)).c_str());
        }

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

        if (m_ref_phase) {
            auto [ref_val, ref_pdf] = m_ref_phase->eval_pdf(ctx, mei, wo, active);
            Log(Debug, "eval_pdf: our pdf=%s ref_pdf=%s",
                std::to_string(dr::slice(pdf,0)).c_str(),
                std::to_string(dr::slice(ref_pdf,0)).c_str());
        }

        return { val, pdf };
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("r_eff_volume",  m_r_eff_volume.get(), ParamFlags::NonDifferentiable);
        cb->put("v_eff_volume",  m_v_eff_volume.get(), ParamFlags::NonDifferentiable);
        cb->put("r_eff_grid",    m_r_eff_grid,         ParamFlags::NonDifferentiable);
        cb->put("v_eff_grid",    m_v_eff_grid,         ParamFlags::NonDifferentiable);
        cb->put("nodes",           m_nodes,              ParamFlags::NonDifferentiable);
        cb->put("phase_mueller",   m_mueller,            ParamFlags::NonDifferentiable);
        cb->put("grid_start",      m_grid_start,         ParamFlags::NonDifferentiable);
        cb->put("grid_len",        m_grid_len,           ParamFlags::NonDifferentiable);
        cb->put("sigma_s_weight",  m_sigma_s_weight,     ParamFlags::NonDifferentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ParticlePhaseFunction[" << std::endl
            << "  n_r = " << m_n_r << "," << std::endl
            << "  n_v = " << m_n_v << "," << std::endl
            << "  total_pts = " << dr::width(m_nodes) << "," << std::endl
            << "  blending_method = "
            << (m_blending_method == BlendingMethod::BlendedCDF
                ? "blended_cdf" : "stochastic") << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(ParticlePhaseFunction)

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

    uint32_t m_bisection_steps = 1u << 20;

    BlendingMethod m_blending_method;

    ref<PhaseFunction<Float, Spectrum>> m_ref_phase;
};

MI_EXPORT_PLUGIN(ParticlePhaseFunction)

NAMESPACE_END(mitsuba)
