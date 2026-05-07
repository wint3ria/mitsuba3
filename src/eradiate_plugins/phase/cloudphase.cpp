#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/volume.h>
#include <mitsuba/render/interaction.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-cloudphase:

Cloud phase function (:monosp:`cloudphase`)
-------------------------------------------

.. pluginparameters::

 * - index_volume
   - |volume|
   - Single-channel gridvolume storing integer indices (0-based) as floats.
     Negative values mark empty cells. Use ``filter_type=nearest``.

 * - n_entries
   - |int|
   - Number of distinct phase function entries N.

 * - nodes
   - (total_pts,) float
   - Flat concatenation of per-entry cos-theta abscissae.
     Entry i occupies nodes[grid_start[i] : grid_start[i]+grid_len[i]].

 * - phase_mueller
   - (total_pts * 6,) float
   - Interleaved Mueller coefficients, layout (total_pts, 6) flattened in
     C order. The 6 channels per point are: M11, M12, M22, M33, M34, M44.
     This plugin thus support spherical and spheroidal particles phase functions.

 * - grid_start
   - (N,) uint32
   - Start offset of entry i in the flat arrays.

 * - grid_len
   - (N,) uint32
   - Number of nodes for entry i.

 * - cdf
   - (total_pts,) float — optional
   - Precomputed normalized CDF. If provided together with ``norm``, skips
     CDF construction. Must be strictly monotone within each entry.

 * - norm
   - (N,) float — optional
   - Precomputed per-entry normalization constants (1/integral of M11).
     Must be provided together with ``cdf``.

Empty cells (negative index volume value) raise an error when sampled.

The plugin precomputes a normalized CDF and per-entry normalization
constant at construction time unless ``cdf`` and ``norm`` are provided.
Sampling uses an O(log len) binary search into the CDF.

*/

template <typename Float, typename Spectrum>
class CloudPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext, Volume)

    using FloatStorage  = DynamicBuffer<Float>;
    using UInt32Storage = DynamicBuffer<UInt32>;
    using Int32Storage  = DynamicBuffer<dr::int32_array_t<Float>>;
    using Vector6f      = dr::Array<Float, 6>;

    explicit CloudPhaseFunction(const Properties &props) : Base(props) {
        m_index_volume = props.get_volume<Volume>("index_volume");
        m_n_entries    = props.get<int>("n_entries");
        m_nodes        = props.get_any<FloatStorage>("nodes");
        m_mueller      = props.get_any<FloatStorage>("phase_mueller");
        m_grid_start   = props.get_any<UInt32Storage>("grid_start");
        m_grid_len     = props.get_any<UInt32Storage>("grid_len");

        if (m_n_entries == 0)
            Throw("CloudPhaseFunction: n_entries is zero!");

        if ((size_t) dr::width(m_mueller) != dr::width(m_nodes) * 6u)
            Throw("CloudPhaseFunction: phase_mueller must have 6 * len(nodes) elements");

        if (props.has_property("cdf") && props.has_property("norm")) {
            m_cdf  = props.get_any<FloatStorage>("cdf");
            m_norm = props.get_any<FloatStorage>("norm");

            if (dr::width(m_cdf) != dr::width(m_nodes))
                Throw("CloudPhaseFunction: cdf must have the same length as nodes");
            if ((size_t) dr::width(m_norm) != (size_t) m_n_entries)
                Throw("CloudPhaseFunction: norm must have n_entries elements");
        } else if (props.has_property("cdf") || props.has_property("norm")) {
            Throw("CloudPhaseFunction: cdf and norm must be provided together");
        } else {
            precompute_cdf();
        }

        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.clear();
        m_components.push_back(m_flags);
    }

    void precompute_cdf() {
        size_t total_pts = dr::width(m_nodes);
        size_t n         = (size_t) m_n_entries;

        m_cdf  = dr::zeros<FloatStorage>(total_pts);
        m_norm = dr::zeros<FloatStorage>(n);

        for (size_t i = 0; i < n; ++i) {
            uint32_t start = dr::slice(m_grid_start, i);
            uint32_t len   = dr::slice(m_grid_len,   i);

            if (len < 2)
                Throw("CloudPhaseFunction: entry %zu has fewer than 2 nodes!", i);

            double running = 0.0;
            dr::scatter(m_cdf, Float(0.f), UInt32(start));
            for (uint32_t k = 0; k < len - 1; ++k) {
                double x0 = (double) dr::slice(m_nodes,   start + k);
                double x1 = (double) dr::slice(m_nodes,   start + k + 1);
                double y0 = (double) dr::slice(m_mueller, (start + k)     * 6u);
                double y1 = (double) dr::slice(m_mueller, (start + k + 1) * 6u);
                running += 0.5 * (y0 + y1) * (x1 - x0);
                dr::scatter(m_cdf, Float((float) running), UInt32(start + k + 1));
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
    }

    std::pair<Int32, Mask> get_index(const MediumInteraction3f &mei,
                                     Mask active) const {
        Float raw  = m_index_volume->eval_1(mei, active);
        Mask valid = active && (raw >= 0.f);
        if (!dr::all(valid))
            Throw("Sampling invalid index position in space.");
        Int32 idx = dr::clip(Int32(raw), 0, m_n_entries - 1);
        return { idx, valid };
    }

    Vector6f gather_mueller(UInt32 point, Mask active) const {
        UInt32 base = point * 6u;
        Vector6f r;
        for (uint32_t c = 0u; c < 6u; ++c)
            r[c] = dr::gather<Float>(m_mueller, base + c, active);
        return r;
    }

    Vector6f interp_mueller(UInt32 start, UInt32 len,
                            Float cos_t, Mask active) const {
        UInt32 lo = dr::binary_search<UInt32>(
            start, start + len,
            [&](UInt32 i) DRJIT_INLINE_LAMBDA {
                return dr::gather<Float>(m_nodes, i, active) < cos_t;
            });

        lo = dr::maximum(dr::minimum(lo, start + len - 1u), start + 1u) - 1u;
        UInt32 hi = lo + 1u;

        Float x0 = dr::gather<Float>(m_nodes, lo, active);
        Float x1 = dr::gather<Float>(m_nodes, hi, active);
        Float t  = (cos_t - x0) / (x1 - x0);

        Vector6f v0 = gather_mueller(lo, active);
        Vector6f v1 = gather_mueller(hi, active);

        return dr::select(active, dr::fmadd(t, v1 - v0, v0), dr::zeros<Vector6f>());
    }

    Float sample_cos_theta(Float u, UInt32 start, UInt32 len,
                           Mask active) const {
        UInt32 lo = dr::binary_search<UInt32>(
            start, start + len,
            [&](UInt32 i) DRJIT_INLINE_LAMBDA {
                return dr::gather<Float>(m_cdf, i, active) < u;
            });

        lo = dr::maximum(lo, start + 1u) - 1u;
        UInt32 hi = lo + 1u;

        Float c0 = dr::gather<Float>(m_cdf, lo, active);
        Float c1 = dr::gather<Float>(m_cdf, hi, active);
        Float x0 = dr::gather<Float>(m_nodes, lo, active);
        Float x1 = dr::gather<Float>(m_nodes, hi, active);

        Float dc = dr::maximum(c1 - c0, dr::Epsilon<Float>);
        Float t  = dr::clip((u - c0) / dc, Float(0), Float(1));

        return dr::fmadd(t, x1 - x0, x0);
    }

    std::pair<Spectrum, Float>
    eval_pdf_flat(const PhaseFunctionContext &ctx,
                  const MediumInteraction3f &mei,
                  const Vector3f &wo,
                  UInt32 index, UInt32 start, UInt32 len,
                  Mask active) const {

        Float cos_t = -dot(wo, mei.wi);

        Vector6f m = interp_mueller(start, len, cos_t, active);
        Float norm = dr::gather<Float>(m_norm, index, active);
        Float pdf  = m[0] * norm * dr::InvTwoPi<ScalarFloat>;

        Spectrum phase_val(0.f);

        if constexpr (is_polarized_v<Spectrum>) {
            phase_val = MuellerMatrix<Float>(
                m[0],  m[1], 0,     0,
                m[1],  m[2], 0,     0,
                0,     0,    m[3],  m[4],
                0,     0,   -m[4],  m[5]);

            phase_val *= norm * dr::InvTwoPi<ScalarFloat>;

            Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? wo : mei.wi,
                     wi_hat = ctx.mode == TransportMode::Radiance ? mei.wi : wo;

            Vector3f x_hat      = dr::cross(-wo_hat, wi_hat),
                     p_axis_in  = dr::normalize(dr::cross(x_hat, -wo_hat)),
                     p_axis_out = dr::normalize(dr::cross(x_hat,  wi_hat));

            phase_val = mueller::rotate_mueller_basis(
                phase_val,
                -wo_hat, p_axis_in,  mueller::stokes_basis(-wo_hat),
                 wi_hat, p_axis_out, mueller::stokes_basis( wi_hat));

            dr::masked(phase_val, dr::isnan(phase_val)) =
                depolarizer<Spectrum>(0.f);
        } else {
            phase_val = Spectrum(m[0]) * norm * dr::InvTwoPi<ScalarFloat>;
        }

        return { phase_val, pdf };
    }

    std::tuple<Vector3f, Spectrum, Float>
    sample(const PhaseFunctionContext &ctx,
           const MediumInteraction3f &mei,
           Float /* sample1 */,
           const Point2f &sample2,
           Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        auto [index, valid] = get_index(mei, active);
        active = active & valid;

        UInt32 start = dr::gather<UInt32>(m_grid_start, UInt32(index), active);
        UInt32 len   = dr::gather<UInt32>(m_grid_len,   UInt32(index), active);

        Float cos_t_prime = sample_cos_theta(sample2.x(), start, len, active);
        Float sin_t_prime = dr::safe_sqrt(1.f - cos_t_prime * cos_t_prime);
        auto [sin_phi, cos_phi] =
            dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());

        Vector3f wo{ sin_t_prime * cos_phi,
                     sin_t_prime * sin_phi,
                     cos_t_prime };
        wo = -mei.to_world(wo);

        auto [phase_val, pdf] = eval_pdf_flat(
            ctx, mei, wo, UInt32(index), start, len, active);
        Spectrum weight = phase_val * dr::rcp(pdf);

        wo     = dr::select(valid, wo,     Vector3f(0.f));
        weight = dr::select(valid, weight, Spectrum(0.f));
        pdf    = dr::select(valid, pdf,    Float(0));

        return { wo, weight, pdf };
    }

    std::pair<Spectrum, Float>
    eval_pdf(const PhaseFunctionContext &ctx,
             const MediumInteraction3f &mei,
             const Vector3f &wo,
             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        auto [index, valid] = get_index(mei, active);
        active = active & valid;

        UInt32 start = dr::gather<UInt32>(m_grid_start, UInt32(index), active);
        UInt32 len   = dr::gather<UInt32>(m_grid_len,   UInt32(index), active);

        auto [val, pdf] = eval_pdf_flat(
            ctx, mei, wo, UInt32(index), start, len, active);

        val = dr::select(valid, val, Spectrum(0.f));
        pdf = dr::select(valid, pdf, Float(0));

        return { val, pdf };
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("index_volume",  m_index_volume.get(), ParamFlags::NonDifferentiable);
        cb->put("nodes",         m_nodes,              ParamFlags::NonDifferentiable);
        cb->put("phase_mueller", m_mueller,            ParamFlags::NonDifferentiable);
        cb->put("grid_start",    m_grid_start,         ParamFlags::NonDifferentiable);
        cb->put("grid_len",      m_grid_len,           ParamFlags::NonDifferentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "CloudPhaseFunction[" << std::endl
            << "  n_entries = " << m_n_entries        << "," << std::endl
            << "  total_pts = " << dr::width(m_nodes) << "," << std::endl
            << "  index_volume = " << string::indent(m_index_volume) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(CloudPhaseFunction)

private:
    ref<Volume> m_index_volume;

    FloatStorage  m_nodes;      ///< cos-theta abscissae       (total_pts,)
    FloatStorage  m_mueller;    ///< Mueller data              (total_pts * 6,)
    UInt32Storage m_grid_start; ///< entry start offsets       (N,)
    UInt32Storage m_grid_len;   ///< entry node counts         (N,)
    FloatStorage  m_cdf;        ///< normalized CDF            (total_pts,)
    FloatStorage  m_norm;       ///< 1/integral per entry      (N,)

    int m_n_entries = 0;
};

MI_EXPORT_PLUGIN(CloudPhaseFunction)

NAMESPACE_END(mitsuba)
