#include <array>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-tabphase_polarized:

Lookup table (polarized) phase function (:monosp:`tabphase_polarized`)
----------------------------------------------------------------------

.. pluginparameters::

 * - m11
   - |string|
   - A comma-separated list of phase matrix coefficient 1,1 of the
     phase function, parametrized by the cosine of the scattering angle.
   - |exposed|

 * - m12
   - |string|
   - A comma-separated list of phase matrix coefficient 1,2 of the
     phase function, parametrized by the cosine of the scattering angle.
   - |exposed|

 * - m22
   - |string|
   - A comma-separated list of phase matrix coefficient 2,2 of the
     phase function, parametrized by the cosine of the scattering angle.
   - |exposed|

 * - m33
   - |string|
   - A comma-separated list of phase matrix coefficient 3,3 of the
     phase function, parametrized by the cosine of the scattering angle.
   - |exposed|

 * - m34
   - |string|
   - A comma-separated list of phase matrix coefficient 3,4 of the
     phase function, parametrized by the cosine of the scattering angle.
   - |exposed|

 * - m44
   - |string|
   - A comma-separated list of phase matrix coefficient 4,4 of the
     phase function, parametrized by the cosine of the scattering angle.
   - |exposed|

 * - nodes
   - |string|
   - A comma-separated list of :math:`\cos \theta` specifying the grid on which
     `values` are defined. Bounds must be [-1, 1] and values must be strictly
     increasing. Must have the same length as `values`.
   - |exposed|

This plugin implements a generic phase function model for isotropic media
parametrized by a lookup table giving values of the phase function as a
function of the cosine of the scattering angle.

.. admonition:: Notes

   * The scattering angle cosine is here defined as the dot product of the
     incoming and outgoing directions, where the incoming, resp. outgoing
     direction points *toward*, resp. *outward* the interaction point.
   * From this follows that :math:`\cos \theta = 1` corresponds to forward
     scattering.
   * The lookup table supports irregularly spaced points between -1 and 1.
   * Phase function values are automatically normalized.
   * For polarized phase functions, this assumes (for the time being) the
     structure of a phase function with spherically symmetric particles,
     i.e. there are only four unique elements of the Mueller matrix:
     `M_{11}`, `M_{12}`, `M_{33}`, and `M_{34}`
*/

/**
 * \brief 1D array defined in terms of an *irregularly* sampled linear
 * interpolant
 *
 * This data structure represents an array that is defined as a linear
 * interpolant of an irregularly discretized signal. The nodes represent
 * the irregular abscissa on which the array is represented.
 *
 * Notes: The data array can be different types as long as they are laid out in
 * a contiguous array of Float. The templated Value will specify the output
 * type of eval_data(). Also note that the size of data should be a multiple of
 * the size of nodes.
 *
 * TODO : This class doesn't fully use vectorization to its advantage. Ideally
 * we want an array of VectorXf, the crux still being that we want to
 * access each column of the nested array to update parameters.
 */
template <typename Float, int Size> struct IrregularInterpolant {

    using FloatStorage   = DynamicBuffer<Float>;
    using UInt32         = dr::uint32_array_t<Float>;
    using Index          = dr::uint32_array_t<Float>;
    using Mask           = dr::mask_t<Float>;
    using Vector2u       = dr::Array<UInt32, 2>;
    using ScalarFloat    = dr::scalar_t<Float>;
    using ScalarVector2f = dr::Array<ScalarFloat, 2>;

    using VectorXf = dr::Array<Float, Size>;

public:
    /// Create an uninitialized IrregularInterpolant instance
    IrregularInterpolant() {}

    // Constructor with data
    IrregularInterpolant(const ScalarFloat *nodes,
                         std::array<std::vector<ScalarFloat>, Size> data,
                         size_t nodes_size)
        : m_nodes(dr::load<FloatStorage>(nodes, nodes_size)){

        for(size_t i = 0; i < Size; ++i)
            m_data[i] = dr::load<FloatStorage>(data[i].data(), nodes_size);

        update();

    }

    /// Update the internal state. Must be invoked when changing the data or
    /// range.
    void update() {

        bool valid_size = true;
        for(int i = 0; i < Size; ++i)
            valid_size &= m_data[i].size() == m_nodes.size();

        if (!valid_size)
            Throw("IrregularInterpolant: 'data' and 'nodes' size "
                  "mismatch!");

        if constexpr (dr::is_jit_v<Float>) {
            if (m_nodes.size() < 2)
                Throw("IrregularInterpolant: needs at least two "
                      "entries!");

            uint32_t size = m_nodes.size() - 1;
            m_range =
                ScalarVector2f(dr::slice(m_nodes, 0), dr::slice(m_nodes, size));
        } else {
            if (m_nodes.size() < 2)
                Throw("IrregularInterpolant: needs at least two "
                      "entries!");

            uint32_t size            = m_nodes.size();
            const ScalarFloat *nodes = m_nodes.data();
            m_range                  = ScalarVector2f(dr::Infinity<ScalarFloat>,
                                                      -dr::Infinity<ScalarFloat>);

            for (size_t i = 0; i < size - 1; ++i) {
                double x0 = (double) nodes[0], x1 = (double) nodes[1];

                m_range.x() = dr::minimum(m_range.x(), (ScalarFloat) x0);
                m_range.y() = dr::maximum(m_range.y(), (ScalarFloat) x1);

                nodes++;
                if (!(x1 > x0)) {
                    Throw("IrregularInterpolant: node positions "
                          "must be strictly increasing!");
                }
            }
        }
    }

    /// Return the nodes of the underlying discretization
    FloatStorage &nodes() { return m_nodes; }

    /// Return the nodes of the underlying discretization (const version)
    const FloatStorage &nodes() const { return m_nodes; }

    /// Return the unnormalized discretized probability density function
    FloatStorage &data(int i) { return m_data[i]; }
    const FloatStorage &data(int i) const { return m_data[i]; }

    /// Evaluate the data value at position x
    VectorXf eval_data(Float x, Mask active = true) const {
        MI_MASK_ARGUMENT(active);

        active &= x >= m_range.x() && x <= m_range.y();

        Index index = dr::binary_search<Index>(
            0, (uint32_t) m_nodes.size(), [&](Index index) DRJIT_INLINE_LAMBDA {
                return dr::gather<Float>(m_nodes, index, active) < x;
            });

        index = dr::maximum(dr::minimum(index, (uint32_t) m_nodes.size() - 1u),
                            1u) -
                1u;

        VectorXf y0, y1 = dr::zeros<VectorXf>();

        Float x0 = dr::gather<Float>(m_nodes, index, active),
              x1 = dr::gather<Float>(m_nodes, index + 1u, active);

        for(size_t i = 0; i< Size; ++i) {
            y0[i] = dr::gather<Float>(m_data[i], index, active);
            y1[i] = dr::gather<Float>(m_data[i], index + 1u, active);
        }

        x = (x - x0) / (x1 - x0);

        return dr::select(active, dr::fmadd(x, y1 - y0, y0), 0.f);
    }

private:
    FloatStorage m_nodes;
    dr::Array<FloatStorage, Size> m_data;
    ScalarVector2f m_range{ 0.f, 0.f };
};

template <typename Float, typename Spectrum>
class TabulatedPolarizedPhaseFunction final
    : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext)

    using VectorArray5 = std::array<std::vector<ScalarFloat>, 5>;
    using Vector5f = dr::Array<Float, 5>;

    TabulatedPolarizedPhaseFunction(const Properties &props) : Base(props) {

        if (props.type("m11") == Properties::Type::String) {

            // Extract required properties, nodes (cos_theta) and m11.
            std::vector<std::string> cos_theta_str =
                string::tokenize(props.get<std::string>("nodes"), " ,");
            std::vector<std::string> m11_str = string::tokenize(props.get<std::string>("m11"), " ,");

            if (cos_theta_str.size() != m11_str.size()) {
                Throw("TabulatedPolarizedPhaseFunction: 'cos_theta_str' and "
                      "'m11_str' parameters must have the same size!");
            }

            std::vector<ScalarFloat> cos_theta, m11;
            cos_theta.reserve(cos_theta_str.size());
            m11.reserve(m11_str.size());

            // convert from string to float arrays
            for (size_t i = 0; i < cos_theta_str.size(); ++i) {
                ScalarVector3f ms{ 0.f, 0.f, 0.f };
                try {
                    cos_theta.push_back(
                        string::stof<ScalarFloat>(cos_theta_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'",
                          cos_theta_str[i]);
                }
                try {
                    m11.push_back(string::stof<ScalarFloat>(m11_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'",
                          m11_str[i]);
                }
            }

            // Initialize the distribution which will drive the phase function
            m_m11 = IrregularContinuousDistribution<Float>(
                cos_theta.data(), m11.data(), m11.size());

            size_t size = 5;
            std::array<std::string, 5> coef_names;
            coef_names = { "m12", "m22", "m33", "m34", "m44" };

            std::array<std::vector<ScalarFloat>, 5> ms_vec;

            for (size_t i = 0; i < size; ++i) {
                // Initialize coefficients to 0 in case no value is passed.
                ms_vec[i] = std::vector<ScalarFloat>(cos_theta_str.size(), 0.f);
                const std::string &raw = props.get<std::string>(coef_names[i], "");

                bool is_init = raw != "";
                size_t node_size = cos_theta_str.size();

                // only need to extract non-default properties.
                if (is_init) {
                    std::vector<std::string> coef_str = string::tokenize(raw, " ,");

                    if (node_size != coef_str.size()) {
                        Throw("TabulatedPolarizedPhaseFunction: the provided "
                              "parameters must have the same size as 'cos_theta_str'!");
                    }

                    for (size_t j = 0; j < node_size; ++j) {
                        try {
                            ms_vec[i][j] = string::stof<ScalarFloat>(coef_str[j]);
                        } catch (...) {
                            Throw("Could not parse floating point value '%s'",
                                coef_str[i]);
                        }
                    }
                }
            }

            m_mvec = IrregularInterpolant<Float, 5>(
                cos_theta.data(), ms_vec, cos_theta.size()
            );
        }

        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.push_back(m_flags);
    }

    std::tuple<Vector3f, Spectrum, Float>
    sample(const PhaseFunctionContext &ctx, const MediumInteraction3f &mei,
           Float /* sample1 */, const Point2f &sample2,
           Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        // Sample a direction in physics convention.
        // We sample cos θ' = cos(π - θ) = -cos θ.
        Float cos_theta_prime = m_m11.sample(sample2.x());
        Float sin_theta_prime =
            dr::safe_sqrt(1.f - cos_theta_prime * cos_theta_prime);
        auto [sin_phi, cos_phi] =
            dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());
        Vector3f wo{ sin_theta_prime * cos_phi, sin_theta_prime * sin_phi,
                     cos_theta_prime };

        // Switch the sampled direction to graphics convention and transform the
        // computed direction to world coordinates
        wo = -mei.to_world(wo);

        auto [phase_val, phase_pdf] = eval_pdf(ctx, mei, wo, active);
        Spectrum phase_weight       = phase_val * dr::rcp(phase_pdf);

        return { wo, phase_weight, phase_pdf };
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext &ctx,
                                        const MediumInteraction3f &mei,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        // The data is laid out in physics convention
        // (with cos θ = 1 corresponding to forward scattering).
        // This parameterization differs from the convention used internally by
        // Mitsuba and is the reason for the minus sign below.
        Float cos_theta = -dot(wo, mei.wi);

        Spectrum phase_val(0.f);

        // Use m11 for the phase and PDF since it is used for sampling
        // cos theta.
        Float m11      = m_m11.eval_pdf(cos_theta, active);
        Float m11_norm = m_m11.normalization();
        Float pdf      = m11 * m11_norm * dr::InvTwoPi<ScalarFloat>;

        if constexpr (is_polarized_v<Spectrum>) {

            Vector5f ms = m_mvec.eval_data(cos_theta, active);
            phase_val =
                MuellerMatrix<Float>(m11,   ms[0], 0, 0,
                                     ms[0], ms[1], 0, 0,
                                     0, 0,  ms[2], ms[3],
                                     0, 0, -ms[3], ms[4]);

            phase_val *= m11_norm * dr::InvTwoPi<ScalarFloat>;

            /* Due to the coordinate system rotations for polarization-aware
                pBSDFs below we need to know the propagation direction of light.
                In the following, light arrives along `-wo_hat` and leaves along
                `+wi_hat`. */
            Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? wo : mei.wi,
                     wi_hat = ctx.mode == TransportMode::Radiance ? mei.wi : wo;

            /* The Stokes reference frame vector of this matrix lies in the
                scattering plane spanned by wi and wo. */
            Vector3f x_hat      = dr::cross(-wo_hat, wi_hat),
                     p_axis_in  = dr::normalize(dr::cross(x_hat, -wo_hat)),
                     p_axis_out = dr::normalize(dr::cross(x_hat, wi_hat));

            /* Rotate in/out reference vector of weight s.t. it aligns with the
            implicit Stokes bases of -wo_hat & wi_hat. */
            phase_val = mueller::rotate_mueller_basis(
                phase_val, -wo_hat, p_axis_in, mueller::stokes_basis(-wo_hat),
                wi_hat, p_axis_out, mueller::stokes_basis(wi_hat));

            // If the cross product x_hat is too small, phase_val may be NaN
            dr::masked(phase_val, dr::isnan(phase_val)) =
                depolarizer<Spectrum>(0.f);
        } else {
            phase_val = Spectrum(m11) * m11_norm * dr::InvTwoPi<ScalarFloat>;
            ;
        }

        return { phase_val, pdf };
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("m11", m_m11.pdf(), ParamFlags::NonDifferentiable);
        cb->put("m12", m_mvec.data(0), ParamFlags::NonDifferentiable);
        cb->put("m22", m_mvec.data(1), ParamFlags::NonDifferentiable);
        cb->put("m33", m_mvec.data(2), ParamFlags::NonDifferentiable);
        cb->put("m34", m_mvec.data(3), ParamFlags::NonDifferentiable);
        cb->put("m44", m_mvec.data(4), ParamFlags::NonDifferentiable);
        cb->put("nodes", m_m11.nodes(), ParamFlags::NonDifferentiable);
    }

    void
    parameters_changed(const std::vector<std::string> & /*keys*/) override {
        m_m11.update();
        m_mvec.nodes() = m_m11.nodes();
        m_mvec.update();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "TabulatedPolarizedPhaseFunction[" << std::endl
            << "  distr = " << string::indent(m_m11) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(TabulatedPolarizedPhaseFunction)
private:
    // m11 of the mueller matrix, used to sample the phase function
    IrregularContinuousDistribution<Float> m_m11;
    // rest of the relevant mueller matrix' terms.
    IrregularInterpolant<Float, 5> m_mvec;
};

MI_EXPORT_PLUGIN(TabulatedPolarizedPhaseFunction)
NAMESPACE_END(mitsuba)
