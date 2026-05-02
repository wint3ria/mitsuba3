#include <algorithm>

#include <mitsuba/core/properties.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/volume.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-multiphase:

Multi phase function (:monosp:`multiphase`)
-------------------------------------------

.. pluginparameters::

 * - weight_i
   - |volume|
   - Weight for phase function i, where i ranges from 0 to N-2 for N phase functions.
     All weights are normalized internally, so they don't need to sum to 1.
   - |exposed|, |differentiable|

 * - use_mis
   - |bool|
   - Enable Multiple Importance Sampling (MIS) for variance reduction. When enabled,
     all phase functions are evaluated at each sampled direction to compute the 
     mixture PDF and importance weight. This reduces variance when mixing very 
     different phase functions at the cost of additional computation. (Default: true)
   - 

 * - (Nested plugin)
   - |phase|
   - Two or more nested phase function instances to be mixed according to their
     respective weights. The phase functions are identified by order of appearance.
   - |exposed|, |differentiable|

This plugin implements a generalized multi-component phase function mixture that
extends the standard two-phase blend to support an arbitrary number N ≥ 2 of phase
functions. This is particularly useful for modeling complex participating media with
multiple populations of scattering elements.

**Key differences from standard** :monosp:`blendphase`:
 * Supports N phases instead of just 2
 * Weights are automatically normalized (don't need to sum to 1)
 * Last weight is not inferred automatically
 * Optional Multiple Importance Sampling for variance reduction

.. tabs::
    .. code-tab:: xml

        <phase type="multiphase">
            <boolean name="use_mis" value="true"/>
            
            <phase name="phase_0" type="isotropic"/>
            <float name="weight_0" value="1.0"/>
            <phase name="phase_1" type="hg">
                <float name="g" value="0.5"/>
            </phase>
            <float name="weight_1" value="1.0"/>
            <phase name="phase_2" type="hg">
                <float name="g" value="0.2"/>
            </phase>
            <float name="weight_2" value="1.0"/>
        </phase>

    .. code-tab:: python

        'type': 'multiphase',
        'use_mis': True,
        'phase_0': {'type': 'isotropic'},
        'weight_0': 1.0,
        'phase_1': {'type': 'hg', 'g': 0.5},
        'weight_1': 1.0,
        'phase_2': {'type': 'hg', 'g': 0.2},
        'weight_2': 1.0

*/

template <typename Float, typename Spectrum>
class MultiPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext, Volume)

    MultiPhaseFunction(const Properties &props) : Base(props) {
        m_mis = props.get<bool>("use_mis", true);

        size_t phase_count = 0;
        for (auto &prop : props.objects()) {
            if (Base *phase = prop.try_get<Base>()) {
                m_nested_phases.push_back(phase);
                m_weights.push_back(props.get_volume<Volume>("weight_" + std::to_string(phase_count)));
                phase_count++;
            }
        }

        if (phase_count < 2)
            Throw("MultiPhase: At least 2 child phase functions must be specified!");

        m_nested_phases_index.reserve(m_nested_phases.size() + 1);
        m_nested_phases_index.push_back(0);
        m_components.clear();
        for (size_t i = 0; i < phase_count; ++i) {
            for (size_t j = 0; j < m_nested_phases[i]->component_count(); ++j) {
                m_components.push_back(m_nested_phases[i]->flags(j));
            }
            m_nested_phases_index.push_back(m_nested_phases[i]->component_count() + m_nested_phases_index.back());
        }

        m_flags = 0;
        for (size_t i = 0; i < phase_count; ++i)
            m_flags |= m_nested_phases[i]->flags();
    }

    void traverse(TraversalCallback *cb) override {
        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            cb->put("phase_" + std::to_string(i), m_nested_phases[i], ParamFlags::Differentiable);
            cb->put("weight_" + std::to_string(i), m_weights[i], ParamFlags::Differentiable);
        }
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext &ctx,
                                                 const MediumInteraction3f &mi,
                                                 Float sample1, const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        std::vector<Float> weight_values(m_nested_phases.size());
        Float weight_sum = 0.f;
        
        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            weight_values[i] = eval_weight(mi, i, active);
            weight_sum += weight_values[i];
        }
        
        Float inv_weight_sum = dr::rcp(weight_sum);

        if (unlikely(ctx.component != (uint32_t) -1)) {
            PhaseFunctionContext ctx2(ctx);
            
            auto position = std::upper_bound(
                m_nested_phases_index.begin(),
                m_nested_phases_index.end(), 
                ctx.component
            );
            size_t phase_index = std::distance(m_nested_phases_index.begin(), position) - 1;
            
            ctx2.component = ctx.component - m_nested_phases_index[phase_index];
            
            auto [wo, w, pdf] = m_nested_phases[phase_index]->sample(
                ctx2, mi, sample1, sample2, active
            );
            
            Float phase_weight = weight_values[phase_index] * inv_weight_sum;
            return { wo, w * phase_weight, pdf * phase_weight };
        }

        Vector3f wo = dr::zeros<Vector3f>();
        Spectrum w = dr::zeros<Spectrum>();
        Float pdf = dr::zeros<Float>();
        
        Float cdf_last = 0.f;
        
        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            Float cdf_next = cdf_last + weight_values[i] * inv_weight_sum;
            Mask mask_i = active && sample1 >= cdf_last && sample1 < cdf_next;
            
            if (dr::any_or<true>(mask_i)) {
                Float sample1_adjusted = (sample1 - cdf_last) / (cdf_next - cdf_last);
                
                auto [wo_i, w_i, pdf_i] = m_nested_phases[i]->sample(
                    ctx, mi, sample1_adjusted, sample2, mask_i
                );
                if(unlikely(!m_mis)) {
                    dr::masked(wo, mask_i) = wo_i;
                    dr::masked(w, mask_i) = w_i;
                    dr::masked(pdf, mask_i) = pdf_i;
                } else {
                    Spectrum phase_value_sum = w_i * pdf_i * weight_values[i];
                    Float pdf_mixture = pdf_i * weight_values[i];
                    
                    for (size_t j = 0; j < m_nested_phases.size(); ++j) {
                        if (j == i) continue;

                        // Ensure phases with zero weights are not called at all
                        Mask mask_j = mask_i && (weight_values[j] > dr::Epsilon<Float>);
                        if (!dr::any_or<true>(mask_j)) continue;
                        
                        auto [val_j, pdf_j] = m_nested_phases[j]->eval_pdf(
                            ctx, mi, wo_i, mask_i
                        );
                        
                        phase_value_sum += weight_values[j] * val_j;
                        pdf_mixture += weight_values[j] * pdf_j;
                    }
                    
                    Spectrum w_mis = dr::select(
                        pdf_mixture > dr::Epsilon<Float>,
                        phase_value_sum / pdf_mixture,
                        Spectrum(0.f)
                    );
                    
                    dr::masked(wo, mask_i) = wo_i;
                    dr::masked(w, mask_i) = w_mis;
                    dr::masked(pdf, mask_i) = pdf_mixture * inv_weight_sum;
                }
            }
            
            cdf_last = cdf_next;
        }

        return { wo, w, pdf };
    }

    MI_INLINE Float eval_weight(const MediumInteraction3f &mi,
                                const size_t index,
                                const Mask &active) const {
        return m_weights[index]->eval_1(mi, active);
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext &ctx,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        std::vector<Float> weight_values;
        weight_values.reserve(m_nested_phases.size());
        Float weight_sum = 0.f;
        
        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            weight_values.push_back(eval_weight(mi, i, active));
            weight_sum += weight_values.back();
        }
        
        Float inv_weight_sum = dr::select(weight_sum < dr::Epsilon<Float>, 1.0f, dr::rcp(weight_sum));

        if (unlikely(ctx.component != (uint32_t) -1)) {
            PhaseFunctionContext ctx2(ctx);

            auto position = std::upper_bound(
                m_nested_phases_index.begin(),
                m_nested_phases_index.end(), 
                ctx.component
            );
            size_t phase_index = std::distance(m_nested_phases_index.begin(), position) - 1;
            
            ctx2.component = ctx.component - m_nested_phases_index[phase_index];
            
            auto [val, pdf] = m_nested_phases[phase_index]->eval_pdf(ctx2, mi, wo, active);
            
            Float phase_weight = weight_values[phase_index] * inv_weight_sum;
            return { val * phase_weight, pdf * phase_weight };
        }

        Spectrum val = 0.f;
        Float pdf = 0.f;

        for (size_t i = 0; i < m_nested_phases.size(); ++i) {

            //Ensure zero weighted phases are never called
            Mask mask_i = active && (weight_values[i] > dr::Epsilon<Float>);
            if (!dr::any_or<true>(mask_i)) continue;
            
            auto [val_i, pdf_i] = m_nested_phases[i]->eval_pdf(ctx, mi, wo, active);
            
            Float phase_weight = weight_values[i] * inv_weight_sum;
            val += val_i * phase_weight;
            pdf += pdf_i * phase_weight;
        }
        
        return { val, pdf };
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MultiPhaseFunction[" << std::endl;
        for (size_t i = 0; i < m_weights.size(); ++i) {
            oss << "  weight[" << i << "] = " << string::indent(m_weights[i]);
            oss << std::endl;
        }
        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            oss << "  nested_phase[" << i << "] = " << string::indent(m_nested_phases[i]);
            if (i < m_nested_phases_index.size() - 1)
                oss << std::endl;
        }
        oss << "  use_mis = " << std::to_string(m_mis) << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(MultiPhaseFunction)
protected:
    std::vector<ref<Volume>> m_weights;
    std::vector<ref<Base>> m_nested_phases;
    std::vector<uint32_t> m_nested_phases_index;
    bool m_mis;
};

MI_EXPORT_PLUGIN(MultiPhaseFunction)
NAMESPACE_END(mitsuba)
