#include <algorithm>

#include <mitsuba/core/properties.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/volume.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MultiPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext, Volume)

    MultiPhaseFunction(const Properties &props) : Base(props) {

        size_t phase_count = 0;
        
        m_mis = false;

        for (auto &prop : props.objects()) {
            if (Base *phase = prop.try_get<Base>()) {
                m_nested_phases.push_back(phase);
                m_weights.push_back(props.get_volume<Volume>("weight" + std::to_string(phase_count)));
                phase_count++;
            }
            if (bool *mis = prop.try_get<bool>()) {
                m_mis = *mis;
            }
        }

        if (phase_count < 2)
            Throw("CumulativeBlendPhase: At least 2 child phase functions must be specified!");


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


    void eval_MIS() {
        if (false)  // TBD implement option in constructor
            return;

    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext &ctx,
                                                 const MediumInteraction3f &mi,
                                                 Float sample1, const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        using std::get;

        std::vector<Float> weight_values(m_nested_phases.size());
        std::vector<Float> weight_index(m_nested_phases.size() + 1);
        std::tuple<Vector3f, Spectrum, Float> result = { Vector3f(0.f), Spectrum(0.f), 0.f };
        Vector3f wo_i;
        Spectrum w_i;
        Float pdf_i;
        Spectrum val_j;
        Float pdf_j;
        Mask M_i;
        Float weight_sum = 0.f, inv_weight_sum;
        Float cdf_last = 0.f, cdf_next;
        Float sample1_adjusted = 0.f;                
        Spectrum phase_value_sum = 0.f;
        Float pdf_mixture = 0.f;

        weight_index[0] = 0.f;
        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            weight_values[i] = eval_weight(mi, i, active);
            weight_sum += weight_values[i];
            weight_index[i + 1] = weight_sum;
        }
        inv_weight_sum = 1.f / weight_sum;

        if (unlikely(ctx.component != (uint32_t) -1)) {
            PhaseFunctionContext ctx2(ctx);
            const std::vector<uint32_t>::const_iterator position = std::upper_bound(
                m_nested_phases_index.begin(),
                m_nested_phases_index.end(), 
                ctx.component
            );
            const size_t index = std::distance(m_nested_phases_index.begin(), position) - 1;
            ctx2.component = ctx.component - m_nested_phases_index[index];
            result = m_nested_phases[index]->sample(
                ctx2, mi, sample1, sample2, active);
            const Float phase_weight = weight_values[index] * inv_weight_sum;
            get<1>(result) *= phase_weight;
            get<2>(result) *= phase_weight;
            return result;
        }
        
        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            cdf_next = weight_index[i+1] * inv_weight_sum;
            M_i = active && sample1 >= cdf_last && sample1 < cdf_next;
            if (dr::any_or<true>(M_i)) {
                dr::masked(sample1_adjusted, M_i) = (sample1 - cdf_last) / (cdf_next - cdf_last);
                std::tie(wo_i, w_i, pdf_i) = m_nested_phases[i]->sample(ctx, mi, sample1_adjusted, sample2, M_i);
                
                if(unlikely(!m_mis)) {
                    dr::masked(get<0>(result), M_i) = wo_i;
                    dr::masked(get<1>(result), M_i) = w_i;
                    dr::masked(get<2>(result), M_i) = pdf_i;
                    continue;
                }

                for (size_t j = 0; j < m_nested_phases.size(); ++j) {
                    std::tie(val_j, pdf_j) = m_nested_phases[j]->eval_pdf(ctx, mi, wo_i, M_i);
                    
                    phase_value_sum += weight_values[j] * val_j * inv_weight_sum;
                    pdf_mixture     += weight_values[j] * pdf_j * inv_weight_sum;
                }

                Spectrum w_mis = dr::select(
                    pdf_mixture > 1e-8f,
                    phase_value_sum / pdf_mixture,
                    Spectrum(0.f)
                );

                dr::masked(get<0>(result), M_i) = wo_i;
                dr::masked(get<1>(result), M_i) = w_mis;
                dr::masked(get<2>(result), M_i) = pdf_mixture;
            }
            cdf_last = cdf_next;
        }

        return result;
    }

    MI_INLINE Float eval_weight(const MediumInteraction3f &mi,
                                const size_t index,
                                const Mask &active) const {
        return dr::clip(m_weights[index]->eval_1(mi, active), 0.f, 1.f);
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext &ctx,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        using std::get;

        Float weight_sum = 0.f, inv_weight_sum;
        std::vector<Float> weight_values;
        weight_values.reserve(m_nested_phases.size());
        std::pair<Spectrum, Float> result = { Spectrum(0.f), 0.f }, temp;

        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            weight_values.push_back(eval_weight(mi, i, active));
            weight_sum += weight_values.back();
        }
        inv_weight_sum = 1.f / weight_sum;

        if (unlikely(ctx.component != (uint32_t) -1)) {
            PhaseFunctionContext ctx2(ctx);
            const std::vector<uint32_t>::const_iterator position = std::upper_bound(
                m_nested_phases_index.begin(),
                m_nested_phases_index.end(), 
                ctx.component
            );
            const size_t index = std::distance(m_nested_phases_index.begin(), position) - 1;
            ctx2.component = ctx.component - m_nested_phases_index[index];
            result = m_nested_phases[index]->eval_pdf(
                ctx2, mi, wo, active);
            const Float phase_weight = weight_values[index] * inv_weight_sum;

            get<0>(result) *= phase_weight;
            get<1>(result) *= phase_weight;
            return result;
        }

        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            temp = m_nested_phases[i]->eval_pdf(ctx, mi, wo, active);
            const Float phase_weight = weight_values[i] * inv_weight_sum;
            dr::masked(get<0>(result), active) += get<0>(temp) * phase_weight;
            dr::masked(get<1>(result), active) += get<1>(temp) * phase_weight;
        }

        return result;
    }

    std::string to_string() const override {
        std::ostringstream oss;
         oss << "MultiPhaseFunction[" << std::endl
            << "  weights = " << string::indent(m_weights) << "," << std::endl
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