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

        
        m_mis = props.get<bool>("use_mis", true);

        size_t phase_count = 0;
        for (auto &prop : props.objects()) {
            if (Base *phase = prop.try_get<Base>()) {
                m_nested_phases.push_back(phase);
                m_weights.push_back(props.get_volume<Volume>("weight" + std::to_string(phase_count)));
                phase_count++;
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
            cb->put("phase" + std::to_string(i), m_nested_phases[i], ParamFlags::Differentiable);
            cb->put("weight" + std::to_string(i), m_weights[i], ParamFlags::Differentiable);
        }
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext &ctx,
                                                 const MediumInteraction3f &mi,
                                                 Float sample1, const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        std::vector<Float> weight_values(m_nested_phases.size());
        std::vector<Float> weight_index(m_nested_phases.size() + 1);
        Vector3f wo = 0.f,  wo_i;
        Spectrum w = 0.f,   w_i;
        Float    pdf = 0.f, pdf_i;
        Spectrum val_j;
        Float pdf_j;
        Mask M_i;
        Float weight_sum = 0.f, inv_weight_sum;
        Float cdf_last = 0.f, cdf_next;
        Float sample1_adjusted = 0.f;                
        Spectrum pha_mixture = 0.f;
        Spectrum w_mis;
        Float pdf_mixture = 0.f;
        size_t p_index;

        weight_index[0] = 0.f;
        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            weight_values[i] = eval_weight(mi, i, active);
            weight_sum += weight_values[i];
            weight_index[i + 1] = weight_sum;
        }
        inv_weight_sum = 1.f / weight_sum;

        std::cout << "start"<< std::endl;

        if (unlikely(ctx.component != (uint32_t) -1)) {
            PhaseFunctionContext ctx2(ctx);
            const std::vector<uint32_t>::const_iterator position = std::upper_bound(
                m_nested_phases_index.begin(),
                m_nested_phases_index.end(), 
                ctx.component
            );
            p_index = std::distance(m_nested_phases_index.begin(), position) - 1;
            ctx2.component = ctx.component - m_nested_phases_index[p_index];
            std::tie(wo, w, pdf) = m_nested_phases[p_index]->sample(
                ctx2, mi, sample1, sample2, active);
            const Float phase_weight = weight_values[p_index] * inv_weight_sum;
            return {wo, w * phase_weight, pdf * phase_weight};
        }

        /*const std::vector<Float>::const_iterator position = std::upper_bound(
            weight_index.begin(),
            weight_index.end(), 
            sample1 * weight_sum,
        );
        index = std::distance(weight_index.begin(), position) - 1;*/

        //cdf_last = weight_index[w_index    ] * inv_weight_sum;
        //cdf_next = weight_index[w_index + 1] * inv_weight_sum;
        
        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            cdf_next = weight_index[i+1] * inv_weight_sum;
            std::cout << "cdf_last: " << cdf_last << std::endl;
            std::cout << "cdf_next: " << cdf_next << std::endl;
            std::cout << "sample1: " << sample1 << std::endl;
            M_i = active && sample1 >= cdf_last && sample1 < cdf_next;
            std::cout << i << " " << M_i << std::endl;
            if (dr::any_or<true>(M_i)) {
                std::cout << "Selected phase: " << i << std::endl;

                std::cout << "(" << cdf_last << ", " << cdf_next << ")" << std::endl;
                std::cout << M_i << std::endl;

                dr::masked(sample1_adjusted, M_i) = (sample1 - cdf_last) / (cdf_next - cdf_last);


                std::cout << "sample1_adjusted: " << sample1_adjusted << std::endl;

                std::tie(wo_i, w_i, pdf_i) = m_nested_phases[i]->sample(ctx, mi, sample1_adjusted, sample2, M_i);

                std::cout << "wo_i: " << wo_i << std::endl;
                std::cout << "w_i: " << w_i << std::endl;
                std::cout << "pdf_i: " << pdf_i << std::endl;
                
                if(unlikely(!m_mis)) {

                    std::cout << "no mis" << std::endl;

                    dr::masked(wo,  M_i) = wo_i;
                    dr::masked(w,   M_i) = w_i;
                    dr::masked(pdf, M_i) = pdf_i;
                    cdf_last = cdf_next;
                    continue;
                }

                pha_mixture = w_i * pdf_i * weight_values[i];
                pdf_mixture = pdf_i * weight_values[i];
                for (size_t j = 0; j < m_nested_phases.size(); ++j) {
                    if (i == j) continue;
                    std::tie(val_j, pdf_j) = m_nested_phases[j]->eval_pdf(ctx, mi, wo_i, M_i);
                    std::cout << "val_j: " << val_j << std::endl;
                    std::cout << "pdf_j: " << pdf_j << std::endl;
                    std::cout << "weight_values[j]: " << weight_values[j] << std::endl;
                    pha_mixture += weight_values[j] * val_j;
                    pdf_mixture += weight_values[j] * pdf_j;
                }

                std::cout << "pha_mixture: " << pha_mixture << std::endl;
                std::cout << "pdf_mixture: " << pdf_mixture << std::endl;

                w_mis = dr::select(
                    pdf_mixture > 0.f,
                    pha_mixture / pdf_mixture,
                    Spectrum(0.f)
                );

                dr::masked(wo,  M_i) = wo_i;
                dr::masked(w,   M_i) = w_mis;
                dr::masked(pdf, M_i) = pdf_mixture * inv_weight_sum;
            }
            cdf_last = cdf_next;
        }

        std::cout << "end"<< std::endl;

        return {wo, w, pdf};
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
            << "  use_mis = " << std::to_string(m_mis) << "," << std::endl;
        for (size_t i = 0; i < m_nested_phases.size(); ++i) {
            oss << "  nested_phase[" << i << "] = " << string::indent(m_nested_phases[i]);
        }
        oss << std::endl
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