import drjit as dr
import mitsuba as mi


def test01_create(variant_scalar_rgb):
    phase = mi.load_dict(
        {
            "type": "multiphase",
            "phase1": {"type": "isotropic"},
            "phase2": {"type": "isotropic"},
            "weight0": 0.2,
            "weight1": 0.8,
        }
    )
    assert phase is not None
    assert phase.flags() == int(mi.PhaseFunctionFlags.Isotropic)

    phase = mi.load_dict(
        {
            "type": "multiphase",
            "phase1": {"type": "isotropic"},
            "phase2": {"type": "hg"},
            "weight0": 0.2,
            "weight1": 0.8,
        }
    )
    assert phase is not None
    assert phase.component_count() == 2
    assert phase.flags(0) == int(mi.PhaseFunctionFlags.Isotropic)
    assert phase.flags(1) == int(mi.PhaseFunctionFlags.Anisotropic)
    assert (
        phase.flags() == mi.PhaseFunctionFlags.Isotropic | mi.PhaseFunctionFlags.Anisotropic
    )


def test02_eval_all(variant_scalar_rgb):
    weight = 0.2
    g = 0.2

    phase = mi.load_dict(
        {
            "type": "multiphase",
            "phase1": {"type": "isotropic"},
            "phase2": {"type": "hg", "g": g},
            "weight0": weight,
            "weight1": 1 - weight,
            "use_mis": False,
        }
    )

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    wo = [0, 0, 1]
    ctx = mi.PhaseFunctionContext()

    # Evaluate the blend of both components
    expected = weight * dr.inv_four_pi + (1-weight) * dr.inv_four_pi * (1.0 - g) / (
        1.0 + g
    ) ** 2
    value = phase.eval_pdf(ctx, mei, wo)[0]
    assert dr.allclose(value, expected)


def test03_sample_all(variants_all_rgb):
    weight = 0.8
    g = 0.2

    phase = mi.load_dict(
        {
            "type": "multiphase",
            "phase_1": {"type": "isotropic"},
            "phase_2": {"type": "hg", "g": g},
            "weight0": weight,
            "weight1": 1 - weight,
            "use_mis": False,
        }
    )

    print(phase)

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    ctx = mi.PhaseFunctionContext()

    # Sample using two different values of 'sample1' and make sure correct
    # components are chosen.

    # -- Sample below weight: first component (isotropic) is selected
    expected_a = dr.inv_four_pi
    wo_a, w_a, pdf_a = phase.sample(ctx, mei, 0.3, [0.5, 0.5])
    assert dr.allclose(pdf_a, expected_a)

    # -- Sample above weight: second component (HG) is selected
    expected_b = dr.inv_four_pi * (1 - g) / (1 + g) ** 2
    wo_b, w_b, pdf_b = phase.sample(ctx, mei, 0.9, [0, 0])
    assert dr.allclose(pdf_b, expected_b)


def test04_eval_components(variant_scalar_rgb):
    weight = 0.2
    g = 0.2

    phase = mi.load_dict(
        {
            "type": "multiphase",
            "phase1": {"type": "isotropic"},
            "phase2": {"type": "hg", "g": g},
            "weight0": weight,
            "weight1": 1 - weight,
        }
    )

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    wo = [0, 0, 1]
    ctx = mi.PhaseFunctionContext()

    # Evaluate the two components separately

    ctx.component = 0
    value0, pdf0 = phase.eval_pdf(ctx, mei, wo)
    expected0 = weight * dr.inv_four_pi
    assert dr.allclose(value0, expected0)
    assert dr.allclose(value0, pdf0)

    ctx.component = 1
    value1, pdf1 = phase.eval_pdf(ctx, mei, wo)
    expected1 = (1-weight) * dr.inv_four_pi * (1.0 - g) / (1.0 + g) ** 2
    assert dr.allclose(value1, expected1)
    assert dr.allclose(value1, pdf1)


def test05_sample_components(variant_scalar_rgb):
    weight = 0.2
    g = 0.2

    phase = mi.load_dict(
        {
            "type": "multiphase",
            "phase1": {"type": "isotropic"},
            "phase2": {"type": "hg", "g": g},
            "weight0": weight,
            "weight1": 1 - weight,
        }
    )

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    ctx = mi.PhaseFunctionContext()

    # Sample using two different values of 'sample1' and make sure correct
    # components are chosen.

    # -- Select component 0: first component is always sampled
    ctx.component = 0

    expected_a = weight * dr.inv_four_pi
    wo_a, w_a, pdf_a = phase.sample(ctx, mei, 0.3, [0.5, 0.5])
    assert dr.allclose(pdf_a, expected_a)

    expected_b = weight * dr.inv_four_pi
    wo_b, w_b, pdf_b = phase.sample(ctx, mei, 0.1, [0.5, 0.5])
    assert dr.allclose(pdf_b, expected_b)

    # -- Select component 1: second component is always sampled
    ctx.component = 1

    expected_a = (1 - weight) * dr.inv_four_pi * (1 - g) / (1 + g) ** 2
    wo_a, w_a, pdf_a = phase.sample(ctx, mei, 0.3, [0.0, 0.0])
    assert dr.allclose(pdf_a, expected_a)

    expected_b = (1 - weight) * dr.inv_four_pi * (1 - g) / (1 + g) ** 2
    wo_b, w_b, pdf_b = phase.sample(ctx, mei, 0.1, [0.0, 0.0])
    assert dr.allclose(pdf_b, expected_b)


def test06_sample_all_mis(variant_scalar_rgb):

    g2, g3 = 0.2, 0.3
    w0, w1, w2 = 10.0, 20.0, 30.0
    W = w0 + w1 + w2
    phase = mi.load_dict(
        {
            "type": "multiphase",
            "iso1": {"type": "isotropic"},
            "hg2": {"type": "hg", "g": g2},
            "hg3": {"type": "hg", "g": g3},
            "weight0": w0,
            "weight1": w1,
            "weight2": w2,
        }
    )
    iso = mi.load_dict({"type": "isotropic"})
    hg1 = mi.load_dict({"type": "hg", "g": g2})
    hg2 = mi.load_dict({"type": "hg", "g": g3})

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    ctx = mi.PhaseFunctionContext()
    
    sample1 = 0.1 # First component selected

    wo, w, pdf = phase.sample(ctx, mei, sample1, [0.5, 0.5])
    sample1_adjusted = sample1 / (w0 / W)
    wo_iso, val_iso, pdf_iso = iso.sample(ctx, mei, sample1_adjusted, [0.5, 0.5])
    val_hg1, pdf_hg1 = hg1.eval_pdf(ctx, mei, wo_iso)
    val_hg2, pdf_hg2 = hg2.eval_pdf(ctx, mei, wo_iso)

    pdf_mixture = (pdf_iso * w0 + pdf_hg1 * w1 + pdf_hg2 * w2)
    pdf_expected = pdf_mixture / W
    val_mixture = (val_iso * pdf_iso * w0 + val_hg1 * w1 + val_hg2 * w2)
    val_expected = val_mixture / pdf_mixture

    assert dr.allclose(pdf, pdf_expected)
    assert dr.allclose(w, val_expected)
    assert dr.allclose(wo, wo_iso)

    sample2 = 0.3 # Second component selected

    wo, w, pdf = phase.sample(ctx, mei, sample2, [0.5, 0.5])
    sample2_adjusted = (sample2 - w0 / W) / (w1 / W)
    wo_hg1, val_hg1, pdf_hg1 = hg1.sample(ctx, mei, sample2_adjusted, [0.5, 0.5])
    val_iso, pdf_iso = iso.eval_pdf(ctx, mei, wo_hg1)
    val_hg2, pdf_hg2 = hg2.eval_pdf(ctx, mei, wo_hg1)

    pdf_mixture = (pdf_iso * w0 + pdf_hg1 * w1 + pdf_hg2 * w2)
    pdf_expected = pdf_mixture / W
    val_mixture = (val_iso * w0 + val_hg1 * pdf_hg1 * w1 + val_hg2 * w2)
    val_expected = val_mixture / pdf_mixture

    assert dr.allclose(pdf, pdf_expected)
    assert dr.allclose(w, val_expected)
    assert dr.allclose(wo, wo_hg1)

    sample3 = 0.9 # Last component selected

    wo, w, pdf = phase.sample(ctx, mei, sample3, [0.5, 0.5])
    sample3_adjusted = (sample3 - (w0 + w1) / W) / (w2 / W)
    wo_hg2, val_hg2, pdf_hg2 = hg2.sample(ctx, mei, sample3_adjusted, [0.5, 0.5])
    val_iso, pdf_iso = iso.eval_pdf(ctx, mei, wo_hg2)
    val_hg1, pdf_hg1 = hg1.eval_pdf(ctx, mei, wo_hg2)

    pdf_mixture = (pdf_iso * w0 + pdf_hg1 * w1 + pdf_hg2 * w2)
    pdf_expected = pdf_mixture / W
    val_mixture = (val_iso * w0 + val_hg1 * w1 + val_hg2 * pdf_hg2 * w2)
    val_expected = val_mixture / pdf_mixture

    assert dr.allclose(pdf, pdf_expected)
    assert dr.allclose(w, val_expected)
    assert dr.allclose(wo, wo_hg2)


def test_compare_blendphase(variant_scalar_rgb):

    weight = 0.2
    g = 0.2

    phase_0 = mi.load_dict({"type": "isotropic"})
    phase_1 = mi.load_dict({"type": "hg", "g": g})

    blendphase = mi.load_dict(
        {
            "type": "blendphase",
            "phase1": phase_0,
            "phase2": phase_1,
            "weight": weight,
        }
    )

    multiphase = mi.load_dict(
        {
            "type": "multiphase",
            "phase1": phase_0,
            "phase2": phase_1,
            "weight0": 1 - weight,
            "weight1": weight,
        }
    )

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    ctx = mi.PhaseFunctionContext()

    wo_a1, w_a1, pdf_a1 = blendphase.sample(ctx, mei, 0.3, [0.5, 0.5])
    wo_a2, w_a2, pdf_a2 = multiphase.sample(ctx, mei, 0.7, [0.5, 0.5])

    wo_b1, w_b1, pdf_b1 = blendphase.sample(ctx, mei, 0.1, [0, 0])
    wo_b2, w_b2, pdf_b2 = multiphase.sample(ctx, mei, 0.9, [0, 0])

    assert dr.allclose(wo_a1, wo_a2)
    assert dr.allclose(w_a1, w_a2)
    assert dr.allclose(pdf_a1, pdf_a2)

    assert dr.allclose(wo_b1, wo_b2)
    assert dr.allclose(w_b1, w_b2)
    assert dr.allclose(pdf_b1, pdf_b2)


def test08_chi2_isotropic_hg(variants_vec_backends_once_rgb):
    from mitsuba.chi2 import PhaseFunctionAdapter, ChiSquareTest, SphericalDomain

    sample_func, pdf_func = PhaseFunctionAdapter("multiphase", {
        "type": "multiphase",
        "phase1": {"type": "isotropic"},
        "phase2": {"type": "hg", "g": 0.2},
        "weight0": 0.8,
        "weight1": 0.2,
    })

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3
    )

    assert chi2.run()


def test09_chi2_hg_rayleigh(variants_vec_backends_once_rgb):
    from mitsuba.chi2 import PhaseFunctionAdapter, ChiSquareTest, SphericalDomain

    sample_func, pdf_func = PhaseFunctionAdapter("multiphase", {
        "type": "multiphase",
        "phase1": {"type": "hg", "g": 0.2},
        "phase2": {"type": "rayleigh"},
        "weight0": 0.6,
        "weight1": 0.4,
    })

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3
    )

    assert chi2.run()