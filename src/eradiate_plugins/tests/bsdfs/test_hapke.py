import drjit as dr
import mitsuba as mi
import numpy as np
import xarray as xr
import pytest

from os.path import join

import matplotlib.pyplot as plt

from ..tools import sample_eval_pdf_bsdf
from mitsuba.scalar_rgb.test.util import find_resource


@pytest.fixture
def static_reference():

    references = find_resource("src/eradiate_plugins/tests/references")

    hapke_reference_filename = join(references, "hapke_principal_plane_example.nc")

    return xr.load_dataset(hapke_reference_filename)


def test_create_hapke(variant_scalar_rgb):
    # Test constructor of 3-parameter version of RPV
    rtls = mi.load_dict({"type": "hapke"})

    assert isinstance(rtls, mi.BSDF)
    assert rtls.component_count() == 1
    assert rtls.flags(0) == mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
    assert rtls.flags() == rtls.flags(0)

    params = mi.traverse(rtls)
    assert "w.value" in params
    assert "b.value" in params
    assert "c.value" in params
    assert "theta.value" in params
    assert "B_0.value" in params
    assert "h.value" in params


def angles_to_directions(theta, phi):
    return mi.Vector3f(
        np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    )


def eval_bsdf(bsdf, wi, wo):
    si = mi.SurfaceInteraction3f()
    si.wi = wi
    ctx = mi.BSDFContext()
    return bsdf.eval(ctx, si, wo, True)[0]


def test_defaults_and_print(variant_scalar_mono):
    rtls = mi.load_dict({"type": "rtls"})
    value = str(rtls)
    reference = "\n".join(
        [
            "RTLSBSDF[",
            "  f_iso = UniformSpectrum[value=0.209741],",
            "  f_vol = UniformSpectrum[value=0.081384],",
            "  f_geo = UniformSpectrum[value=0.004140],",
            "  h = 2,",
            "  r = 1,",
            "  b = 1",
            "]",
        ]
    )
    assert reference == value


regression_test_geometries = [
    (0.18430089, 1.46592582, 3.92553451, 0.39280281),
]



def angles_to_directions(theta, phi):
    return mi.Vector3f(
        np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    )

def test_hapke_hotspot(variant_scalar_mono):
    c = 0.273
    c = (1 + c) / 2

    hapke = mi.load_dict({ # Pommerol et al. (2013)
        "type": "hapke",
        "w": 0.526,
        "theta": 13.3,
        "b": 0.187,
        "c": c,
        "h": 0.083,
        "B_0": 1.0,
    })

    theta_i = np.deg2rad(30.0)

    theta_o = np.deg2rad(30.0)

    phi_i = np.deg2rad(0.0)

    phi_o = np.deg2rad(0.0)

    wi = angles_to_directions(theta_i, phi_i)
    wo = angles_to_directions(theta_o, phi_o)

    values = np.asarray(eval_bsdf(hapke, wi, wo)) / np.abs(np.cos(theta_o))
    assert np.allclose(values, 0.24746648)


def test_hapke_grazing_viewing_angle(variant_scalar_mono):
    c = 0.273
    c = (1 + c) / 2

    hapke = mi.load_dict({ # Pommerol et al. (2013)
        "type": "hapke",
        "w": 0.526,
        "theta": 13.3,
        "b": 0.187,
        "c": c,
        "h": 0.083,
        "B_0": 1.0,
    })

    theta_i = np.deg2rad(30.0)
    theta_o = np.deg2rad(-89.0)
    phi_i = np.deg2rad(0.0)
    phi_o = np.deg2rad(0.0)

    wi = angles_to_directions(theta_i, phi_i)
    wo = angles_to_directions(theta_o, phi_o)

    values = np.asarray(eval_bsdf(hapke, wi, wo)) / np.abs(np.cos(theta_o))
    assert np.allclose(values, 0.15426355)


def test_hapke_backward(variant_scalar_mono):
    c = 0.273
    c = (1 + c) / 2

    hapke = mi.load_dict({ # Pommerol et al. (2013)
        "type": "hapke",
        "w": 0.526,
        "theta": 13.3,
        "b": 0.187,
        "c": c,
        "h": 0.083,
        "B_0": 1.0,
    })

    theta_i = np.deg2rad(30.0)
    theta_o = np.deg2rad(80.0)
    phi_i = np.deg2rad(0.0)
    phi_o = np.deg2rad(0.0)

    wi = angles_to_directions(theta_i, phi_i)
    wo = angles_to_directions(theta_o, phi_o)

    values = np.asarray(eval_bsdf(hapke, wi, wo)) / np.abs(np.cos(theta_o))
    assert np.allclose(values, 0.19555340)


def test_hapke_principal_plane(variant_llvm_mono_double):

    c = 0.273
    c = (1 + c) / 2

    hapke = mi.load_dict({ # Pommerol et al. (2013)
        "type": "hapke",
        "w": 0.526,
        "theta": 13.3,
        "b": 0.187,
        "c": c,
        "h": 0.083,
        "B_0": 1.0,
    })

    theta_i = np.deg2rad(30.0 * np.ones((180,)))
    theta_o = np.deg2rad(np.arange(90, -90, -1))
    phi_i = np.deg2rad(np.zeros((180,)))
    phi_o = np.deg2rad(np.zeros((180,)))

    wi = angles_to_directions(theta_i, phi_i)
    wo = angles_to_directions(theta_o, phi_o)

    values = np.asarray(eval_bsdf(hapke, wi, wo)) / np.abs(np.cos(theta_o))

    fig = plt.figure()

    plt.plot(np.rad2deg(theta_o), values)
    plt.grid()

    plt.savefig("test.png")
    plt.close()


def test_hapke_deviated_plane(variant_llvm_mono_double):

    c = 0.273
    c = (1 + c) / 2

    hapke = mi.load_dict({ # Pommerol et al. (2013)
        "type": "hapke",
        "w": 0.526,
        "theta": 13.3,
        "b": 0.187,
        "c": c,
        "h": 0.083,
        "B_0": 1.0,
    })

    theta_i = np.deg2rad(30.0 * np.ones((180,)))
    theta_o = np.deg2rad(np.arange(90, -90, -1))
    phi_i = np.deg2rad(30.0 * np.ones((180,)))
    phi_o = np.deg2rad(np.zeros((180,)))

    wi = angles_to_directions(theta_i, phi_i)
    wo = angles_to_directions(theta_o, phi_o)

    values = np.asarray(eval_bsdf(hapke, wi, wo)) / np.abs(np.cos(theta_o))

    fig = plt.figure()

    plt.plot(np.rad2deg(theta_o), values)
    plt.grid()

    plt.savefig("test3.png")
    plt.close()


def test_hapke_hemisphere(variant_llvm_mono_double):

    c = 0.273
    c = (1 + c) / 2

    hapke = mi.load_dict({ # Pommerol et al. (2013)
        "type": "hapke",
        "w": 0.526,
        "theta": 13.3,
        "b": 0.187,
        "c": c,
        "h": 0.083,
        "B_0": 1.0,
    })

    azimuths_s = 360
    zeniths_s = 90

    azimuths = np.radians(np.linspace(0, 360, azimuths_s))
    zeniths =  np.linspace(0, 89, zeniths_s)

    r, theta = np.meshgrid(np.sin(np.deg2rad(zeniths)), azimuths)
    values = np.random.random((azimuths.size, zeniths.size))

    theta_i = np.deg2rad(30.0 * np.ones((zeniths_s,))).reshape(-1, 1) @ np.ones((1, azimuths_s))
    theta_o = np.deg2rad(zeniths).reshape(-1, 1) @ np.ones((1, azimuths_s))
    phi_i = np.zeros((zeniths_s, azimuths_s))
    phi_o = np.ones((zeniths_s, 1)) @ azimuths.reshape(1, -1)

    wi = angles_to_directions(theta_i.reshape(1, -1), phi_i.reshape(1, -1))
    wo = angles_to_directions(theta_o.reshape(1, -1), phi_o.reshape(1, -1))

    values = np.asarray(eval_bsdf(hapke, wi, wo)) / np.abs(np.cos(theta_o)).reshape(1, -1)
    values = values.reshape((zeniths_s, azimuths_s)).T

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    contour = ax.contourf(theta, r, values, levels=25, cmap="turbo")
    plt.colorbar(contour)
    plt.grid(False)

    plt.savefig("test2.png")
    plt.close()


def test_hapke_static_reference(variant_llvm_mono_double, static_reference):

    hapke = mi.load_dict({ # Pommerol et al. (2013)
        "type": "hapke",
        "w": static_reference.w.item(),
        "theta": static_reference.theta.item(),
        "b": static_reference.b.item(),
        "c": static_reference.c.item(),
        "h": static_reference.h.item(),
        "B_0": static_reference.B0.item(),
    })

    theta_i = np.deg2rad(30.0 * np.ones((180,)))
    theta_o = np.deg2rad(static_reference.svza.values)
    phi_i = np.deg2rad(np.zeros((180,)))
    phi_o = np.deg2rad(np.zeros((180,)))

    wi = angles_to_directions(theta_i, phi_i)
    wo = angles_to_directions(theta_o, phi_o)

    values = np.asarray(eval_bsdf(hapke, wi, wo)) / np.abs(np.cos(theta_o))

    fig = plt.figure()
    plt.grid()
    plt.plot(np.rad2deg(theta_o), values, marker='+', label="Eradiate/Mitsuba")
    plt.plot(np.rad2deg(theta_o), static_reference.reflectance, label="Nguyen, Jacquemoud et al. (reference)")
    plt.legend()
    plt.savefig("test4.png")

    ref = static_reference.reflectance.values

    assert np.allclose(ref, values)


def test_chi2_hapke(variant_llvm_mono_double):
    from mitsuba.chi2 import BSDFAdapter, ChiSquareTest, SphericalDomain

    sample_func, pdf_func = BSDFAdapter("hapke",
    """
        <float name="w" value="0.526"/>
        <float name="theta" value="13.3"/>
        <float name="b" value="0.187"/>
        <float name="c" value="0.273"/>
        <float name="h" value="0.083"/>
        <float name="B_0" value="1"/>
    """)

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3,
    )

    assert chi2.run()

