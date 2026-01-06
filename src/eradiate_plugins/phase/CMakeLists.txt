set(MI_PLUGIN_PREFIX "phasefunctions")

add_plugin(rayleigh_polarized rayleigh_polarized.cpp)
add_plugin(tabphase_irregular tabphase_irregular.cpp)
add_plugin(tabphase_polarized tabphase_polarized.cpp)
add_plugin(multiphase multiphase.cpp)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
