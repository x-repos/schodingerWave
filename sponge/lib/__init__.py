# qwave-sponge: sponge-layer (complex absorbing potential) boundaries for
# explicit-circuit quantum simulation of acoustic and elastic waves.
#
# op_library.py, evolution.py, absorbing.py, schrodingerisation.py are
# vendored byte-identical from https://github.com/x-repos/qawave-cpml @ a13b82b
# (absorbing.py also carries the CPML classes; only the sponge, the
# separated-block op strings and the profile helpers are used here).
# elastic.py, elastic_separated.py are vendored byte-identical from
# qwave-elastic @ 3038d99; elastic_sponge.py is ElasticSponge2d extracted
# from qwave-elastic/elib/elastic_cpml.py.
from .op_library import *
from .evolution import *
from .absorbing import (
    pml_sigma_profile,
    split_generator,
    WaveEquationCPML1d,          # 1D collapsed form = diagonal damping (used for 1D gate counts)
    WaveEquationSponge2d,
    WaveEquationSeparatedEvolution2d,
)
from .schrodingerisation import Schrodingerisation, warping_profile
from .elastic import ElasticWaveEvolution2d, isotropic_voigt_sqrt
from .elastic_separated import ElasticSeparatedEvolution2d
from .elastic_sponge import ElasticSponge2d
from .sponge1d import WaveEquationSponge1d
