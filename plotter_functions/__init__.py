# -*- coding: utf-8 -*-

"""
This module contains the functions that are used to plot quantities of interest
"""

# import the functions
from .category import _divide_en_categ, _divide_eta_categ
from .checkFractionCEH import checkFractionCEH
from .checkShowerExt import checkShowerExt, findMinima
from .energy_profile_cat import _energy_profile_per_cat
from .energyProfile import _energy_profile
from .plotENprofile import doENprofile
from .plotFractionCEH import plotFractionCEH
from .plotGun import doGunPlots
from .plotHisto import doHisto
from .plotMultiplicity import doMultiplicityPlots
from .plotMultiplicity_cat import doMultiplicityPlots_cat
from .plotVisualization import doVisualizationPlots
from .traincheck_pi_all import plot_traincheck_all, traincheck_all
from .traincheck_pi_CEEshowerExt import (plot_showerExtCEE,
                                         traincheck_showerExtCEE)
from .traincheck_pi_enFracCEE import plot_enFracCEE, traincheck_enFracCEE

# Set the license and package information
__license__ = ''
__url__ = ''

# package import
__all__ = ['_divide_eta_categ',
           '_divide_en_categ',
           'doHisto',
           '_energy_profile',
           '_energy_profile_per_cat',
           'doENprofile',
           'doGunPlots',
           'doMultiplicityPlots',
           'doMultiplicityPlots_cat',
           'doVisualizationPlots',
           'checkShowerExt',
           'findMinima',
           'checkFractionCEH',
           'plotFractionCEH',
           'traincheck_showerExtCEE',
           'plot_showerExtCEE',
           'traincheck_enFracCEE',
           'plot_enFracCEE',
           'traincheck_all',
           'plot_traincheck_all'
        ]
