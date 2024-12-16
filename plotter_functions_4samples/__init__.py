# -*- coding: utf-8 -*-

"""
This module contains the functions that are used to plot quantities of interest
"""

# import the functions
from .category import _divide_en_categ, _divide_eta_categ
from .plotGun import doGunPlots
from .plotHisto import doHisto

# Set the license and package information
__license__ = ''
__url__ = ''

# package import
__all__ = ['_divide_eta_categ',
           '_divide_en_categ',
           'doGunPlots',
           'doHisto'
        ]
