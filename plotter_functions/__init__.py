# -*- coding: utf-8 -*-

"""
This module contains the functions that are used to plot quantities of interest
"""

# import the functions
from .category import _divide_en_categ, _divide_eta_categ
from .energyProfile import _energy_profile
from .openFiles import openFiles
from .plotHisto import doHisto

# Set the license and package information
__license__ = ''
__url__ = ''

# package import
__all__ = ['openFiles',
           '_divide_eta_categ',
           '_divide_en_categ',
           'doHisto',
           '_energy_profile'
        ]
