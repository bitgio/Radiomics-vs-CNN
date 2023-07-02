# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# DEVE PUNTARE ALLA CARTELLA DOVE C'È IL CODICE PYTHON, 
# 1 cartella indietro <=> cd ../   ; 2 cartelle indietro <=> cd ../.. 
sys.path.insert(0, os.path.abspath('../RadCNN'))

project = 'Radiomics vs CNN'

sys.path.insert(0, os.path.abspath('../RadCNN/senza_filtro'))

copyright = '2023, Giovanni Bitonti, Ana Pascual'
author = 'Giovanni Bitonti, Ana Pascual'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = []
