"""Sphinx configuration for astrosylva."""

from __future__ import annotations

project = "astrosylva"
author = "astrosylva contributors"
project_copyright = "2026, astrosylva contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "astrosylva"
# Logo lives at the repo root in assets/; Sphinx resolves the path
# relative to this conf.py and copies the file into the built _static.
html_logo = "../assets/astrosylva.png"
html_theme_options = {
    "sidebar_hide_name": True,  # the logo already says the project name
}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
