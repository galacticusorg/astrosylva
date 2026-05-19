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

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
