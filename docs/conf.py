"""Sphinx configuration for the plantbench documentation.

    pixi run docs
    uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html
    .venv/bin/pip install -e ".[docs]" && .venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
"""

from importlib.metadata import version as _version

project = "plantbench"
author = "Kyle Territo, Luis A. Briceno-Mena and Jose A. Romagnoli"
copyright = f"2026, {author}"
release = _version("plantbench")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",
]

# autodoc2 writes its own index; the API page in the toctree is api.md.
exclude_patterns = ["_build", "apidocs/index.rst", "_installation.md"]

# The guides and the case cards are Markdown, read by GitHub as well as here.
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# The API reference is read from the source without importing it, so the optional
# dependencies of plantbench.task need not be installed.  The docstrings are written in
# Markdown (backticks for code, indented code blocks) and are parsed as such.
autodoc2_packages = [
    {
        "path": "../plantbench",
        "exclude_dirs": ["__pycache__", "_template"],
        "exclude_files": ["__main__.py"],
    },
]
autodoc2_docstring_parser_regexes = [(r".*", "myst")]
autodoc2_hidden_objects = ["private", "dunder", "inherited"]
autodoc2_render_plugin = "myst"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# The theme and options of the SHAP documentation; custom.css adds a dark scheme, which
# sphinx_rtd_theme does not have, and follows the reader's system setting.
html_theme = "sphinx_rtd_theme"
html_title = f"plantbench {release}"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/logo-dark.svg"
html_favicon = "_static/logo-mark.svg"
html_theme_options = {
    "logo_only": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#343131",
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
pygments_style = "sphinx"
