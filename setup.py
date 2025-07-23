from pathlib import Path
from setuptools import setup, find_packages

SHORT_DESCRIPTION = """LensKit-Auto is built as a wrapper around the Python LensKit recommender-system library.
                    It automates algorithm selection and hyper parameter optimization an can build ensemble models based on the LensKit models."""

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="lenskit-auto",
    version="0.2.0",
    author="Tobias Vente",
    python_requires=">=3.12, <3.13",
    packages=find_packages(),
    install_requires=[
        "smac>=2.3.1",
        "lenskit>=2025.2.0",
        "numpy>=2.2.6",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.1",
        "scipy>=1.15.3",
        "numba>=0.61.2",
        "typing-extensions>=4.13.2",
        "matplotlib>=3.10.0",
    ],
    extras_require={
        "test": ["pytest>=8.4.1", "pytest-cov>=6.2.1"],
        "doc": ["sphinx>=4.2", "sphinx-rtd-theme>=1.0.0", "nbsphinx>=0.8.9"],
    },
    entry_points={},
    description=SHORT_DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ISG-Siegen/lenskit-auto",
    project_urls={"Documentation": "https://lenskit-auto.readthedocs.io"},
)
