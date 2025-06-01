from pathlib import Path
from setuptools import setup, find_packages

SHORT_DESCRIPTION = """LensKit-Auto is built as a wrapper around the Python LensKit recommender-system library. It automates algorithm selection and hyper parameter optimization an can build ensemble models based on the LensKit models."""

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="lkauto",
    version="0.1.1",
    author="Tobias Vente",
    python_requires=">=3.8, <=3.9",
    packages=find_packages(),
    install_requires=[
        "smac~=2.3",
        "matplotlib~=3.6",
        "lenskit>=0.14.2",
        "numpy>2.2",
        "tables~=3.8",
        "typing~=3.5"
    ],
    extras_require={
        "doc": ["nbsphinx==0.8.9", "sphinx-rtd-theme==1.*", "numpy==1.21.6", "Jinja2<3.1"],
        "test": ["pytest>=6.2.5", "pytest-cov>=2.12.1"],
    },
    entry_points={},
    description=SHORT_DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ISG-Siegen/lenskit-auto",
    project_urls={"Documentation": "https://lenskit-auto.readthedocs.io"},
)
