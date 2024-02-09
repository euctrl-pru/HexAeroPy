# mypackage/__init__.py

# Package version
__version__ = "1.0.0"

# Specify what is available to import from the package
__all__ = ["identify_runways", "choropleth_map"]

# Import from submodules to make available at the package level
from .runways import identify_runways
from .h3maps import choropleth_map

# Initialization code (optional)
#print(f"Initializing Hex version {__version__}.")

# You can also include any package-wide constants or configuration settings here
#PACKAGE_CONSTANT = "This is a package-wide constant."