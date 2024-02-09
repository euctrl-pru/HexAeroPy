# mypackage/__init__.py

# Package version
__version__ = "1.0.0"

# Specify what is available to import from the package
__all__ = ["identify_runways", "choropleth_map", "load_dataset", "add_trajectory"]

# Import from submodules to make available at the package level
from .runways import identify_runways
from .runways import load_dataset
from .h3maps import choropleth_map
from .h3maps import add_trajectory

# Initialization code (optional)
#print(f"HexAeroPy version {__version__}.")
#print("Brought to you by EUROCONTROL, the European Organisation for the Safety of Air Navigation.")

# You can also include any package-wide constants or configuration settings here
#PACKAGE_CONSTANT = "This is a package-wide constant."