# mypackage/__init__.py

# Package version
__version__ = "1.0.2"

# Specify what is available to import from the package
__all__ = ["identify_runways", "choropleth_map", "load_dataset", "add_trajectory"]

# Import from submodules to make available at the package level
from .runways import identify_runways
from .runways import load_dataset
from .h3maps import choropleth_map
from .h3maps import add_trajectory
from .setup_module import ensure_data_available

# Initialization package data 
ensure_data_available(local_install = False)
