# This is the __init__.py file for the orion.libs package.
# You can use this file to initialize the package and import necessary modules.

# Example of importing a module within the package
# from . import some_module

# You can also define package-level variables or functions here
from pathlib import Path
ROOT_PATH = Path(__file__).parent.parent
__version__ = "1.0.0"