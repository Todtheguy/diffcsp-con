import os
import glob
import importlib

# Get the path to the `scripts` directory
scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')

# Dynamically import each Python file in `scripts`
for script_file in glob.glob(os.path.join(scripts_dir, "*.py")):
    module_name = os.path.basename(script_file)[:-3]  # Strip .py extension
    if module_name != "__init__":  # Skip any __init__.py if present
        module = importlib.import_module(f"scripts.{module_name}")
        globals()[module_name] = module  # Add to `diffcsp` namespace