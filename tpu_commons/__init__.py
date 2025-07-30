import os
import warnings

# The warning message to be displayed to the user
PROD_WARNING = (
    "🚨  CAUTION: You are using 'tpu_commons' , which is experimental "
    "and NOT intended for production use yet. Please see the README for more details."
)

warnings.warn(PROD_WARNING, UserWarning, stacklevel=2)

# Must run pathwaysutils.initialize() before any JAX operations.
if "proxy" in os.environ.get('JAX_PLATFORMS', '').lower():
    import pathwaysutils
    pathwaysutils.initialize()
    print("Running vLLM with Pathways. "
          "Module pathwaysutils is imported.")
else:
    print("Running vLLM without Pathways. "
          "Module pathwaysutils is not imported.")
