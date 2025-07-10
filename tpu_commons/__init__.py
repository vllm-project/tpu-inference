import warnings

# The warning message to be displayed to the user
PROD_WARNING = (
    "🚨  CAUTION: You are using 'tpu_commons' , which is experimental "
    "and NOT intended for production use yet. Please see the README for more details."
)

warnings.warn(PROD_WARNING, UserWarning, stacklevel=2)
