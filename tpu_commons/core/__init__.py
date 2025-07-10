from importlib import util

PATHWAYS_ENABLED = False

if util.find_spec("pathwaysutils"):
    import pathwaysutils

    pathwaysutils.initialize()
    PATHWAYS_ENABLED = True
else:
    print("Running uLLM without Pathways. "
          "Module pathwaysutils is not imported.")
