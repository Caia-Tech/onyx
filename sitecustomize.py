import os

# Keep this repo's tests isolated from whatever pytest plugins are installed in
# the user's environment (some require external services / native deps).
if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") is None:
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
