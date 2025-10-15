import os
from datetime import datetime, timezone

__version__= os.environ.get('VERSION')

def generate_nightly_version(base_version: str = __version__) -> str:
    datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    nightly_version=f"{base_version}.dev{datetime_str}"
    return nightly_version


def generate_release_version(base_version: str = __version__) -> str:
    return base_version

def get_version():
    release_type = os.environ.get('RELEASE_TYPE')
    if release_type == 'nightly':
        return generate_nightly_version()
    else:
        return generate_release_version()
    
