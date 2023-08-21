####
from lib.install.dependencies import install_tokenmixer_dependencies
install_tokenmixer_dependencies()

from .comfyUI.compatibility import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
