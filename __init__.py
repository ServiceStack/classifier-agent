from .classifier_agent_node import RegisterClassifierNodes

NODE_CLASS_MAPPINGS = {
    **RegisterClassifierNodes.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **RegisterClassifierNodes.NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
