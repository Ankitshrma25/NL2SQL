"""Offline, schema-agnostic, self-correcting Natural Language -> SQL pipeline."""
from .config import ModelConfig, PipelineConfig
from .orchestrator import ChatResponse, NL2SQLChatbot

__all__ = ["NL2SQLChatbot", "ChatResponse", "PipelineConfig", "ModelConfig"]
__version__ = "1.0.0"
