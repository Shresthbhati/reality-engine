"""Minimal render module for Studio (core).

Provides Camera and Viewport types without depending on the full
physics/rendering engine in the child project.
"""
from engine.render.viewport import Camera, Viewport

__all__ = ["Camera", "Viewport"]