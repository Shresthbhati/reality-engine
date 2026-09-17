"""Core math types shared between Reality Engine core and child projects.

This module provides fundamental 3D math types (Vec3, Mat3, Quat) that are
used by both the core city-construction pipeline and child simulation projects.
"""
from engine.math.math3 import Vec3, Mat3, Quat

__all__ = ["Vec3", "Mat3", "Quat"]