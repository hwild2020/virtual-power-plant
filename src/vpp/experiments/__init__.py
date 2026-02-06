"""
Experiments package for VPP optimization research.

Contains data generation, training pipelines, and analysis tools.
"""

from .generate_data import DataGenerator, generate_training_data

__all__ = ["DataGenerator", "generate_training_data"]
