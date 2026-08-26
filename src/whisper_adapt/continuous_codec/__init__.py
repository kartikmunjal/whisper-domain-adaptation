"""Continuous-latent audio codec baseline."""

from .model import ContinuousAudioVAE, ContinuousCodecConfig, uniform_quantize

__all__ = ["ContinuousAudioVAE", "ContinuousCodecConfig", "uniform_quantize"]
