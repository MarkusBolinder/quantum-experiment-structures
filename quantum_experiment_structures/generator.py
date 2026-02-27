"""Module to generate a random causal contextuality scenario."""

from copy import deepcopy
import random

SKELETON = {"ms": [], "c": [], "h": {"ms": None, "o": None, "e": None, "c": None}}

MEASUREMENT = [{"m": None, "e": [], "o": [], "c": []}]
