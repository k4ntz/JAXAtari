import random
import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin

class BackgroundBlackColorMod(JaxAtariInternalModPlugin):    
    """Background always black"""
    
    constants_overrides = {
        "COLOR_CODES" : [
        [(0,0,0), (113, 115, 25)],
        [(0,0,0), (63, 1, 106)],
        [(0,0,0), (145, 120, 43)],
        [(0,0,0), (65, 144, 58)]
    ]
    }

class BackgroundRandomColorMod(JaxAtariInternalModPlugin):    
    """Background random color"""
    
    constants_overrides = {
        "COLOR_CODES" : [
        [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), (113, 115, 25)],
        [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), (63, 1, 106)],
        [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), (145, 120, 43)],
        [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), (65, 144, 58)]
    ]
    }

class NumberRandomColorMod(JaxAtariInternalModPlugin):    
    """Numbers random color"""
    
    constants_overrides = {
        "COLOR_CODES" : [
        [(18, 46, 137), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))],
        [(143, 114, 41), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))],
        [(110, 110, 15), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))],
        [(161, 104, 35), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))]
    ]
    }

class BiggerNumbersMod(JaxAtariInternalModPlugin):    
    """2-Number Problems"""
    
    constants_overrides = {
        "problemNumLen": 2,
        "numArrLen": 8,
        "initialArrPos": 3,
        "spacing": 15,
        "BAR_OFFSET": 18,
        "Y_OFFSET": 22,
        "symbol": (37, 100),
        "num0": (52, 55),
        "num1": (52, 95),
        "PAD_OBS": 0,
    }