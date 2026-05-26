from flax import struct

from jaxatari.environments import JaxEnvironment

class CrazyClimberState(struct.PyTreeNode):
    pass

class CrazyClimberObservation(struct.PyTreeNode):
    pass

class CrazyClimberInfo(struct.PyTreeNode):
    pass

class CrazyClimberConstants(struct.PyTreeNode):
    pass

class JaxCrazyClimber(JaxEnvironment[CrazyClimberState, CrazyClimberObservation, CrazyClimberInfo, CrazyClimberConstants]):
    pass