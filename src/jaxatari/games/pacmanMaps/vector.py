from typing import NamedTuple
import jax.numpy as jnp
import chex


class Vector2(NamedTuple):
    """
    JAX-compatible 2D vector class using NamedTuple for immutability.
    All operations return new Vector2 instances and are JIT-compatible.
    
    Create with: Vector2(x=0, y=0) or Vector2(x=jnp.array(0.0), y=jnp.array(0.0))
    """
    x: chex.Array
    y: chex.Array
    thresh: chex.Array = jnp.array(0.000001, dtype=jnp.float32)
    
    def __add__(self, other):
        """Vector addition: self + other."""
        if isinstance(other, Vector2):
            return Vector2(x=self.x + other.x, y=self.y + other.y)
        else:
            raise TypeError(f"Unsupported operand type for +: Vector2 and {type(other)}")
    
    def __sub__(self, other):
        """Vector subtraction: self - other."""
        if isinstance(other, Vector2):
            return Vector2(x=self.x - other.x, y=self.y - other.y)
        else:
            raise TypeError(f"Unsupported operand type for -: Vector2 and {type(other)}")
    
    def __neg__(self):
        """Negation: -self."""
        return Vector2(x=-self.x, y=-self.y)
    
    def __mul__(self, scalar):
        """Scalar multiplication: self * scalar."""
        return Vector2(x=self.x * scalar, y=self.y * scalar)
    
    def __div__(self, scalar):
        """Division: self / scalar."""
        if scalar != 0:
            return Vector2(x=self.x / jnp.float32(scalar), y=self.y / jnp.float32(scalar))
        return None
    
    def __truediv__(self, scalar):
        """True division: self / scalar."""
        return self.__div__(scalar)
    
    def __eq__(self, other):
        """Check if two vectors are approximately equal (within threshold 1e-6)."""
        # thresh = jnp.array(0.000001, dtype=jnp.float32)
        if isinstance(other, Vector2):
            dx = jnp.abs(self.x - other.x)
            dy = jnp.abs(self.y - other.y)
            return jnp.logical_and(dx < self.thresh, dy < self.thresh)
        return False
    
    def magnitudeSquared(self):
        """Return the squared magnitude (x² + y²)."""
        return self.x ** 2 + self.y ** 2
    
    def magnitude(self):
        """Return the magnitude (length) of the vector."""
        return jnp.sqrt(self.magnitudeSquared())
    
    def copy(self):
        """Return a copy of this vector."""
        return Vector2(x=self.x, y=self.y)
    
    def asTuple(self):
        """Return (x, y) as a tuple."""
        return (self.x, self.y)
    
    def asInt(self):
        """Return (int(x), int(y)) as a tuple."""
        return (jnp.int32(self.x), jnp.int32(self.y))
    
    def __str__(self):
        """String representation."""
        return "<"+str(self.x)+", "+str(self.y)+">"
