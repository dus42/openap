"""JAX backend for OpenAP.

This backend provides JAX support for automatic differentiation,
JIT compilation, and GPU acceleration.
"""

from typing import Any


class JaxBackend:
    """JAX backend implementation.

    This backend uses JAX for:
    - Automatic differentiation (grad, jacobian, hessian)
    - JIT compilation for performance
    - GPU/TPU acceleration
    - Vectorization (vmap)

    JAX is imported lazily to avoid requiring it as a dependency
    for users who only need NumPy functionality.
    """

    def __init__(self):
        """Initialize JAX backend with lazy import."""
        self._jnp = None

    @property
    def jnp(self):
        """Lazy import of JAX numpy."""
        if self._jnp is None:
            try:
                import jax.numpy as jnp

                self._jnp = jnp
            except ImportError:
                raise ImportError(
                    "JAX is required for this backend. "
                    "Install with: pip install jax jaxlib"
                )
        return self._jnp

    # --- Basic Math Functions ---

    def sqrt(self, x: Any) -> Any:
        return self.jnp.sqrt(x)

    def exp(self, x: Any) -> Any:
        return self.jnp.exp(x)

    def log(self, x: Any) -> Any:
        return self.jnp.log(x)

    def power(self, x: Any, y: Any) -> Any:
        return self.jnp.power(x, y)

    # --- Trigonometric Functions ---

    def sin(self, x: Any) -> Any:
        return self.jnp.sin(x)

    def cos(self, x: Any) -> Any:
        return self.jnp.cos(x)

    def tan(self, x: Any) -> Any:
        return self.jnp.tan(x)

    def arcsin(self, x: Any) -> Any:
        return self.jnp.arcsin(x)

    def arccos(self, x: Any) -> Any:
        return self.jnp.arccos(x)

    def arctan(self, x: Any) -> Any:
        return self.jnp.arctan(x)

    def arctan2(self, y: Any, x: Any) -> Any:
        return self.jnp.arctan2(y, x)

    # --- Comparison and Conditional ---

    def abs(self, x: Any) -> Any:
        return self.jnp.abs(x)

    def smooth_abs(self, x: Any, softness: float = 1.0) -> Any:
        return self.jnp.sqrt(x**2 + softness**2)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return self.jnp.where(condition, x, y)

    def maximum(self, x: Any, y: Any) -> Any:
        return self.jnp.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> Any:
        return self.jnp.minimum(x, y)

    def clip(self, x: Any, min_val: Any, max_val: Any) -> Any:
        return self.jnp.clip(x, min_val, max_val)

    def smooth_max(self, x: Any, y: Any, softness: float = 1.0) -> Any:
        delta = x - y
        return 0.5 * (x + y + self.jnp.sqrt(delta**2 + softness**2))

    def smooth_min(self, x: Any, y: Any, softness: float = 1.0) -> Any:
        delta = x - y
        return 0.5 * (x + y - self.jnp.sqrt(delta**2 + softness**2))

    def smooth_clip(
        self, x: Any, min_val: Any, max_val: Any, softness: float = 1.0
    ) -> Any:
        return self.smooth_min(
            self.smooth_max(x, min_val, softness),
            max_val,
            softness,
        )

    def smooth_switch(
        self,
        selector: Any,
        threshold: Any,
        left: Any,
        right: Any,
        softness: float = 1.0,
    ) -> Any:
        weight = 0.5 * (
            1 + self.jnp.tanh((selector - threshold) / (2 * softness))
        )
        return left + weight * (right - left)

    # --- Interpolation ---

    def interp(self, x: Any, xp: Any, fp: Any) -> Any:
        """Linear interpolation.

        Note: xp and fp are converted to JAX arrays if they are lists.
        """
        xp = self.jnp.array(xp) if isinstance(xp, list) else xp
        fp = self.jnp.array(fp) if isinstance(fp, list) else fp
        return self.jnp.interp(x, xp, fp)

    # --- Array Creation ---

    def linspace(self, start: Any, stop: Any, num: int) -> Any:
        return self.jnp.linspace(start, stop, num)

    # --- Modulo ---

    def fmod(self, x: Any, y: Any) -> Any:
        return self.jnp.fmod(x, y)

    # --- Constants ---

    @property
    def pi(self) -> float:
        return self.jnp.pi
