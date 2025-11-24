# Module for Fourier interpolation of 2D functions sampled on a polar grid
# Author: A. Corstanje (a.corstanje@astro.ru.nl), 2020 - 2023
#
# See article: A. Corstanje et al. 2023, JINST 18 P09005, arXiv 2306.13514, doi 10.1088/1748-0221/18/09/P09005
# Please cite this when using code and/or methods in your analysis
import jax

jax.config.update("jax_enable_x64", True)
from typing_extensions import Self, Any, Dict, Tuple
from functools import partial
import jax.numpy as jnp
from jax import tree_util

from .utilities import (
    batched_fourier_interp_1d,
    batched_fourier_signal_interp,
    batched_fourier_sum,
    batched_fourier_sum_1d,
)

@partial(jax.jit, static_argnames=("single_axis", "meshgrid"))
def _eval_radial_interpolator(
    rad: jnp.ndarray,
    rad_grid: jnp.ndarray,
    fft_grid: jnp.ndarray,
    single_axis: bool,
    meshgrid: bool,
):
    """
    Module-level jitted wrapper to call the radial interpolator.
    - rad: query radii (scalar or array)
    - rad_grid: (R,)
    - fft_grid: (R, nphi, ...)  (the angular_FFT arranged per-radius)
    single_axis, meshgrid are static for stable compilation.
    """
    if single_axis:
        return batched_fourier_interp_1d(rad=rad, rad_grid=rad_grid, fft_grid=fft_grid)
    else:
        return batched_fourier_signal_interp(rad=rad, rad_grid=rad_grid, fft_grid=fft_grid)


class interp2d_fourier:
    """
    Produce a callable instance (given by the function __call__) to interpolate a function value(x, y) sampled at the input positions (x, y).

    Parameters
    ----------
    x : np.ndarray
        1D array of x positions of simulated antennas
    y : np.ndarray
        idem for y
    values : np.ndarray
        the function values (as 1D array) at positions (x, y)
    single_axis : bool, default=True
    """

    # @classmethod
    # def get_ordering_indices(cls, x, y):
    #     """
    #     Produce ordering indices to create (radius, phi) 2D-array from unordered x and y (1D-)arrays.

    #     Parameters
    #     ----------
    #     x : np.ndarray
    #         1D array of x positions
    #     y : np.ndarray
    #         1D array of y positions
    #     """
    #     radius = jnp.sqrt(x**2 + y**2)
    #     phi = jnp.arctan2(y, x)  # uses interval -pi..pi
    #     phi = jnp.around(
    #         phi, 15
    #     )  # based on observation that offsets from 0 up to 1e-16 can result from arctan2
    #     phi = jnp.where(phi<0, phi + 2 * jnp.pi, phi)
    #     # phi = phi.at[phi < 0].set(
    #     #     phi[phi < 0] + 2 * jnp.pi
    #     # )  # put into 0..2pi for ordering.
    #     phi_sorting = jnp.argsort(phi)
    #     # Assume star-shaped pattern, i.e. radial # steps = number of (almost) identical phi-values
    #     # May not work very near (0, 0)
    #     phi0 = phi[phi_sorting][0]

    #     test = phi[phi_sorting] - phi0
    #     # radial_steps = len(jnp.where(jnp.abs(test) < 0.0001)[0])
    #     radial_steps = jnp.sum(jnp.abs(test) < 0.0001)
    #     phi_steps = len(phi_sorting) // radial_steps
    #     # phi_sorting = phi_sorting.reshape((phi_steps, radial_steps))
    #     phi_sorting = jnp.reshape(phi_sorting, (phi_steps, radial_steps))
    #     indices = jnp.argsort(radius[phi_sorting], axis=1)
    #     phi_sorting = jnp.take_along_axis(phi_sorting, indices, axis=1)
    #     ordering_indices = phi_sorting.T  # get shape (radial_steps, phi_steps)

    #     return ordering_indices

    @classmethod
    def cos_sin_components(cls, fourier):
        """
        Convert complex FFT as from np.fft.rfft to real-valued cos, sin components.

        Parameters
        -----------
        fourier : np.ndarray
            complex Fourier components, with Fourier series running along the last axis.
        """
        cos_components = 2 * jnp.real(fourier)
        cos_components = cos_components.at[..., 0].set(cos_components[..., 0] * 0.5)
        cos_components = cos_components.at[..., -1].set(cos_components[..., -1] * 0.5)
        sin_components = -2 * jnp.imag(fourier)

        return cos_components, sin_components

    def __init__(
        self: Self,
        x: jax.typing.ArrayLike,
        y: jax.typing.ArrayLike,
        values: jax.typing.ArrayLike,
        ordered_indices: jax.Array,
        single_axis: bool = True,
        meshgrid: bool = False,
    ) -> None:
        """
        Initialize the interpolator.

        Note that this differs from the numpy version, where instead the grid for the interpolator (x and y-values) are set, and that the actual interpolation is done in __call__.
        """
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        values = jnp.asarray(values)

        # Convert (x, y) to (r, phi), make 2d position array, sorting positions and values by r and phi
        radius = jnp.sqrt(x**2 + y**2)

        # Store the (unique) radius values
        radial_axis = radius[ordered_indices][:, 0]
        # Check if the radius does not vary along angular direction (with tolerance)
        # if jnp.max(jnp.std(radius[ordered_indices], axis=1)) > 0.1 * jnp.min(radius):
        #     raise ValueError(
        #         "Radius must be (approx.) constant along angular direction. Are you sure that you are using a starshape and the core is set properly?"
        #     )
        # FFT over the angular direction, for each radius
        angular_FFT = jnp.fft.rfft(values[ordered_indices], axis=1)
        fourier_norm = float(values[ordered_indices].shape[1])
        angular_FFT /= fourier_norm  # normalize

        # store minimal arrays as leaves
        self.radial_axis = radial_axis          # shape (R,)
        self.angular_FFT = angular_FFT          # shape (R, nphi, ...)
        # store phi0 (the reference offset used in __call__)
        phi = jnp.arctan2(y, x)
        phi = jnp.around(phi, 15)
        phi = jnp.where(phi < 0, phi + 2 * jnp.pi, phi)
        # we need the phi0 used in ordering: take the first after sorting
        phi_sorting = jnp.argsort(phi)
        self._phi0 = phi[phi_sorting][0]

        # static flags: keep as small python values in aux via tree_flatten
        self._single_axis_flag = bool(single_axis)
        self._meshgrid_flag = bool(meshgrid)

    # PyTree protocol: children (leaves) and aux (static metadata)
    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        children = (self.radial_axis, self.angular_FFT, self._phi0)
        aux = {
            "single_axis": self._single_axis_flag,
            "meshgrid": self._meshgrid_flag,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux: Dict[str, Any], children: Tuple[Any, ...]) -> "interp2d_fourier":
        obj = cls.__new__(cls)
        obj.radial_axis, obj.angular_FFT, obj._phi0 = children
        obj._single_axis_flag = bool(aux["single_axis"])
        obj._meshgrid_flag = bool(aux["meshgrid"])
        return obj

    def __call__(self, x, y, max_fourier_mode=None):
        """
        Interpolate the input used in __init__ for input positions (x, y).

        Parameters
        ----------
        x : float or np.ndarray
            x positions as float or numpy ND array
        y : float or np.ndarray
            idem for y
        max_fourier_mode : int, optional
            cutoff for spatial frequencies along circles, i.e. do Fourier sum up to (incl.) this mode.
            Default None i.e. do all modes
        """
        x_q = jnp.asarray(x)
        y_q = jnp.asarray(y)

        rad_q = jnp.sqrt(x_q ** 2 + y_q ** 2)
        phi_q = jnp.arctan2(y_q, x_q) - self._phi0
        # ensure array-like phi for broadcasting
        phi_q = jnp.asarray(phi_q) if jnp.isscalar(phi_q) else phi_q

        # Interpolate Fourier components over all values of radius
        # Evaluate radial interpolator with the module-level jitted wrapper
        fourier = _eval_radial_interpolator(rad_q, self.radial_axis, self.angular_FFT, self._single_axis_flag, self._meshgrid_flag)
        # decide which axis contains Fourier components depending on meshgrid
        fourier_comp_axis = 2 if self._meshgrid_flag else 1
        fourier_len = fourier.shape[fourier_comp_axis]

        # convert to cos/sin
        cos_components, sin_components = interp2d_fourier.cos_sin_components(fourier)

        # determine Fourier mode multipliers
        limit = (max_fourier_mode + 1) if (max_fourier_mode is not None) else fourier_len
        mult = jnp.arange(limit, dtype=int)
        phi_k = phi_q[..., jnp.newaxis] * mult  # shape (..., limit)

        # choose correct summation helper
        fourier_summer = batched_fourier_sum if self._meshgrid_flag else batched_fourier_sum_1d

        # compute sum_k( c_k cos(k phi) + s_k sin(k phi) )
        result = fourier_summer(phi_k, cos_components[..., 0:limit], sin_components[..., 0:limit])

        return result

    # Some getters for the angular FFT, its radial interpolator function, and the radial axis points used

    def get_angular_FFT(self):
        return self.angular_FFT

    def get_radial_axis(self):
        return self.radial_axis
