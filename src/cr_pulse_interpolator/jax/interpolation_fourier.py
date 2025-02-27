# Module for Fourier interpolation of 2D functions sampled on a polar grid
# Author: A. Corstanje (a.corstanje@astro.ru.nl), 2020 - 2023
#
# See article: A. Corstanje et al. 2023, JINST 18 P09005, arXiv 2306.13514, doi 10.1088/1748-0221/18/09/P09005 
# Please cite this when using code and/or methods in your analysis

import jax.numpy as jnp
import interpax as intp


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
    radial_method : str, default='cubic'
        the interp1d method for interpolating Fourier components along the radial axis
    fill_value : array-like or (array-like, array_like) or “extrapolate”, default='extrapolate'
        the fill value to pass to interp1d to use for a radius outside the min..max radius interval from
        the input. Set to 'extrapolate' to extrapolate beyond radial limits; accuracy outside the interval is limited.
        If set to None, the `fill_value` is set such the output is constant for r < r_min, and 0 for r > r_max.
    recover_concentric_rings : bool, default=False
        set True if the grid is not purely circular-symmetric; results may not be accurate.
    """

    @classmethod
    def get_ordering_indices(cls, x, y):
        """
        Produce ordering indices to create (radius, phi) 2D-array from unordered x and y (1D-)arrays.

        Parameters
        ----------
        x : np.ndarray
            1D array of x positions
        y : np.ndarray
            1D array of y positions
        """
        radius = jnp.sqrt(x ** 2 + y ** 2)
        phi = jnp.arctan2(y, x)  # uses interval -pi..pi
        phi = jnp.around(phi, 15)  # based on observation that offsets from 0 up to 1e-16 can result from arctan2
        phi = phi.at[phi < 0].set(phi[phi < 0] + 2 * jnp.pi)  # put into 0..2pi for ordering.
        phi_sorting = jnp.argsort(phi)
        # Assume star-shaped pattern, i.e. radial # steps = number of (almost) identical phi-values
        # May not work very near (0, 0)
        cls._phi0 = phi[phi_sorting][0]

        test = phi[phi_sorting] - cls._phi0
        radial_steps = len(jnp.where(jnp.abs(test) < 0.0001)[0])
        phi_steps = len(phi_sorting) // radial_steps
        phi_sorting = phi_sorting.reshape((phi_steps, radial_steps))
        indices = jnp.argsort(radius[phi_sorting], axis=1)
        phi_sorting = jnp.array([phi_sorting[i][indices[i]] for i in range(phi_steps)])
        # for i in range(phi_steps):  # Sort by radius; should be possible without for-loop...
        #     phi_sorting = phi_sorting[i][indices[i]]
        ordering_indices = phi_sorting.T  # get shape (radial_steps, phi_steps)

        return ordering_indices

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

    def __init__(self, x, y, values, radial_method='cubic2', fill_value=None, recover_concentric_rings=False):
        # Convert (x, y) to (r, phi), make 2d position array, sorting positions and values by r and phi
        radius = jnp.sqrt(x ** 2 + y ** 2)

        ordering_indices = self.get_ordering_indices(x, y)
        values_ordered = jnp.copy(values)[ordering_indices]

        # Store the (unique) radius values
        self.radial_axis = radius[ordering_indices][:, 0]
        # Check if the radius does not vary along angular direction (with tolerance)
        if jnp.max(jnp.std(radius[ordering_indices], axis=1)) > 0.1 * jnp.min(radius):
            if not recover_concentric_rings:
                raise ValueError("Radius must be (approx.) constant along angular direction. "
                                 "You can try to \"fix\" that by using \"recover_concentric_rings=True\"")
            else:
                self.radial_axis = jnp.mean(radius[ordering_indices], axis=1)
                values_ordered_interpolated = []
                for x, y in zip(radius[ordering_indices].T, values_ordered.T):
                    intpf = intp.Interpolator1D(
                        x=x, f=y, method=radial_method, extrap=True, axis=0)
                    values_ordered_interpolated.append(intpf(self.radial_axis))
                values_ordered = jnp.array(values_ordered_interpolated).T

        # FFT over the angular direction, for each radius
        self.angular_FFT = jnp.fft.rfft(values_ordered, axis=1)
        length = values_ordered.shape[-1]
        self.angular_FFT /= float(length)  # normalize

        # construct radial axes by repeating same values in second axis of angular component
        # needed to pass into interpax.interp1d
        radial_axes_interpd = jnp.repeat(jnp.expand_dims(self.radial_axis, axis=1), repeats=self.angular_FFT.shape[1], total_repeat_length=self.angular_FFT.shape[1], axis=1)

        # Produce interpolator function, interpolating the FFT components as a function of radius
        print(radial_axes_interpd.shape, self.angular_FFT.shape)
        if fill_value is None:
            fill_value = (self.angular_FFT[0], jnp.zeros_like(self.angular_FFT[0]))
        self.interpolator_radius = intp.Interpolator1D(
            x=self.radial_axis, f=self.angular_FFT, method=radial_method, extrap=False, axis=0
        )  # Interpolates the Fourier components along the radial axis

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
        radius = jnp.sqrt(x ** 2 + y ** 2)
        phi = jnp.arctan2(y, x) - self._phi0

        # Interpolate Fourier components over all values of radius
        fourier = self.interpolator_radius(radius)
        fourier_len = fourier.shape[-1]

        (cos_components, sin_components) = interp2d_fourier.cos_sin_components(fourier)

        # Multipliers for Fourier modes, as k in cos(k*phi), sin(k*phi)
        limit = max_fourier_mode + 1 if max_fourier_mode is not None else fourier_len
        mult = jnp.linspace(0, limit - 1, limit).astype(int)

        # The Fourier sum done explicitly, as sum_k( c_k cos(k phi) + s_k sin(k phi) )
        result = jnp.zeros_like(radius)
        if isinstance(phi, float):
            result += jnp.sum(cos_components[..., 0:limit] * jnp.cos(phi * mult))
            result += jnp.sum(sin_components[..., 0:limit] * jnp.sin(phi * mult))
        else:
            result += jnp.sum(cos_components[..., 0:limit] * jnp.cos(phi[..., jnp.newaxis] * mult), axis=-1)
            result += jnp.sum(sin_components[..., 0:limit] * jnp.sin(phi[..., jnp.newaxis] * mult), axis=-1)

        return result

    # Some getters for the angular FFT, its radial interpolator function, and the radial axis points used

    def get_angular_FFT(self):

        return self.angular_FFT

    def get_angular_FFT_interpolator(self):

        return self.interpolator_radius

    def get_radial_axis(self):

        return self.radial_axis
