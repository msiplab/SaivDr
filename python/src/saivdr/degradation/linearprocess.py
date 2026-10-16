"""Linear degradation processes (MATLAB saivdr.degradation.linearprocess).

Images are tensors whose last two dimensions are (height, width), e.g.
``N x C x H x W``; every leading dimension is processed independently.
"""

import math

import torch
import torch.nn.functional as F

__all__ = [
    "AbstLinearSystem",
    "BlurSystem",
    "DecimationSystem",
    "PixelLossSystem",
    "fspecial_gaussian",
    "fspecial_average",
    "imfilter",
]


def fspecial_gaussian(size, sigma, dtype=torch.float64):
    """Gaussian kernel equivalent to MATLAB ``fspecial('gaussian',size,sigma)``."""
    if isinstance(size, int):
        size = (size, size)
    hy, hx = ((s - 1) / 2 for s in size)
    y = torch.arange(-hy, hy + 1, dtype=torch.float64).reshape(-1, 1)
    x = torch.arange(-hx, hx + 1, dtype=torch.float64).reshape(1, -1)
    h = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    h[h < torch.finfo(torch.float64).eps * h.max()] = 0
    return (h / h.sum()).to(dtype)


def fspecial_average(size, dtype=torch.float64):
    """Averaging kernel equivalent to MATLAB ``fspecial('average',size)``."""
    if isinstance(size, int):
        size = (size, size)
    return torch.full(tuple(size), 1.0 / math.prod(size), dtype=dtype)


def _pad(x, pads, option, value):
    """Pad the last two dimensions by ``pads = (left, right, top, bottom)``."""
    if option == "Value":
        return F.pad(x, pads, mode="constant", value=value)
    if option == "Circular":
        return F.pad(x, pads, mode="circular")
    if option == "Replicate":
        return F.pad(x, pads, mode="replicate")
    if option == "Symmetric":
        # Mirror including the edge pixel, as MATLAB 'symmetric'
        left, right, top, bottom = pads
        h, w = x.shape[-2:]
        rows = torch.arange(-top, h + bottom, device=x.device)
        cols = torch.arange(-left, w + right, device=x.device)
        rows = torch.where(rows < 0, -rows - 1, rows)
        rows = torch.where(rows >= h, 2 * h - rows - 1, rows)
        cols = torch.where(cols < 0, -cols - 1, cols)
        cols = torch.where(cols >= w, 2 * w - cols - 1, cols)
        return x.index_select(-2, rows).index_select(-1, cols)
    raise ValueError(f"Invalid boundary option: {option}")


def imfilter(x, kernel, mode="conv", boundary="Value", value=0.0):
    """Filter the last two dimensions as MATLAB ``imfilter`` with the 'same' output size.

    ``mode`` is 'conv' or 'corr', and ``boundary`` is one of 'Value',
    'Symmetric', 'Replicate' and 'Circular'.
    """
    kernel = torch.as_tensor(kernel, dtype=x.dtype, device=x.device)
    if kernel.ndim == 0:
        kernel = kernel.reshape(1, 1)
    kh, kw = kernel.shape
    # Center of the kernel as in imfilter: floor((size+1)/2), 1-based. For
    # 'conv', the kernel is rotated by 180 degrees and the same center is
    # used, so that 'conv' and 'corr' are not adjoint for even sizes (as MATLAB).
    cy, cx = (kh + 1) // 2 - 1, (kw + 1) // 2 - 1
    if mode == "conv":
        kernel = torch.flip(kernel, dims=(0, 1))
    elif mode != "corr":
        raise ValueError(f"Invalid mode: {mode}")
    shape = x.shape
    y = x.reshape(-1, 1, *shape[-2:])
    y = _pad(y, (cx, kw - 1 - cx, cy, kh - 1 - cy), boundary, value)
    y = F.conv2d(y, kernel.reshape(1, 1, kh, kw))
    return y.reshape(shape)


class AbstLinearSystem:
    """Base class of the linear processes.

    ``forward`` (``ProcessingMode='Normal'`` in MATLAB) and ``adjoint``
    (``'Adjoint'``) act on the last two dimensions. :meth:`lambda_max`
    estimates the maximum eigenvalue of ``P^T P`` by the power method as
    MATLAB does (its random initial vector differs from MATLAB's, so the
    estimate agrees up to the tolerance).
    """

    def __init__(self, tol_of_power_method=1e-8, max_iter_of_power_method=int(1e6)):
        self.tol_of_power_method = tol_of_power_method
        self.max_iter_of_power_method = max_iter_of_power_method
        self._lambda_max = {}

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def adjoint(self, x):
        raise NotImplementedError

    def original_dimension(self, observed_dimension):
        """Size (height, width) of the original image for an observed size."""
        return tuple(observed_dimension)

    def lambda_max(self, observed_dimension, dtype=torch.float64, device=None, seed=0):
        """Maximum eigenvalue of the Gram matrix ``P^T P`` for the observed size."""
        key = (tuple(observed_dimension), dtype, str(device))
        if key not in self._lambda_max:
            size = self.original_dimension(observed_dimension)
            g = torch.Generator().manual_seed(seed)
            upst = torch.rand(size, generator=g, dtype=torch.float64).to(dtype=dtype, device=device)
            lpre = 1.0
            for _ in range(self.max_iter_of_power_method):
                upre = upst / upst.norm()
                upst = self.adjoint(self.forward(upre))
                lpst = (upst.flatten() @ upst.flatten()) / (upst.flatten() @ upre.flatten())
                lpst = float(lpst)
                err = abs(lpst - lpre) / abs(lpre)
                lpre = lpst
                if err <= self.tol_of_power_method:
                    break
            self._lambda_max[key] = lpst
        return self._lambda_max[key]


def _blur_kernel(blur_type, size, sigma, custom, default_size, default_sigma, average_size):
    if blur_type == "Identical":
        return torch.ones(1, 1, dtype=torch.float64), None, None
    if blur_type == "Average":
        size = average_size if size is None else size
        return fspecial_average(size), size, None
    if blur_type == "Gaussian":
        sigma = default_sigma if sigma is None else sigma
        size = default_size(sigma) if size is None else size
        return fspecial_gaussian(size, sigma), size, sigma
    if blur_type == "Custom":
        if custom is None:
            raise ValueError("CustomKernel should be specified.")
        return torch.as_tensor(custom, dtype=torch.float64), size, sigma
    raise ValueError(f"Invalid blur type: {blur_type}")


class BlurSystem(AbstLinearSystem):
    """Blur process, equivalent to MATLAB ``saivdr.degradation.linearprocess.BlurSystem``.

    ``blur_type`` is one of 'Identical', 'Gaussian' (default sigma 2 and size
    ``2*ceil(4*sigma)+1``), 'Average' (default size 3) and 'Custom'.
    ``boundary_option`` is one of 'Value', 'Symmetric', 'Replicate' and
    'Circular'.
    """

    def __init__(
        self,
        blur_type="Identical",
        boundary_option="Value",
        size_of_kernel=None,
        sigma_of_gaussian_kernel=None,
        custom_kernel=None,
        boundary_value=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blur_type = blur_type
        self.boundary_option = boundary_option
        self.boundary_value = boundary_value
        self.blur_kernel, self.size_of_kernel, self.sigma_of_gaussian_kernel = _blur_kernel(
            blur_type,
            size_of_kernel,
            sigma_of_gaussian_kernel,
            custom_kernel,
            lambda s: 2 * math.ceil(4 * s) + 1,
            2.0,
            (3, 3),
        )

    def forward(self, x):
        return imfilter(x, self.blur_kernel, "conv", self.boundary_option, self.boundary_value)

    def adjoint(self, x):
        return imfilter(x, self.blur_kernel, "corr", self.boundary_option, self.boundary_value)


class DecimationSystem(AbstLinearSystem):
    """Blur followed by downsampling, equivalent to MATLAB ``DecimationSystem``.

    ``blur_type`` 'Gaussian' defaults to size ``4*factor+1`` and sigma
    ``max(factor)``; 'Average' uses the decimation factor as the size.
    """

    def __init__(
        self,
        vertical_decimation_factor=2,
        horizontal_decimation_factor=2,
        blur_type="Identical",
        boundary_option="Value",
        size_of_gaussian_kernel=None,
        sigma_of_gaussian_kernel=None,
        custom_kernel=None,
        boundary_value=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decimation_factor = (int(vertical_decimation_factor), int(horizontal_decimation_factor))
        self.blur_type = blur_type
        self.boundary_option = boundary_option
        self.boundary_value = boundary_value
        dv, dh = self.decimation_factor
        self.blur_kernel, self.size_of_gaussian_kernel, self.sigma_of_gaussian_kernel = _blur_kernel(
            blur_type,
            size_of_gaussian_kernel,
            sigma_of_gaussian_kernel,
            custom_kernel,
            lambda s: (4 * dv + 1, 4 * dh + 1),
            float(max(dv, dh)),
            (dv, dh),
        )
        # Phase of the upsampling in the adjoint, as offset in MATLAB
        self.offset = tuple((s + 1) % 2 for s in self.blur_kernel.shape)

    def original_dimension(self, observed_dimension):
        return tuple(d * o for d, o in zip(self.decimation_factor, observed_dimension))

    def forward(self, x):
        v = imfilter(x, self.blur_kernel, "conv", self.boundary_option, self.boundary_value)
        dv, dh = self.decimation_factor
        return v[..., ::dv, ::dh]

    def adjoint(self, x):
        dv, dh = self.decimation_factor
        ov, oh = self.offset
        shape = x.shape[:-2] + (x.shape[-2] * dv, x.shape[-1] * dh)
        v = x.new_zeros(shape)
        v[..., ov::dv, oh::dh] = x
        return imfilter(v, self.blur_kernel, "corr", self.boundary_option, self.boundary_value)


class PixelLossSystem(AbstLinearSystem):
    """Pixel loss, equivalent to MATLAB ``PixelLossSystem``.

    With ``loss_type='Random'``, pixels are lost with probability ``density``
    using a generator seeded by ``seed`` (the mask differs from MATLAB's,
    whose random numbers cannot be reproduced). ``loss_type='Specified'``
    uses ``mask`` (nonzero: observed), which allows the MATLAB mask to be
    reproduced. The mask is shared by all the leading dimensions.
    """

    def __init__(self, loss_type="Random", density=0.5, seed=0, mask=None, **kwargs):
        super().__init__(**kwargs)
        if loss_type not in ("Random", "Specified"):
            raise ValueError(f"Invalid loss type: {loss_type}")
        self.loss_type = loss_type
        self.density = density
        self.seed = seed
        self.mask = None if mask is None else torch.as_tensor(mask) != 0
        if self.mask is not None and loss_type == "Specified":
            self.density = float(self.mask.double().mean())

    def _mask(self, x):
        if self.loss_type == "Specified":
            if self.mask is None:
                raise ValueError("Mask should be specified.")
        elif self.mask is None or self.mask.shape != x.shape[-2:]:
            g = torch.Generator().manual_seed(self.seed)
            self.mask = torch.rand(x.shape[-2:], generator=g, dtype=torch.float64) > self.density
        return self.mask.to(x.device)

    def forward(self, x):
        return x * self._mask(x).to(x.dtype)

    def adjoint(self, x):
        return self.forward(x)
