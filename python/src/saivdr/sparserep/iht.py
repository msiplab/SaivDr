"""Iterative hard thresholding (MATLAB saivdr.sparserep.IterativeHardThresholding)."""

import torch

__all__ = ["IterativeHardThresholding"]


class IterativeHardThresholding:
    """Iterative hard thresholding, equivalent to MATLAB ``IterativeHardThresholding``.

    Approximates an image by ``number_of_sparse_coefficients`` coefficients
    of the dictionary ``synthesizer`` (with the adjoint ``analyzer``).

    Parameters
    ----------
    synthesizer, analyzer : torch.nn.Module
        Single-component synthesizer and analyzer, e.g. the networks of
        :func:`saivdr.dcnn.create_nsolt_2d`.
    number_of_sparse_coefficients : int
        Number of nonzero coefficients kept for each image.
    mu : float
        Step size.
    tol_rmse : float
        Tolerance of ``||c_k - c_{k-1}||_2 / numel(c)``.
    max_iter : int
    """

    def __init__(
        self,
        synthesizer,
        analyzer,
        number_of_sparse_coefficients=1,
        mu=1 - 1e-3,
        tol_rmse=1e-7,
        max_iter=1000,
    ):
        self.synthesizer = synthesizer
        self.analyzer = analyzer
        self.number_of_sparse_coefficients = int(number_of_sparse_coefficients)
        self.mu = mu
        self.tol_rmse = tol_rmse
        self.max_iter = max_iter
        self.n_iter = 0

    @staticmethod
    def _flatten(coefs):
        return torch.cat([c.reshape(c.size(0), -1) for c in coefs], dim=1)

    @staticmethod
    def _unflatten(vec, like):
        out, sidx = [], 0
        for c in like:
            n = c[0].numel()
            out.append(vec[:, sidx : sidx + n].reshape(c.shape))
            sidx += n
        return tuple(out)

    @torch.no_grad()
    def __call__(self, source):
        """Approximate each image of ``source`` (``N x 1 x H x W``).

        Returns ``(result, coefs)``, where ``coefs`` is the tuple of the
        sparse coefficients in the format of the synthesizer inputs.
        """
        result = torch.zeros_like(source)
        like = self.analyzer(result)
        coefvec = self._flatten(like)
        nCoefs = self.number_of_sparse_coefficients
        self.n_iter = 0
        while True:
            self.n_iter += 1
            precoefvec = coefvec
            gradvec = self._flatten(self.analyzer(source - result))
            coefvec = precoefvec + self.mu * gradvec
            # Keep exactly nCoefs coefficients of the largest magnitude
            idx = torch.topk(coefvec.abs(), nCoefs, dim=1).indices
            mask = torch.zeros_like(coefvec).scatter_(1, idx, 1.0)
            coefvec = mask * coefvec
            result = self.synthesizer(*self._unflatten(coefvec, like))
            rmse = (coefvec - precoefvec).norm(dim=1).max() / coefvec[0].numel()
            if rmse < self.tol_rmse or self.n_iter >= self.max_iter:
                break
        return result, self._unflatten(coefvec, like)
