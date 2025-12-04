"""Tests for the kernel functions module."""

import numpy as np
import pytest
from optiml.kernels import (
    RBF,
    Matern,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
    Periodic,
    WhiteNoise,
    ConstantKernel,
    SumKernel,
    ProductKernel,
    create_kernel,
    default_kernel,
)


class TestRBFKernel:
    """Tests for RBF (Squared Exponential) kernel."""

    def test_rbf_symmetric(self):
        """Test RBF kernel is symmetric."""
        kernel = RBF(length_scale=1.0, variance=1.0)
        X = np.random.rand(10, 3)
        
        K = kernel(X)
        
        assert K.shape == (10, 10)
        np.testing.assert_array_almost_equal(K, K.T)

    def test_rbf_diagonal_ones(self):
        """Test RBF kernel has ones on diagonal when variance=1."""
        kernel = RBF(length_scale=1.0, variance=1.0)
        X = np.random.rand(10, 3)
        
        K = kernel(X)
        
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(10))

    def test_rbf_positive_semidefinite(self):
        """Test RBF kernel is positive semi-definite."""
        kernel = RBF(length_scale=1.0, variance=1.0)
        X = np.random.rand(10, 3)
        
        K = kernel(X)
        eigenvalues = np.linalg.eigvalsh(K)
        
        assert np.all(eigenvalues >= -1e-10)

    def test_rbf_length_scale_effect(self):
        """Test length scale affects kernel values."""
        kernel_short = RBF(length_scale=0.1)
        kernel_long = RBF(length_scale=10.0)
        
        X = np.array([[0], [1]])
        
        K_short = kernel_short(X)
        K_long = kernel_long(X)
        
        # Shorter length scale = faster decay
        assert K_short[0, 1] < K_long[0, 1]

    def test_rbf_ard(self):
        """Test ARD (Automatic Relevance Determination)."""
        kernel = RBF(length_scale=1.0, ard=True, n_dims=3)
        
        assert len(kernel.length_scale) == 3
        params = kernel.get_params()
        assert "length_scale_0" in params


class TestMaternKernel:
    """Tests for Matérn kernel family."""

    def test_matern_nu_values(self):
        """Test valid nu values."""
        for nu in [0.5, 1.5, 2.5, 5.0]:
            kernel = Matern(nu=nu)
            X = np.random.rand(5, 2)
            K = kernel(X)
            assert K.shape == (5, 5)

    def test_matern_invalid_nu(self):
        """Test invalid nu raises error."""
        with pytest.raises(ValueError):
            Matern(nu=1.0)  # Not a valid nu

    def test_matern12_exponential(self):
        """Test Matérn 1/2 is exponential kernel."""
        kernel = Matern12(length_scale=1.0)
        X = np.array([[0], [1]])
        
        K = kernel(X)
        expected_offdiag = np.exp(-1.0)
        
        assert K[0, 1] == pytest.approx(expected_offdiag)

    def test_matern52_smoothness(self):
        """Test Matérn 5/2 is smoother than Matérn 1/2."""
        m12 = Matern12(length_scale=1.0)
        m52 = Matern52(length_scale=1.0)
        
        X = np.array([[0], [0.1], [0.2]])
        
        K12 = m12(X)
        K52 = m52(X)
        
        # Both should be valid kernels
        assert np.all(np.linalg.eigvalsh(K12) >= -1e-10)
        assert np.all(np.linalg.eigvalsh(K52) >= -1e-10)


class TestRationalQuadraticKernel:
    """Tests for Rational Quadratic kernel."""

    def test_rq_basic(self):
        """Test basic RQ kernel computation."""
        kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
        X = np.random.rand(10, 3)
        
        K = kernel(X)
        
        assert K.shape == (10, 10)
        np.testing.assert_array_almost_equal(K, K.T)

    def test_rq_approaches_rbf(self):
        """Test RQ approaches RBF as alpha → ∞."""
        rq_large_alpha = RationalQuadratic(length_scale=1.0, alpha=1000.0, variance=1.0)
        rbf = RBF(length_scale=1.0, variance=1.0)
        
        X = np.random.rand(5, 2)
        
        K_rq = rq_large_alpha(X)
        K_rbf = rbf(X)
        
        np.testing.assert_array_almost_equal(K_rq, K_rbf, decimal=2)


class TestPeriodicKernel:
    """Tests for Periodic kernel."""

    def test_periodic_basic(self):
        """Test basic periodic kernel."""
        kernel = Periodic(length_scale=1.0, period=2.0)
        X = np.array([[0], [1], [2], [3], [4]])
        
        K = kernel(X)
        
        assert K.shape == (5, 5)
        # Points separated by period should be similar
        assert K[0, 2] > K[0, 1]  # 0 and 2 are one period apart

    def test_periodic_symmetry(self):
        """Test periodic kernel is symmetric."""
        kernel = Periodic(period=1.0)
        X = np.random.rand(10, 1) * 5
        
        K = kernel(X)
        
        np.testing.assert_array_almost_equal(K, K.T)


class TestWhiteNoiseKernel:
    """Tests for White Noise kernel."""

    def test_white_noise_diagonal(self):
        """Test white noise only affects diagonal."""
        kernel = WhiteNoise(noise_variance=0.5)
        X = np.random.rand(5, 2)
        
        K = kernel(X)
        
        np.testing.assert_array_almost_equal(np.diag(K), np.full(5, 0.5))
        # Off-diagonal should be zero when X1 != X2
        K_cross = kernel(X[:3], X[3:])
        np.testing.assert_array_almost_equal(K_cross, np.zeros((3, 2)))


class TestCompositeKernels:
    """Tests for composite kernels (Sum, Product)."""

    def test_sum_kernel(self):
        """Test sum of kernels."""
        k1 = RBF(length_scale=1.0, variance=1.0)
        k2 = WhiteNoise(noise_variance=0.1)
        
        kernel = k1 + k2
        
        assert isinstance(kernel, SumKernel)
        
        X = np.random.rand(5, 2)
        K = kernel(X)
        K1 = k1(X)
        K2 = k2(X)
        
        np.testing.assert_array_almost_equal(K, K1 + K2)

    def test_product_kernel(self):
        """Test product of kernels."""
        k1 = RBF(length_scale=1.0, variance=1.0)
        k2 = ConstantKernel(constant=2.0)
        
        kernel = k1 * k2
        
        assert isinstance(kernel, ProductKernel)
        
        X = np.random.rand(5, 2)
        K = kernel(X)
        K1 = k1(X)
        K2 = k2(X)
        
        np.testing.assert_array_almost_equal(K, K1 * K2)

    def test_composite_params(self):
        """Test parameter handling in composite kernels."""
        k1 = RBF(length_scale=1.0)
        k2 = Matern52(length_scale=2.0)
        
        kernel = k1 + k2
        params = kernel.get_params()
        
        assert "k1_length_scale" in params
        assert "k2_length_scale" in params


class TestKernelFactory:
    """Tests for kernel factory function."""

    def test_create_rbf(self):
        """Test creating RBF kernel."""
        kernel = create_kernel("rbf", length_scale=2.0)
        assert isinstance(kernel, RBF)
        assert kernel.length_scale == 2.0

    def test_create_matern52(self):
        """Test creating Matérn 5/2 kernel."""
        kernel = create_kernel("matern52")
        assert isinstance(kernel, Matern52)

    def test_create_unknown(self):
        """Test unknown kernel raises error."""
        with pytest.raises(ValueError):
            create_kernel("unknown_kernel")


class TestDefaultKernel:
    """Tests for default kernel creation."""

    def test_default_kernel_1d(self):
        """Test default kernel for 1D."""
        kernel = default_kernel(n_dims=1)
        
        X = np.random.rand(10, 1)
        K = kernel(X)
        
        assert K.shape == (10, 10)

    def test_default_kernel_multidim(self):
        """Test default kernel for multiple dimensions."""
        kernel = default_kernel(n_dims=5, use_ard=True)
        
        X = np.random.rand(10, 5)
        K = kernel(X)
        
        assert K.shape == (10, 10)


class TestKernelHyperparameters:
    """Tests for hyperparameter handling."""

    def test_get_set_params(self):
        """Test getting and setting parameters."""
        kernel = RBF(length_scale=1.0, variance=2.0)
        
        params = kernel.get_params()
        assert params["length_scale"] == 1.0
        assert params["variance"] == 2.0
        
        kernel.set_params(length_scale=3.0)
        assert kernel.length_scale == 3.0

    def test_params_array(self):
        """Test log-transformed parameter arrays."""
        kernel = RBF(length_scale=1.0, variance=1.0)
        
        params_array = kernel.get_params_array()
        assert len(params_array) == 2
        
        # Log transform: log(1.0) = 0
        np.testing.assert_array_almost_equal(params_array, [0.0, 0.0])

    def test_clone(self):
        """Test kernel cloning."""
        kernel = RBF(length_scale=2.0, variance=3.0)
        clone = kernel.clone()
        
        assert clone.length_scale == kernel.length_scale
        assert clone.variance == kernel.variance
        
        # Changing clone shouldn't affect original
        clone.length_scale = 10.0
        assert kernel.length_scale == 2.0

    def test_bounds(self):
        """Test hyperparameter bounds."""
        kernel = RBF(length_scale_bounds=(0.1, 10.0))
        bounds = kernel.bounds
        
        assert len(bounds) == kernel.n_params
        # Bounds should be in log space
        assert bounds[0][0] == pytest.approx(np.log(0.1))
        assert bounds[0][1] == pytest.approx(np.log(10.0))
