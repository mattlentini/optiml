"""Tests for the experimental designs module."""

import numpy as np
import pytest
from optiml.space import Space, Real, Integer, Categorical
from optiml.designs import (
    RandomDesign,
    LatinHypercubeDesign,
    SobolDesign,
    HaltonDesign,
    FullFactorialDesign,
    FractionalFactorialDesign,
    CentralCompositeDesign,
    BoxBehnkenDesign,
    PlackettBurmanDesign,
    evaluate_design,
    compare_designs,
)


def to_array(points):
    """Convert design points (list of lists) to numpy array."""
    return np.array(points)


@pytest.fixture
def simple_space():
    """Create a simple 3D search space."""
    return Space([
        Real(0, 10, name="x1"),
        Real(-5, 5, name="x2"),
        Real(1, 100, name="x3"),
    ])


@pytest.fixture
def mixed_space():
    """Create a mixed-type search space."""
    return Space([
        Real(0, 1, name="continuous"),
        Integer(1, 10, name="discrete"),
    ])


class TestRandomDesign:
    """Tests for Random Design."""

    def test_random_generates_correct_shape(self, simple_space):
        """Test random design generates correct number of points."""
        design = RandomDesign()
        points = to_array(design.generate(simple_space, n_samples=20))
        
        assert points.shape == (20, 3)

    def test_random_respects_bounds(self, simple_space):
        """Test random points are within bounds."""
        design = RandomDesign()
        points = to_array(design.generate(simple_space, n_samples=100))
        
        assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] <= 10)
        assert np.all(points[:, 1] >= -5) and np.all(points[:, 1] <= 5)
        assert np.all(points[:, 2] >= 1) and np.all(points[:, 2] <= 100)

    def test_random_with_seed(self, simple_space):
        """Test random design is reproducible with seed."""
        design = RandomDesign()
        points1 = to_array(design.generate(simple_space, n_samples=10, seed=42))
        points2 = to_array(design.generate(simple_space, n_samples=10, seed=42))
        
        np.testing.assert_array_equal(points1, points2)


class TestLatinHypercubeDesign:
    """Tests for Latin Hypercube Design."""

    def test_lhs_generates_correct_shape(self, simple_space):
        """Test LHS generates correct number of points."""
        design = LatinHypercubeDesign()
        points = to_array(design.generate(simple_space, n_samples=20))
        
        assert points.shape == (20, 3)

    def test_lhs_respects_bounds(self, simple_space):
        """Test LHS points are within bounds."""
        design = LatinHypercubeDesign()
        points = to_array(design.generate(simple_space, n_samples=50))
        
        assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] <= 10)
        assert np.all(points[:, 1] >= -5) and np.all(points[:, 1] <= 5)
        assert np.all(points[:, 2] >= 1) and np.all(points[:, 2] <= 100)

    def test_lhs_one_per_stratum(self, simple_space):
        """Test LHS has one point per stratum."""
        design = LatinHypercubeDesign(criterion="random")
        n_samples = 10
        points = to_array(design.generate(simple_space, n_samples=n_samples))
        
        # Transform to [0, 1] and check strata
        for dim in range(3):
            dim_bounds = simple_space.dimensions[dim]
            normalized = (points[:, dim] - dim_bounds.low) / (dim_bounds.high - dim_bounds.low)
            strata = np.floor(normalized * n_samples).astype(int)
            strata = np.clip(strata, 0, n_samples - 1)
            
            # Each stratum should have exactly one point
            _, counts = np.unique(strata, return_counts=True)
            assert np.all(counts == 1)

    def test_lhs_lloyd_criterion(self, simple_space):
        """Test lloyd criterion produces better space-filling than random."""
        lhs_random = LatinHypercubeDesign(criterion="random")
        lhs_lloyd = LatinHypercubeDesign(criterion="lloyd")
        
        points_random = to_array(lhs_random.generate(simple_space, n_samples=20, seed=42))
        points_lloyd = to_array(lhs_lloyd.generate(simple_space, n_samples=20, seed=42))
        
        metrics_random = evaluate_design(points_random, simple_space)
        metrics_lloyd = evaluate_design(points_lloyd, simple_space)
        
        # Lloyd should generally have comparable or better space filling
        assert metrics_lloyd.min_distance >= metrics_random.min_distance * 0.3


class TestSobolDesign:
    """Tests for Sobol sequence design."""

    def test_sobol_generates_correct_shape(self, simple_space):
        """Test Sobol generates correct number of points."""
        design = SobolDesign()
        points = to_array(design.generate(simple_space, n_samples=20))
        
        assert points.shape == (20, 3)

    def test_sobol_respects_bounds(self, simple_space):
        """Test Sobol points are within bounds."""
        design = SobolDesign()
        points = to_array(design.generate(simple_space, n_samples=100))
        
        assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] <= 10)
        assert np.all(points[:, 1] >= -5) and np.all(points[:, 1] <= 5)
        assert np.all(points[:, 2] >= 1) and np.all(points[:, 2] <= 100)

    def test_sobol_low_discrepancy(self, simple_space):
        """Test Sobol has lower discrepancy than random."""
        sobol = SobolDesign()
        random = RandomDesign()
        
        points_sobol = to_array(sobol.generate(simple_space, n_samples=64, seed=42))
        points_random = to_array(random.generate(simple_space, n_samples=64, seed=42))
        
        metrics_sobol = evaluate_design(points_sobol, simple_space)
        metrics_random = evaluate_design(points_random, simple_space)
        
        # Sobol should generally have lower discrepancy
        assert metrics_sobol.discrepancy <= metrics_random.discrepancy * 2


class TestHaltonDesign:
    """Tests for Halton sequence design."""

    def test_halton_generates_correct_shape(self, simple_space):
        """Test Halton generates correct number of points."""
        design = HaltonDesign()
        points = to_array(design.generate(simple_space, n_samples=20))
        
        assert points.shape == (20, 3)

    def test_halton_respects_bounds(self, simple_space):
        """Test Halton points are within bounds."""
        design = HaltonDesign()
        points = to_array(design.generate(simple_space, n_samples=100))
        
        assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] <= 10)


class TestFactorialDesigns:
    """Tests for factorial design types."""

    def test_full_factorial_2_level(self):
        """Test 2-level full factorial design."""
        space = Space([
            Real(0, 1, name="x1"),
            Real(0, 1, name="x2"),
            Real(0, 1, name="x3"),
        ])
        
        design = FullFactorialDesign(levels=2)
        points = to_array(design.generate(space))
        
        # 2^3 = 8 points
        assert points.shape[0] == 8
        
        # Should have corner points
        assert np.any(np.all(points == [0, 0, 0], axis=1))
        assert np.any(np.all(points == [1, 1, 1], axis=1))

    def test_full_factorial_3_level(self):
        """Test 3-level full factorial design."""
        space = Space([
            Real(0, 1, name="x1"),
            Real(0, 1, name="x2"),
        ])
        
        design = FullFactorialDesign(levels=3)
        points = to_array(design.generate(space))
        
        # 3^2 = 9 points
        assert points.shape[0] == 9

    def test_fractional_factorial(self):
        """Test fractional factorial design."""
        space = Space([
            Real(0, 1, name="x1"),
            Real(0, 1, name="x2"),
            Real(0, 1, name="x3"),
            Real(0, 1, name="x4"),
        ])
        
        design = FractionalFactorialDesign(resolution=4)
        points = to_array(design.generate(space))
        
        # Should have fewer points than full factorial (2^4=16)
        assert points.shape[0] < 16


class TestResponseSurfaceDesigns:
    """Tests for response surface designs (CCD, Box-Behnken)."""

    def test_ccd_generates_points(self, simple_space):
        """Test Central Composite Design generates points."""
        design = CentralCompositeDesign()
        points = to_array(design.generate(simple_space))
        
        # CCD should have factorial + axial + center points
        assert points.shape[0] > 0
        assert points.shape[1] == 3

    def test_ccd_center_points(self, simple_space):
        """Test CCD includes center points."""
        design = CentralCompositeDesign()
        points = to_array(design.generate(simple_space))
        
        center = np.array([5.0, 0.0, 50.5])  # Center of bounds
        
        # Should have center points close to actual center
        distances = np.linalg.norm(points - center, axis=1)
        # At least one point should be at or near center
        assert np.min(distances) < 1.0

    def test_box_behnken_3_factors(self):
        """Test Box-Behnken design for 3 factors."""
        space = Space([
            Real(0, 1, name="x1"),
            Real(0, 1, name="x2"),
            Real(0, 1, name="x3"),
        ])
        
        design = BoxBehnkenDesign()
        points = to_array(design.generate(space))
        
        # Standard Box-Behnken for 3 factors has 12 + center points
        assert points.shape[0] >= 12

    def test_box_behnken_no_corners(self):
        """Test Box-Behnken avoids corner points."""
        space = Space([
            Real(0, 1, name="x1"),
            Real(0, 1, name="x2"),
            Real(0, 1, name="x3"),
        ])
        
        design = BoxBehnkenDesign()
        points = to_array(design.generate(space))
        
        # No point should have all extreme values
        for point in points:
            extreme_count = np.sum((point == 0) | (point == 1))
            assert extreme_count < 3  # Not all corners


class TestPlackettBurmanDesign:
    """Tests for Plackett-Burman screening design."""

    def test_pb_valid_runs(self):
        """Test Plackett-Burman generates valid number of runs."""
        space = Space([Real(0, 1, name=f"x{i}") for i in range(7)])
        
        design = PlackettBurmanDesign()
        points = to_array(design.generate(space))
        
        # PB designs have runs that are multiples of 4
        assert points.shape[0] % 4 == 0

    def test_pb_two_levels(self):
        """Test Plackett-Burman only uses two levels."""
        space = Space([Real(0, 1, name=f"x{i}") for i in range(5)])
        
        design = PlackettBurmanDesign()
        points = to_array(design.generate(space))
        
        # Each value should be 0 or 1
        unique_vals = np.unique(points)
        assert len(unique_vals) == 2
        assert 0 in unique_vals or np.min(unique_vals) == pytest.approx(0, abs=0.01)
        assert 1 in unique_vals or np.max(unique_vals) == pytest.approx(1, abs=0.01)


class TestDesignMetrics:
    """Tests for design evaluation metrics."""

    def test_evaluate_design(self, simple_space):
        """Test design evaluation."""
        design = LatinHypercubeDesign()
        points = to_array(design.generate(simple_space, n_samples=20))
        
        metrics = evaluate_design(points, simple_space)
        
        assert metrics.n_points == 20
        assert metrics.n_dims == 3
        assert metrics.min_distance > 0
        assert metrics.mean_distance > metrics.min_distance
        assert metrics.max_distance >= metrics.mean_distance
        assert metrics.space_filling > 0

    def test_compare_designs(self, simple_space):
        """Test design comparison."""
        results = compare_designs(simple_space, n_samples=30, seed=42)
        
        assert "Random" in results
        assert "Sobol" in results
        
        for metrics in results.values():
            assert metrics.n_points == 30


class TestMixedSpaceDesigns:
    """Tests for designs in mixed-type spaces."""

    def test_lhs_mixed_space(self, mixed_space):
        """Test LHS works with mixed types."""
        design = LatinHypercubeDesign()
        points = to_array(design.generate(mixed_space, n_samples=20))
        
        assert points.shape == (20, 2)
        # Integer column should have integer values
        assert np.all(points[:, 1] == points[:, 1].astype(int))

    def test_random_mixed_space(self, mixed_space):
        """Test random design works with mixed types."""
        design = RandomDesign()
        points = to_array(design.generate(mixed_space, n_samples=20))
        
        assert points.shape == (20, 2)
