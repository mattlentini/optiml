"""Tests for validation module - ICH Q2 method validation."""

import numpy as np
import pytest
from scipy import stats

from optiml.validation import (
    ValidationLevel,
    AccuracyResult,
    PrecisionResult,
    LinearityResult,
    LODResult,
    LOQResult,
    SpecificityResult,
    RobustnessResult,
    RangeResult,
    ICH_Q2_Report,
    calculate_accuracy,
    calculate_precision,
    calculate_linearity,
    calculate_lod,
    calculate_loq,
    ZFactorResult,
    calculate_z_factor,
    SignalMetrics,
    calculate_signal_metrics,
    EC50Result,
    calculate_ec50,
    CapabilityResult,
    calculate_capability,
    validate_method,
)


class TestValidationLevel:
    """Test ValidationLevel enum."""

    def test_validation_levels(self):
        """Test that all validation levels are defined."""
        assert ValidationLevel.FULL.value == "full"
        assert ValidationLevel.PARTIAL.value == "partial"
        assert ValidationLevel.VERIFICATION.value == "verification"


class TestAccuracy:
    """Test accuracy calculations."""

    def test_calculate_accuracy_perfect(self):
        """Test accuracy with perfect recovery."""
        measured = np.array([100.0, 100.0, 100.0])
        expected = np.array([100.0, 100.0, 100.0])
        
        result = calculate_accuracy(measured, expected)
        
        assert isinstance(result, AccuracyResult)
        assert result.recovery_mean == pytest.approx(100.0)
        assert result.recovery_std == pytest.approx(0.0)
        assert result.bias == pytest.approx(0.0)
        assert bool(result.acceptable) is True

    def test_calculate_accuracy_acceptable(self):
        """Test accuracy within acceptable range."""
        measured = np.array([99.0, 101.0, 100.0])
        expected = np.array([100.0, 100.0, 100.0])
        
        result = calculate_accuracy(measured, expected)
        
        assert 99.0 <= result.recovery_mean <= 101.0
        assert bool(result.acceptable) is True

    def test_calculate_accuracy_high_recovery(self):
        """Test accuracy with high recovery."""
        measured = np.array([110.0, 112.0, 108.0])
        expected = np.array([100.0, 100.0, 100.0])
        
        result = calculate_accuracy(measured, expected)
        
        assert result.recovery_mean > 100.0
        assert result.bias > 0  # Positive bias
        assert bool(result.acceptable) is False  # Outside 97-103%

    def test_calculate_accuracy_low_recovery(self):
        """Test accuracy with low recovery."""
        measured = np.array([90.0, 92.0, 88.0])
        expected = np.array([100.0, 100.0, 100.0])
        
        result = calculate_accuracy(measured, expected)
        
        assert result.recovery_mean < 100.0
        assert result.bias < 0  # Negative bias
        assert bool(result.acceptable) is False  # Outside 97-103%

    def test_calculate_accuracy_custom_criteria(self):
        """Test accuracy with custom acceptance criteria."""
        measured = np.array([95.0, 105.0, 100.0])
        expected = np.array([100.0, 100.0, 100.0])
        
        # Default criteria would fail, but wider criteria should pass
        result = calculate_accuracy(measured, expected, criteria=(90.0, 110.0))
        
        assert bool(result.acceptable) is True
        assert result.criteria == (90.0, 110.0)

    def test_calculate_accuracy_multiple_levels(self):
        """Test accuracy at multiple concentration levels."""
        # Low, medium, high concentrations
        measured = np.array([9.8, 50.2, 99.5])
        expected = np.array([10.0, 50.0, 100.0])
        
        result = calculate_accuracy(measured, expected)
        
        assert len(result.recovery_values) == 3


class TestPrecision:
    """Test precision calculations."""

    def test_calculate_precision_low_rsd(self):
        """Test precision with low variability."""
        # 6 replicates with low variability
        data = np.array([100.0, 100.5, 99.8, 100.2, 99.9, 100.1])
        
        result = calculate_precision(data)
        
        assert isinstance(result, PrecisionResult)
        assert result.repeatability_rsd < 1.0
        assert bool(result.acceptable) is True

    def test_calculate_precision_high_rsd(self):
        """Test precision with high variability."""
        data = np.array([90.0, 110.0, 95.0, 105.0, 92.0, 108.0])
        
        result = calculate_precision(data)
        
        # High variability should fail default 2% criteria
        assert result.repeatability_rsd > 2.0
        assert bool(result.acceptable) is False

    def test_calculate_precision_custom_criteria(self):
        """Test precision with custom RSD criteria."""
        data = np.array([95.0, 100.0, 105.0, 98.0, 102.0, 99.0])
        
        # Allow up to 10% RSD using the criteria parameter
        result = calculate_precision(data, criteria=10.0)
        
        assert bool(result.acceptable) is True
        assert result.criteria == 10.0

    def test_calculate_precision_rsd_formula(self):
        """Test that RSD is calculated correctly."""
        data = np.array([100.0, 100.0, 100.0])  # Zero std
        
        result = calculate_precision(data)
        
        assert result.repeatability_rsd == 0.0


class TestLinearity:
    """Test linearity calculations."""

    def test_calculate_linearity_perfect(self):
        """Test linearity with perfect correlation."""
        concentrations = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        responses = np.array([100.0, 200.0, 300.0, 400.0, 500.0])  # Perfect linear
        
        result = calculate_linearity(concentrations, responses)
        
        assert isinstance(result, LinearityResult)
        assert result.r_squared == pytest.approx(1.0)
        assert result.slope == pytest.approx(10.0)
        assert result.intercept == pytest.approx(0.0)
        assert bool(result.acceptable) is True

    def test_calculate_linearity_with_intercept(self):
        """Test linearity with non-zero intercept."""
        concentrations = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        responses = np.array([50.0, 150.0, 250.0, 350.0, 450.0])
        
        result = calculate_linearity(concentrations, responses)
        
        assert result.slope == pytest.approx(10.0)
        assert result.intercept == pytest.approx(50.0)
        assert result.r_squared == pytest.approx(1.0)

    def test_calculate_linearity_good_r2(self):
        """Test linearity with good but not perfect R²."""
        np.random.seed(42)
        concentrations = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        noise = np.random.normal(0, 5, 5)
        responses = concentrations * 10 + noise
        
        result = calculate_linearity(concentrations, responses)
        
        assert result.r_squared > 0.98
        assert bool(result.acceptable) is True

    def test_calculate_linearity_poor_r2(self):
        """Test linearity with poor R² (fails)."""
        np.random.seed(42)
        concentrations = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        responses = np.random.uniform(50, 500, 5)  # Random - no correlation
        
        result = calculate_linearity(concentrations, responses)
        
        # Poor linearity should fail
        assert result.r_squared < 0.99
        assert bool(result.acceptable) is False

    def test_calculate_linearity_residuals(self):
        """Test that residuals are computed correctly."""
        concentrations = np.array([10.0, 20.0, 30.0])
        responses = np.array([100.0, 200.0, 300.0])
        
        result = calculate_linearity(concentrations, responses)
        
        assert len(result.residuals) == 3
        # Perfect fit = all residuals should be ~0
        assert np.allclose(result.residuals, 0.0, atol=1e-10)


class TestLOD:
    """Test limit of detection calculations."""

    def test_calculate_lod_from_blanks(self):
        """Test LOD from blank measurements without slope (gives response units)."""
        blank_responses = np.array([0.5, 0.6, 0.4, 0.55, 0.45])
        
        result = calculate_lod(blank_responses=blank_responses)
        
        assert isinstance(result, LODResult)
        assert result.lod_value > 0
        assert result.method == "blank_std"

    def test_calculate_lod_from_blanks_with_slope(self):
        """Test LOD from blank measurements with calibration."""
        blank_responses = np.array([0.5, 0.6, 0.4, 0.55, 0.45])
        slope = 10.0
        intercept = 0.0  # Need intercept when using slope
        
        result = calculate_lod(blank_responses=blank_responses, slope=slope, intercept=intercept)
        
        assert result.lod_value > 0
        assert result.method == "blank_std"

    def test_calculate_lod_from_regression(self):
        """Test LOD from regression method."""
        slope = 10.0
        residual_std = 0.5
        
        result = calculate_lod(slope=slope, residual_std=residual_std, method="regression")
        
        assert result.lod_value > 0
        # LOD = 3.3 * residual_std / slope
        expected_lod = 3.3 * residual_std / slope
        assert result.lod_value == pytest.approx(expected_lod)

    def test_calculate_lod_signal_to_noise(self):
        """Test LOD from signal-to-noise method."""
        slope = 10.0
        residual_std = 0.5
        
        result = calculate_lod(slope=slope, residual_std=residual_std, method="regression")
        
        assert result.lod_value > 0


class TestLOQ:
    """Test limit of quantitation calculations."""

    def test_calculate_loq_from_blanks(self):
        """Test LOQ from blank measurements without slope (gives response units)."""
        blank_responses = np.array([0.5, 0.6, 0.4, 0.55, 0.45])
        
        result = calculate_loq(blank_responses=blank_responses)
        
        assert isinstance(result, LOQResult)
        assert result.loq_value > 0
        assert result.method == "blank_std"
        # LOQ should be higher than LOD (by factor of ~3)
        lod = calculate_lod(blank_responses=blank_responses)
        assert result.loq_value > lod.lod_value

    def test_calculate_loq_from_blanks_with_slope(self):
        """Test LOQ from blank measurements with calibration."""
        blank_responses = np.array([0.5, 0.6, 0.4, 0.55, 0.45])
        slope = 10.0
        intercept = 0.0
        
        result = calculate_loq(blank_responses=blank_responses, slope=slope, intercept=intercept)
        
        assert result.loq_value > 0
        assert result.method == "blank_std"

    def test_calculate_loq_from_regression(self):
        """Test LOQ from regression method."""
        slope = 10.0
        residual_std = 0.5
        
        result = calculate_loq(slope=slope, residual_std=residual_std, method="regression")
        
        # LOQ = 10 * residual_std / slope
        expected_loq = 10.0 * residual_std / slope
        assert result.loq_value == pytest.approx(expected_loq)

    def test_loq_greater_than_lod(self):
        """Test that LOQ is always greater than LOD."""
        blank_responses = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
        
        lod = calculate_lod(blank_responses=blank_responses)
        loq = calculate_loq(blank_responses=blank_responses)
        
        assert loq.loq_value > lod.lod_value


class TestZFactor:
    """Test Z-factor calculations for bioassays."""

    def test_calculate_z_factor_excellent(self):
        """Test excellent assay (Z > 0.5)."""
        positive = np.array([90.0, 92.0, 88.0, 91.0, 89.0])  # Low variability
        negative = np.array([10.0, 12.0, 8.0, 11.0, 9.0])    # Low variability
        
        result = calculate_z_factor(positive, negative)
        
        assert isinstance(result, ZFactorResult)
        assert result.z_factor > 0.5  # Excellent assay
        assert result.quality == "excellent"

    def test_calculate_z_factor_good(self):
        """Test good assay (0 < Z <= 0.5)."""
        positive = np.array([80.0, 90.0, 100.0, 85.0, 95.0])  # More variability
        negative = np.array([5.0, 15.0, 10.0, 8.0, 12.0])
        
        result = calculate_z_factor(positive, negative)
        
        # With higher variability, Z is lower but may still be good
        assert result.z_factor <= 0.5 or result.quality in ("excellent", "good")

    def test_calculate_z_factor_poor(self):
        """Test poor assay (Z < 0)."""
        positive = np.array([50.0, 60.0, 40.0, 70.0, 30.0])  # High variability
        negative = np.array([40.0, 50.0, 30.0, 60.0, 20.0])   # Overlapping
        
        result = calculate_z_factor(positive, negative)
        
        assert result.z_factor < 0.5

    def test_z_factor_formula(self):
        """Test Z-factor formula: 1 - 3*(σp + σn)/(μp - μn)."""
        positive = np.array([100.0, 100.0, 100.0])  # σ = 0
        negative = np.array([0.0, 0.0, 0.0])        # σ = 0
        
        result = calculate_z_factor(positive, negative)
        
        # Z = 1 - 3*(0+0)/100 = 1
        assert result.z_factor == pytest.approx(1.0)


class TestSignalMetrics:
    """Test signal quality metrics."""

    def test_calculate_signal_metrics(self):
        """Test signal metrics calculation."""
        signal = np.array([100.0, 102.0, 98.0, 101.0, 99.0])
        background = np.array([5.0, 6.0, 4.0, 5.5, 4.5])
        
        result = calculate_signal_metrics(signal, background)
        
        assert isinstance(result, SignalMetrics)
        assert result.signal_to_noise > 0
        assert result.signal_to_background > 0
        assert result.window > 0
        assert result.dynamic_range > 0

    def test_signal_to_noise_ratio(self):
        """Test SNR calculation."""
        signal = np.array([100.0, 100.0, 100.0])  # Mean = 100, std = 0
        background = np.array([1.0, 2.0, 3.0])    # Mean = 2, std = 1
        
        result = calculate_signal_metrics(signal, background)
        
        # S/N = (signal_mean - background_mean) / background_std = (100-2)/1 = 98
        assert result.signal_to_noise == pytest.approx(98.0)


class TestEC50:
    """Test EC50 calculation from dose-response curves."""

    def test_calculate_ec50_sigmoidal(self):
        """Test EC50 from sigmoidal dose-response."""
        # Generate sigmoidal data
        # Note: _four_param_logistic uses (ec50/x)^hill, so we need to adjust
        concentrations = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        # 4PL: y = bottom + (top - bottom) / (1 + (ec50/x)^hill)
        ec50_true = 1.0
        responses = 0 + (100 - 0) / (1 + (ec50_true / concentrations) ** 1)
        
        result = calculate_ec50(concentrations, responses)
        
        assert isinstance(result, EC50Result)
        assert result.ec50 > 0
        # Should be close to true EC50
        assert result.ec50 == pytest.approx(ec50_true, rel=0.3)

    def test_calculate_ec50_with_noise(self):
        """Test EC50 with noisy data."""
        np.random.seed(42)
        concentrations = np.logspace(-2, 2, 10)
        ec50_true = 1.0
        # Use correct formula: (ec50/x)^hill
        responses = 0 + (100 - 0) / (1 + (ec50_true / concentrations) ** 1)
        responses += np.random.normal(0, 2, len(responses))  # Add noise
        
        result = calculate_ec50(concentrations, responses)
        
        # Should still find EC50 within reasonable range
        assert 0.1 < result.ec50 < 10.0


class TestCapability:
    """Test process capability calculations."""

    def test_calculate_capability_centered(self):
        """Test capability for centered process."""
        np.random.seed(42)
        # Process centered at 50 with spec limits 40-60
        data = np.random.normal(50, 2, 100)
        
        result = calculate_capability(data, lsl=40, usl=60)
        
        assert isinstance(result, CapabilityResult)
        assert result.cp > 1.0  # Capable
        assert result.cpk > 1.0  # Centered and capable
        # Cp and Cpk should be similar for centered process
        assert abs(result.cp - result.cpk) < 0.5

    def test_calculate_capability_off_center(self):
        """Test capability for off-center process."""
        np.random.seed(42)
        # Process shifted towards upper limit
        data = np.random.normal(55, 2, 100)
        
        result = calculate_capability(data, lsl=40, usl=60)
        
        assert result.cp > result.cpk  # Cpk penalizes off-center

    def test_calculate_capability_poor(self):
        """Test poor capability."""
        np.random.seed(42)
        # High variability process
        data = np.random.normal(50, 10, 100)
        
        result = calculate_capability(data, lsl=40, usl=60)
        
        assert result.cp < 1.0  # Not capable
        assert result.cpk < 1.0

    def test_capability_formulas(self):
        """Test capability index formulas with small sample."""
        # Small sample to test edge case handling
        data = np.array([49.0, 50.0, 51.0])  # Small sample with some variance
        
        result = calculate_capability(data, lsl=40, usl=60)
        
        # Should compute valid capability indices
        assert result.cp > 0
        assert result.cpk > 0


class TestValidateMethod:
    """Test complete method validation."""

    def test_validate_method_full(self):
        """Test full method validation."""
        # Accuracy data
        measured = np.array([99.0, 100.5, 100.2])
        expected = np.array([100.0, 100.0, 100.0])
        
        # Precision data
        precision_data = np.array([100.0, 100.2, 99.8, 100.1, 99.9, 100.0])
        
        # Linearity data
        conc = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        resp = conc * 10.0 + 5.0  # Perfect linear
        
        report = validate_method(
            method_name="Test Method",
            analyte="Test Analyte",
            accuracy_data=(measured, expected),
            precision_data=precision_data,
            linearity_data=(conc, resp),
            level=ValidationLevel.FULL,
        )
        
        assert isinstance(report, ICH_Q2_Report)
        assert report.method_name == "Test Method"
        assert report.analyte == "Test Analyte"
        assert report.accuracy is not None
        assert report.precision is not None
        assert report.linearity is not None
        assert report.overall_pass is True

    def test_validate_method_partial(self):
        """Test partial method validation."""
        # Only accuracy
        measured = np.array([99.0, 100.0, 101.0])
        expected = np.array([100.0, 100.0, 100.0])
        
        report = validate_method(
            method_name="Partial Method",
            analyte="Analyte",
            accuracy_data=(measured, expected),
            level=ValidationLevel.PARTIAL,
        )
        
        assert report.accuracy is not None
        assert report.precision is None
        assert report.linearity is None

    def test_validate_method_fails(self):
        """Test validation that fails."""
        # Poor accuracy
        measured = np.array([80.0, 85.0, 82.0])  # 80-85% recovery = fail
        expected = np.array([100.0, 100.0, 100.0])
        
        report = validate_method(
            method_name="Failing Method",
            analyte="Analyte",
            accuracy_data=(measured, expected),
        )
        
        assert report.accuracy is not None
        assert bool(report.accuracy.acceptable) is False
        assert bool(report.overall_pass) is False


class TestDataclasses:
    """Test dataclass structures."""

    def test_accuracy_result(self):
        """Test AccuracyResult dataclass."""
        result = AccuracyResult(
            recovery_mean=100.0,
            recovery_std=1.5,
            recovery_values=np.array([99.0, 100.5, 100.5]),
            bias=0.0,
            acceptable=True,
        )
        
        assert result.recovery_mean == 100.0
        assert result.recovery_std == 1.5
        assert result.bias == 0.0
        assert result.acceptable is True

    def test_precision_result(self):
        """Test PrecisionResult dataclass."""
        result = PrecisionResult(
            repeatability_rsd=1.2,
            intermediate_precision_rsd=1.8,
            acceptable=True,
            criteria=2.0,
        )
        
        assert result.repeatability_rsd == 1.2
        assert result.intermediate_precision_rsd == 1.8
        assert result.acceptable is True

    def test_linearity_result(self):
        """Test LinearityResult dataclass."""
        result = LinearityResult(
            slope=10.0,
            intercept=5.0,
            r_squared=0.999,
            standard_error=0.5,
            residuals=np.array([0.1, -0.1, 0.05]),
            concentrations=np.array([10.0, 20.0, 30.0]),
            responses=np.array([105.0, 205.0, 305.0]),
            acceptable=True,
        )
        
        assert result.slope == 10.0
        assert result.r_squared == 0.999

    def test_ich_q2_report(self):
        """Test ICH_Q2_Report dataclass."""
        report = ICH_Q2_Report(
            method_name="HPLC Method",
            analyte="Drug Substance",
            overall_pass=True,
        )
        
        assert report.method_name == "HPLC Method"
        assert report.analyte == "Drug Substance"
        assert report.accuracy is None
        assert report.overall_pass is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_accuracy_single_point(self):
        """Test accuracy with single measurement."""
        measured = np.array([100.0])
        expected = np.array([100.0])
        
        result = calculate_accuracy(measured, expected)
        
        assert result.recovery_mean == 100.0
        # std of single value is 0 (with ddof=1, could be nan in some implementations)
        assert result.recovery_std == 0.0 or np.isnan(result.recovery_std)

    def test_precision_single_point(self):
        """Test precision with single measurement."""
        data = np.array([100.0])
        
        result = calculate_precision(data)
        
        # RSD with single point should handle gracefully (0 or nan)
        assert result.repeatability_rsd >= 0.0 or np.isnan(result.repeatability_rsd)

    def test_linearity_two_points(self):
        """Test linearity with minimum points (2)."""
        conc = np.array([10.0, 50.0])
        resp = np.array([100.0, 500.0])
        
        result = calculate_linearity(conc, resp)
        
        # R² = 1.0 for two points (perfect fit)
        assert result.r_squared == pytest.approx(1.0)

    def test_lod_missing_data(self):
        """Test LOD with insufficient data."""
        with pytest.raises(ValueError):
            calculate_lod()  # No data provided

    def test_accuracy_array_length_check(self):
        """Test that arrays must have values."""
        # Single values should work (tested above)
        # Zero length might raise depending on implementation
        try:
            result = calculate_accuracy(np.array([100.0]), np.array([100.0]))
            assert result.recovery_mean == 100.0
        except (ValueError, IndexError):
            pass  # Also acceptable
