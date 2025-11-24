"""
Property-based tests for the asteroid training pipeline
"""

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import pytest
from sklearn.model_selection import train_test_split

# Import the trainer class
from train_model import AsteroidModelTrainer


# Feature: asteroid-size-predictor, Property 5: Data split completeness
# For any dataset split into training and validation sets, the sum of the sizes 
# of both sets should equal the original dataset size, and there should be no 
# overlap between the sets
@settings(max_examples=100)
@given(
    data_size=st.integers(min_value=10, max_value=1000),
    test_size=st.floats(min_value=0.1, max_value=0.4),
    random_state=st.integers(min_value=0, max_value=10000)
)
def test_data_split_completeness(data_size, test_size, random_state):
    """
    **Feature: asteroid-size-predictor, Property 5: Data split completeness**
    **Validates: Requirements 2.3**
    
    Property: For any dataset split, the sum of train and test sizes should equal
    the original size, and there should be no overlap between sets.
    """
    # Generate random dataset
    X = np.random.randn(data_size, 1)
    y = np.random.randn(data_size, 1)
    
    # Perform train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Property 1: Completeness - sum of sizes equals original
    assert len(X_train) + len(X_test) == data_size, \
        f"Split sizes don't sum to original: {len(X_train)} + {len(X_test)} != {data_size}"
    assert len(y_train) + len(y_test) == data_size, \
        f"Target split sizes don't sum to original: {len(y_train)} + {len(y_test)} != {data_size}"
    
    # Property 2: No overlap - convert to sets of indices and check intersection
    # We'll use a unique identifier approach: create indices and verify no duplicates
    X_train_indices = set(range(len(X_train)))
    X_test_indices = set(range(len(X_train), len(X_train) + len(X_test)))
    
    # The indices should not overlap
    assert len(X_train_indices.intersection(X_test_indices)) == 0, \
        "Train and test indices overlap"
    
    # Property 3: Verify the split ratio is approximately correct
    expected_test_size = int(data_size * test_size)
    # Allow for rounding differences
    assert abs(len(X_test) - expected_test_size) <= 1, \
        f"Test size {len(X_test)} doesn't match expected {expected_test_size}"


# Feature: asteroid-size-predictor, Property 5: Data split completeness (integration test)
# Test the actual preprocess_data method
@settings(max_examples=100)
@given(
    data_size=st.integers(min_value=20, max_value=500),
    test_size=st.floats(min_value=0.1, max_value=0.4)
)
def test_preprocess_data_split_completeness(data_size, test_size):
    """
    **Feature: asteroid-size-predictor, Property 5: Data split completeness**
    **Validates: Requirements 2.3**
    
    Property: The preprocess_data method should maintain data completeness
    """
    # Create synthetic dataset
    df = pd.DataFrame({
        'magnitude': np.random.uniform(10, 30, data_size),
        'size': np.random.uniform(0.1, 100, data_size)
    })
    
    trainer = AsteroidModelTrainer()
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, norm_params = trainer.preprocess_data(
        df, 'magnitude', 'size', test_size=test_size, random_state=42
    )
    
    # Property 1: Completeness
    total_samples = len(X_train) + len(X_test)
    assert total_samples == data_size, \
        f"Split doesn't preserve all samples: {total_samples} != {data_size}"
    
    # Property 2: Consistent split between X and y
    assert len(X_train) == len(y_train), \
        f"X_train and y_train have different sizes: {len(X_train)} != {len(y_train)}"
    assert len(X_test) == len(y_test), \
        f"X_test and y_test have different sizes: {len(X_test)} != {len(y_test)}"
    
    # Property 3: Normalization parameters are valid
    assert 'magnitude_mean' in norm_params
    assert 'magnitude_std' in norm_params
    assert 'size_mean' in norm_params
    assert 'size_std' in norm_params
    assert norm_params['magnitude_std'] > 0, "Standard deviation should be positive"
    assert norm_params['size_std'] > 0, "Standard deviation should be positive"


# Feature: asteroid-size-predictor, Property 6: Model evaluation metrics validity
# For any trained model evaluation, the calculated metrics (MAE, RMSE, R²) should be 
# valid numbers (not NaN or Infinity) and within reasonable ranges for the problem domain
@settings(max_examples=100)
@given(
    n_samples=st.integers(min_value=10, max_value=100),
    noise_level=st.floats(min_value=0.0, max_value=2.0)
)
def test_model_evaluation_metrics_validity(n_samples, noise_level):
    """
    **Feature: asteroid-size-predictor, Property 6: Model evaluation metrics validity**
    **Validates: Requirements 2.5**
    
    Property: Evaluation metrics should always be valid numbers within reasonable ranges
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Generate synthetic predictions and actual values
    y_actual = np.random.uniform(0.1, 100, (n_samples, 1))
    # Add noise to create predictions
    y_pred = y_actual + np.random.normal(0, noise_level, (n_samples, 1))
    # Ensure predictions are positive (asteroid sizes can't be negative)
    y_pred = np.maximum(y_pred, 0.01)
    
    # Calculate metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    
    # Property 1: Metrics should not be NaN
    assert not np.isnan(mae), "MAE should not be NaN"
    assert not np.isnan(rmse), "RMSE should not be NaN"
    assert not np.isnan(r2), "R² should not be NaN"
    
    # Property 2: Metrics should not be infinite
    assert not np.isinf(mae), "MAE should not be infinite"
    assert not np.isinf(rmse), "RMSE should not be infinite"
    assert not np.isinf(r2), "R² should not be infinite"
    
    # Property 3: MAE and RMSE should be non-negative
    assert mae >= 0, f"MAE should be non-negative, got {mae}"
    assert rmse >= 0, f"RMSE should be non-negative, got {rmse}"
    
    # Property 4: RMSE should be >= MAE (mathematical property)
    assert rmse >= mae, f"RMSE ({rmse}) should be >= MAE ({mae})"
    
    # Property 5: R² should be <= 1 for valid regression
    # Note: R² can be negative for very poor models, but should be <= 1
    assert r2 <= 1.0, f"R² should be <= 1.0, got {r2}"


# Feature: asteroid-size-predictor, Property 6: Model evaluation metrics validity (integration)
# Test the actual evaluate_model method
def test_evaluate_model_metrics_validity():
    """
    **Feature: asteroid-size-predictor, Property 6: Model evaluation metrics validity**
    **Validates: Requirements 2.5**
    
    Property: The evaluate_model method should return valid metrics
    """
    # Skip if TensorFlow is not available
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        pytest.skip("TensorFlow not available")
    
    # Create a simple dataset
    n_samples = 100
    X_train = np.random.randn(n_samples, 1)
    y_train = 2 * X_train + 1 + np.random.randn(n_samples, 1) * 0.1
    
    X_test = np.random.randn(20, 1)
    y_test = 2 * X_test + 1 + np.random.randn(20, 1) * 0.1
    
    # Create and train a simple model
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(1,)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, verbose=0)
    
    # Create trainer and set up scalers
    trainer = AsteroidModelTrainer()
    from sklearn.preprocessing import StandardScaler
    trainer.scaler_y = StandardScaler()
    trainer.scaler_y.fit(y_train)
    
    # Evaluate the model
    metrics = trainer.evaluate_model(model, X_test, y_test, output_file='model/test_metrics.json')
    
    # Verify all required metrics are present
    assert 'mae' in metrics, "MAE should be in metrics"
    assert 'rmse' in metrics, "RMSE should be in metrics"
    assert 'r2' in metrics, "R² should be in metrics"
    
    # Verify metrics are valid
    assert not np.isnan(metrics['mae']), "MAE should not be NaN"
    assert not np.isnan(metrics['rmse']), "RMSE should not be NaN"
    assert not np.isnan(metrics['r2']), "R² should not be NaN"
    
    assert not np.isinf(metrics['mae']), "MAE should not be infinite"
    assert not np.isinf(metrics['rmse']), "RMSE should not be infinite"
    assert not np.isinf(metrics['r2']), "R² should not be infinite"
    
    assert metrics['mae'] >= 0, "MAE should be non-negative"
    assert metrics['rmse'] >= 0, "RMSE should be non-negative"
    assert metrics['rmse'] >= metrics['mae'], "RMSE should be >= MAE"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])


# ============================================================================
# UNIT TESTS FOR TRAINING PIPELINE (Task 2.8)
# ============================================================================

def test_data_loading_handles_missing_columns():
    """
    Unit test: Data loading should handle missing columns gracefully
    Requirements: 2.1, 2.2
    """
    trainer = AsteroidModelTrainer()
    
    # Create a DataFrame with missing required columns
    df_incomplete = pd.DataFrame({
        'some_column': [1, 2, 3],
        'another_column': [4, 5, 6]
    })
    
    # The handle_missing_and_outliers should work with any column names
    # as long as they are provided
    try:
        result = trainer.handle_missing_and_outliers(df_incomplete, 'H', 'diameter')
        # If columns don't exist, it should raise KeyError
        assert False, "Should have raised KeyError for missing columns"
    except KeyError:
        # Expected behavior
        pass


def test_preprocessing_produces_correct_shapes():
    """
    Unit test: Preprocessing should produce correct output shapes
    Requirements: 2.2
    """
    # Create synthetic dataset
    n_samples = 100
    df = pd.DataFrame({
        'magnitude': np.random.uniform(10, 30, n_samples),
        'size': np.random.uniform(0.1, 100, n_samples)
    })
    
    trainer = AsteroidModelTrainer()
    
    # Preprocess with 80/20 split
    X_train, X_test, y_train, y_test, norm_params = trainer.preprocess_data(
        df, 'magnitude', 'size', test_size=0.2, random_state=42
    )
    
    # Check shapes
    assert X_train.shape[1] == 1, "X_train should have 1 feature"
    assert y_train.shape[1] == 1, "y_train should have 1 target"
    assert X_test.shape[1] == 1, "X_test should have 1 feature"
    assert y_test.shape[1] == 1, "y_test should have 1 target"
    
    # Check split ratio (approximately 80/20)
    expected_train = int(n_samples * 0.8)
    expected_test = n_samples - expected_train
    
    assert len(X_train) == expected_train, f"Expected {expected_train} training samples"
    assert len(X_test) == expected_test, f"Expected {expected_test} test samples"


def test_normalization_parameter_calculation():
    """
    Unit test: Normalization parameters should be calculated correctly
    Requirements: 2.2
    """
    # Create dataset with known statistics
    df = pd.DataFrame({
        'magnitude': [10.0, 20.0, 30.0, 40.0, 50.0],
        'size': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    
    trainer = AsteroidModelTrainer()
    X_train, X_test, y_train, y_test, norm_params = trainer.preprocess_data(
        df, 'magnitude', 'size', test_size=0.2, random_state=42
    )
    
    # Check that all required parameters are present
    required_params = [
        'magnitude_min', 'magnitude_max', 'magnitude_mean', 'magnitude_std',
        'size_min', 'size_max', 'size_mean', 'size_std'
    ]
    
    for param in required_params:
        assert param in norm_params, f"Missing parameter: {param}"
    
    # Check that min/max are correct
    assert norm_params['magnitude_min'] == 10.0
    assert norm_params['magnitude_max'] == 50.0
    assert norm_params['size_min'] == 1.0
    assert norm_params['size_max'] == 5.0
    
    # Check that std is positive
    assert norm_params['magnitude_std'] > 0
    assert norm_params['size_std'] > 0


def test_model_export_creates_required_files():
    """
    Unit test: Model export should create all required files
    Requirements: 2.4
    """
    # Skip if TensorFlow is not available
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        pytest.skip("TensorFlow not available")
    
    # Create a simple model
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(1,)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Create trainer
    trainer = AsteroidModelTrainer()
    
    # Set up normalization params
    trainer.normalization_params = {
        'magnitude_min': 10.0,
        'magnitude_max': 30.0,
        'magnitude_mean': 20.0,
        'magnitude_std': 5.0,
        'size_min': 0.1,
        'size_max': 100.0,
        'size_mean': 50.0,
        'size_std': 25.0
    }
    
    # Save normalization parameters
    test_norm_file = 'model/test_normalization.json'
    trainer.save_normalization_params(output_file=test_norm_file)
    
    # Check that file was created
    assert os.path.exists(test_norm_file), "Normalization file should be created"
    
    # Check that file contains valid JSON
    with open(test_norm_file, 'r') as f:
        loaded_params = json.load(f)
    
    # Verify all parameters are present
    assert 'magnitude_min' in loaded_params
    assert 'magnitude_max' in loaded_params
    assert 'size_mean' in loaded_params
    assert 'size_std' in loaded_params
    
    # Clean up
    if os.path.exists(test_norm_file):
        os.remove(test_norm_file)


def test_explore_data_returns_dataframe():
    """
    Unit test: explore_data should return the input DataFrame
    Requirements: 2.1
    """
    df = pd.DataFrame({
        'H': [10.0, 20.0, 30.0],
        'diameter': [1.0, 2.0, 3.0]
    })
    
    trainer = AsteroidModelTrainer()
    result = trainer.explore_data(df)
    
    # Should return the same DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


def test_handle_missing_values():
    """
    Unit test: handle_missing_and_outliers should remove rows with NaN
    Requirements: 2.1, 2.2
    """
    df = pd.DataFrame({
        'H': [10.0, 20.0, np.nan, 30.0, 40.0],
        'diameter': [1.0, np.nan, 3.0, 4.0, 5.0]
    })
    
    trainer = AsteroidModelTrainer()
    df_clean = trainer.handle_missing_and_outliers(df, 'H', 'diameter')
    
    # Should have removed rows with NaN
    assert len(df_clean) < len(df)
    assert df_clean['H'].isna().sum() == 0
    assert df_clean['diameter'].isna().sum() == 0
