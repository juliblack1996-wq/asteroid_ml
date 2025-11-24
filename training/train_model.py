"""
Asteroid Size Predictor - Training Pipeline
Trains a machine learning model to predict asteroid size from absolute magnitude
"""

import os
import json
import numpy as np
import pandas as pd

# Optional imports - will be checked at runtime
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("Warning: kagglehub not installed. Install with: pip install kagglehub")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: tensorflow not installed. Install with: pip install tensorflow")

try:
    import tensorflowjs as tfjs
    TFJS_AVAILABLE = True
except ImportError:
    TFJS_AVAILABLE = False
    print("Warning: tensorflowjs not installed. Install with: pip install tensorflowjs")


class AsteroidModelTrainer:
    """Handles the complete training pipeline for asteroid size prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        if SKLEARN_AVAILABLE:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        self.normalization_params = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, dataset_name="sakhawat18/asteroid-dataset"):
        """
        Load asteroid dataset from Kaggle using kagglehub
        
        Args:
            dataset_name: Kaggle dataset identifier
            
        Returns:
            pd.DataFrame: Loaded asteroid data
        """
        print(f"Downloading dataset: {dataset_name}")
        
        # Download dataset using kagglehub
        path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {path}")
        
        # Find the CSV file in the downloaded path
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        
        # Load the first CSV file found
        csv_path = os.path.join(path, csv_files[0])
        print(f"Loading data from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def explore_data(self, df):
        """
        Explore the dataset to understand distributions and identify valid ranges
        
        Args:
            df: DataFrame with asteroid data
        """
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        # Display basic info
        print("\nDataset Info:")
        print(f"Total records: {len(df)}")
        print(f"\nColumns: {list(df.columns)}")
        
        # Check for magnitude-related columns
        magnitude_cols = [col for col in df.columns if 'magnitude' in col.lower() or col.lower() == 'h']
        print(f"\nMagnitude columns found: {magnitude_cols}")
        
        # Check for size/diameter columns
        size_cols = [col for col in df.columns if 'diameter' in col.lower() or 'size' in col.lower()]
        print(f"Size/diameter columns found: {size_cols}")
        
        # Analyze absolute magnitude if present
        if magnitude_cols:
            mag_col = magnitude_cols[0]
            print(f"\nAbsolute Magnitude ({mag_col}) Statistics:")
            print(df[mag_col].describe())
            print(f"Missing values: {df[mag_col].isna().sum()}")
            print(f"Valid range: [{df[mag_col].min():.2f}, {df[mag_col].max():.2f}]")
        
        # Analyze diameter if present
        if size_cols:
            for size_col in size_cols[:2]:  # Show first 2 size columns
                print(f"\n{size_col} Statistics:")
                print(df[size_col].describe())
                print(f"Missing values: {df[size_col].isna().sum()}")
        
        # Check for missing values overall
        print(f"\nMissing values per column:")
        missing = df.isnull().sum()
        print(missing[missing > 0])
        
        return df

    def handle_missing_and_outliers(self, df, magnitude_col, size_col):
        """
        Handle missing values and outliers in the dataset
        
        Args:
            df: DataFrame with asteroid data
            magnitude_col: Name of the magnitude column
            size_col: Name of the size column
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        initial_count = len(df)
        
        # Remove rows with missing values in key columns
        df_clean = df[[magnitude_col, size_col]].copy()
        df_clean = df_clean.dropna()
        
        print(f"Removed {initial_count - len(df_clean)} rows with missing values")
        
        # Remove outliers using IQR method
        for col in [magnitude_col, size_col]:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR for more lenient outlier removal
            upper_bound = Q3 + 3 * IQR
            
            before = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            removed = before - len(df_clean)
            
            if removed > 0:
                print(f"Removed {removed} outliers from {col}")
        
        print(f"\nFinal dataset size: {len(df_clean)} rows")
        print(f"Magnitude range: [{df_clean[magnitude_col].min():.2f}, {df_clean[magnitude_col].max():.2f}]")
        print(f"Size range: [{df_clean[size_col].min():.4f}, {df_clean[size_col].max():.4f}] km")
        
        return df_clean
    
    def preprocess_data(self, df_clean, magnitude_col, size_col, test_size=0.2, random_state=42):
        """
        Preprocess data: extract features, split, and normalize
        
        Args:
            df_clean: Cleaned DataFrame
            magnitude_col: Name of magnitude column
            size_col: Name of size column
            test_size: Proportion of data for testing (default 0.2 for 80/20 split)
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, normalization_params)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for preprocessing")
        
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Extract features (X) and target (y)
        X = df_clean[[magnitude_col]].values
        y = df_clean[[size_col]].values
        
        print(f"Feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTrain/Test Split ({int((1-test_size)*100)}/{int(test_size*100)}):")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Fit scalers on training data only
        self.scaler_X.fit(X_train)
        self.scaler_y.fit(y_train)
        
        # Transform both train and test data
        X_train_scaled = self.scaler_X.transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Calculate and store normalization parameters
        self.normalization_params = {
            'magnitude_min': float(X.min()),
            'magnitude_max': float(X.max()),
            'magnitude_mean': float(self.scaler_X.mean_[0]),
            'magnitude_std': float(self.scaler_X.scale_[0]),
            'size_min': float(y.min()),
            'size_max': float(y.max()),
            'size_mean': float(self.scaler_y.mean_[0]),
            'size_std': float(self.scaler_y.scale_[0])
        }
        
        print("\nNormalization Parameters:")
        print(f"Magnitude - Mean: {self.normalization_params['magnitude_mean']:.4f}, "
              f"Std: {self.normalization_params['magnitude_std']:.4f}")
        print(f"Size - Mean: {self.normalization_params['size_mean']:.4f}, "
              f"Std: {self.normalization_params['size_std']:.4f}")
        
        # Store for later use
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train_scaled
        self.y_test = y_test_scaled
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, self.normalization_params
    
    def train_model(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train a neural network regression model
        
        Args:
            X_train: Training features (normalized)
            y_train: Training targets (normalized)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Proportion of training data for validation
            
        Returns:
            keras.Model: Trained model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for model training")
        
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Build a simple neural network for regression
        model = keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # Output layer for regression
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("\nTraining complete!")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        self.model = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, output_file='model/metrics.json'):
        """
        Evaluate the trained model and save metrics
        
        Args:
            model: Trained Keras model
            X_test: Test features (normalized)
            y_test: Test targets (normalized)
            output_file: Path to save metrics JSON
            
        Returns:
            dict: Dictionary containing MAE, RMSE, and R² metrics
        """
        if not TF_AVAILABLE or not SKLEARN_AVAILABLE:
            raise ImportError("TensorFlow and scikit-learn are required for evaluation")
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Make predictions on test set
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Denormalize predictions and actual values for meaningful metrics
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_actual = self.scaler_y.inverse_transform(y_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'test_samples': int(len(X_test))
        }
        
        # Print metrics
        print("\nEvaluation Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f} km")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f} km")
        print(f"R² Score: {r2:.4f}")
        print(f"Test samples: {len(X_test)}")
        
        # Save metrics to JSON file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to: {output_file}")
        
        return metrics
    
    def export_model(self, model, output_dir='model'):
        """
        Export the trained model to TensorFlow.js format
        
        Args:
            model: Trained Keras model
            output_dir: Directory to save the model files
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for model export")
        
        print("\n" + "="*60)
        print("MODEL EXPORT")
        print("="*60)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model in Keras format first
        keras_model_path = os.path.join(output_dir, 'keras_model.h5')
        model.save(keras_model_path)
        print(f"Keras model saved to: {keras_model_path}")
        
        # Try to convert to TensorFlow.js format if available
        if TFJS_AVAILABLE:
            try:
                print("\nConverting to TensorFlow.js format...")
                tfjs.converters.save_keras_model(model, output_dir)
                print(f"TensorFlow.js model saved to: {output_dir}")
                print(f"  - model.json (model architecture)")
                print(f"  - group*.bin (model weights)")
            except Exception as e:
                print(f"\nWarning: Could not convert to TensorFlow.js format: {e}")
                print("You can convert manually later using:")
                print(f"  tensorflowjs_converter --input_format=keras {keras_model_path} {output_dir}")
        else:
            print("\nTensorFlow.js not available. Model saved in Keras format only.")
            print("To convert to TensorFlow.js format, install tensorflowjs and run:")
            print(f"  pip install tensorflowjs")
            print(f"  tensorflowjs_converter --input_format=keras {keras_model_path} {output_dir}")
        
    def save_normalization_params(self, output_file='model/normalization.json'):
        """
        Save normalization parameters to JSON file
        
        Args:
            output_file: Path to save normalization parameters
        """
        print("\n" + "="*60)
        print("SAVING NORMALIZATION PARAMETERS")
        print("="*60)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save parameters
        with open(output_file, 'w') as f:
            json.dump(self.normalization_params, f, indent=2)
        
        print(f"Normalization parameters saved to: {output_file}")
        print("\nParameters:")
        for key, value in self.normalization_params.items():
            print(f"  {key}: {value:.6f}")


def main():
    """Main training pipeline execution"""
    print("="*60)
    print("ASTEROID SIZE PREDICTOR - TRAINING PIPELINE")
    print("="*60)
    
    trainer = AsteroidModelTrainer()
    
    # Task 2.1: Load and explore data
    df = trainer.load_data()
    df = trainer.explore_data(df)
    
    # Use the correct column names from the Kaggle dataset
    # H = absolute magnitude
    # diameter = diameter in kilometers
    magnitude_col = 'H'
    size_col = 'diameter'
    
    print(f"\nUsing columns: magnitude='{magnitude_col}', size='{size_col}'")
    
    # Clean the data
    df_clean = trainer.handle_missing_and_outliers(df, magnitude_col, size_col)
    
    print("\n" + "="*60)
    print("Task 2.1 Complete: Data loading and exploration finished")
    print("="*60)
    
    # Task 2.2: Preprocess data
    X_train, X_test, y_train, y_test, norm_params = trainer.preprocess_data(
        df_clean, magnitude_col, size_col, test_size=0.2
    )
    
    print("\n" + "="*60)
    print("Task 2.2 Complete: Data preprocessing finished")
    print("="*60)
    
    # Task 2.4: Train model
    model = trainer.train_model(X_train, y_train, epochs=100, batch_size=32)
    
    print("\n" + "="*60)
    print("Task 2.4 Complete: Model training finished")
    print("="*60)
    
    # Task 2.5: Evaluate model
    metrics = trainer.evaluate_model(model, X_test, y_test)
    
    print("\n" + "="*60)
    print("Task 2.5 Complete: Model evaluation finished")
    print("="*60)
    
    # Task 2.7: Export model and save normalization parameters
    trainer.export_model(model, output_dir='model')
    trainer.save_normalization_params(output_file='model/normalization.json')
    
    print("\n" + "="*60)
    print("Task 2.7 Complete: Model export finished")
    print("="*60)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - model/model.json (TensorFlow.js model)")
    print("  - model/group*.bin (model weights)")
    print("  - model/normalization.json (normalization parameters)")
    print("  - model/metrics.json (evaluation metrics)")
    print("\nYou can now use these files in the web application!")


if __name__ == "__main__":
    main()
