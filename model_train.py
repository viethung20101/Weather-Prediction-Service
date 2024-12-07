from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self._setup_logging()
        self._setup_gpu()
        
    def _setup_logging(self):
        """Cấu hình logging"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _setup_gpu(self):
        """Cấu hình GPU và các optimization"""
        self.logger.info("Setting up GPU...")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                self.logger.error(f"GPU setup error: {e}")
        else:
            self.logger.warning("No GPU found, using CPU")

        # Optimization settings
        tf.config.optimizer.set_jit(True)
        tf.config.threading.set_intra_op_parallelism_threads(12)
        tf.config.threading.set_inter_op_parallelism_threads(12)

    def create_model(self, sequence_length, features=1):
        """
        Tạo model architecture
        
        Args:
            sequence_length (int): Độ dài chuỗi đầu vào
            features (int): Số features đầu vào
        """
        self.logger.info(f"Creating model with sequence length {sequence_length}")
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            model = Sequential([
                LSTM(50, activation='relu', return_sequences=True, 
                     input_shape=(sequence_length, features)),
                Dropout(0.3),
                LSTM(50, activation='relu', return_sequences=True),
                Dropout(0.3),
                LSTM(50, activation='relu'),
                Dropout(0.3),
                Dense(1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_squared_error']
            )
        return model

    def train_model(self, train_data, test_data, sequence_length=30, 
                   batch_size=32, epochs=100):
        """
        Train model với dữ liệu đã được xử lý
        
        Args:
            train_data: Dữ liệu training đã được scale
            test_data: Dữ liệu test đã được scale
            sequence_length: Độ dài chuỗi đầu vào
            batch_size: Kích thước batch
            epochs: Số epochs training
        """
        self.logger.info("Preparing training data...")
        train_generator = TimeseriesGenerator(
            train_data, train_data, 
            length=sequence_length, 
            batch_size=batch_size
        )

        test_generator = TimeseriesGenerator(
            test_data, test_data,
            length=sequence_length,
            batch_size=batch_size
        )

        model = self.create_model(sequence_length)
        
        self.logger.info("Starting model training...")
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            verbose=1
        )
        
        self.logger.info("Training completed")
        return model, history

    def predict_sequence(self, model, sequence_data, scaler):
        """
        Dự đoán với một chuỗi dữ liệu
        
        Args:
            model: Model đã train
            sequence_data: Dữ liệu đầu vào đã được xử lý
            scaler: Scaler đã fit với dữ liệu
        """
        try:
            prediction = model.predict(sequence_data, verbose=0)
            return scaler.inverse_transform(prediction)
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise

    def evaluate_model(self, model, test_generator, scaler):
        """
        Đánh giá model
        
        Args:
            model: Model đã train
            test_generator: Generator cho dữ liệu test
            scaler: Scaler đã fit với dữ liệu
        """
        try:
            predictions = model.predict(test_generator)
            predictions = scaler.inverse_transform(predictions)
            actuals = scaler.inverse_transform(test_generator.targets)
            
            mse = np.mean((predictions - actuals) ** 2)
            mae = np.mean(np.abs(predictions - actuals))
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse))
            }
        except Exception as e:
            self.logger.error(f"Evaluation error: {e}")
            raise
