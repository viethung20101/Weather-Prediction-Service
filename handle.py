import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from model_train import ModelTrainer
import tensorflow as tf

class Handle:
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.scaler = MinMaxScaler()
        self.models = {}  # Dictionary to store models for each province
        self.sequence_length = 30
        self.batch_size = 32
        self.epochs = 100
        
        # Tạo thư mục models nếu chưa tồn tại
        os.makedirs('models', exist_ok=True)
        
        # Train các model ban đầu nếu chưa có
        self._initialize_models()
    
    def _initialize_models(self):
        """Khởi tạo và train các model lần đầu nếu chưa có"""
        try:
            df = pd.read_csv('data/weather.csv')
            provinces = df['province'].unique()
            
            for province in provinces:
                model_path = f'models/weather_model_{province}.h5'
                if not os.path.exists(model_path):
                    print(f"Training initial model for {province}...")
                    self.train_province_model(province)
                else:
                    # Load model đã có
                    model = tf.keras.models.load_model(model_path)
                    self.models[province] = {
                        'model': model,
                        'scaler': self.scaler,
                        'last_trained': datetime.now()
                    }
            print("Model initialization completed")
        except FileNotFoundError:
            print("No weather.csv file found. Models will be trained when data is available.")
        except Exception as e:
            print(f"Error during model initialization: {e}")

    def prepare_data(self, df, province):
        """Xử lý dữ liệu cho từng tỉnh"""
        province_data = df[df['province'] == province]['rain'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(province_data)
        
        # Chia train/test
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        return train_data, test_data
        
    def train_province_model(self, province):
        """Train model cho từng tỉnh"""
        df = pd.read_csv('data/weather.csv')
        train_data, test_data = self.prepare_data(df, province)
        
        model, history = self.model_trainer.train_model(
            train_data, 
            test_data,
            self.sequence_length,
            self.batch_size,
            self.epochs
        )
        
        self.models[province] = {
            'model': model,
            'scaler': self.scaler,
            'last_trained': datetime.now()
        }
        
        # Lưu model
        model.save(f'models/weather_model_{province}.h5')
        
    def predict_rainfall(self, province, date):
        """Dự đoán lượng mưa cho tỉnh và ngày cụ thể"""
        if province not in self.models:
            self.train_province_model(province)
            
        model_info = self.models[province]
        # Lấy dữ liệu sequence gần nhất để predict
        df = pd.read_csv('data/weather.csv')
        recent_data = df[df['province'] == province].tail(self.sequence_length)['rain'].values
        scaled_data = model_info['scaler'].transform(recent_data.reshape(-1, 1))
        sequence = scaled_data.reshape(1, self.sequence_length, 1)
        
        prediction = self.model_trainer.predict_sequence(
            model_info['model'],
            sequence,
            model_info['scaler']
        )
        
        return float(prediction[0][0])
        
    def retrain_models(self):
        """Hàm để train lại model định kỳ"""
        for province in self.models.keys():
            self.train_province_model(province)
