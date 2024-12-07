# Weather Rainfall Prediction API

API dự đoán lượng mưa cho các tỉnh thành Việt Nam sử dụng Deep Learning (LSTM).

## Tính năng

- Dự đoán lượng mưa cho từng tỉnh thành
- Tự động train model cho các tỉnh mới
- Cập nhật dữ liệu thời tiết và train lại model
- API endpoints cho prediction và data management
- Tự động train lại model định kỳ

## Yêu cầu hệ thống

- Python 3.12.7
- Docker (tùy chọn)

## Cài đặt

### Sử dụng Docker

1; Build Docker image:

```bash
docker build -t weather-api .
```

2; Run Docker container:

```bash
docker run -d \
-p 8181:8181 \
-v $(pwd)/data:/app/data \
-v $(pwd)/models:/app/models \
--name weather-prediction \
weather-prediction-api
```

### Cài đặt thủ công

1; Tạo virtual environment:

```bash
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```

2; Cài đặt các dependencies:

```bash
pip install -r requirements.txt
```

3; Chạy API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8181
```

## Cấu trúc dự án

weather-prediction-service/
├── main.py # FastAPI application
├── handle.py # Model handling logic
├── model_train.py # LSTM model training
├── requirements.txt # Python dependencies
├── Dockerfile
├── data/ # Data directory
│ └── weather.csv # Training data
└── models/ # Trained models directory

## API Endpoints

### 1. Dự đoán lượng mưa

- **URL:**

  `POST /predict`

- **Request Body:**

  ```json
  {
    "province": "Ha Noi",
    "date": "2024-01-01"
  }
  ```

- **Response:**

  ```json
  {
    "prediction": 10.5
  }
  ```

### 2. Cập nhật dữ liệu thời tiết

- **URL:**

  `POST /update-weather-data`

- **Request Body:**

  ```json
  {
    "province": "Ha Noi",
    "date": "2024-01-01",
    "rain": 25.5
  }
  ```

- **Response:**

  ```json
  {
    "message": "Data updated successfully"
  }
  ```

### 3. Force retrain model

- **URL:**

  `POST /retrain-model`

- **Response:**

  ```json
  {
    "message": "Model retrained successfully"
  }
  ```

## Tự đông hoá

- Models được tự động train khi khởi động nếu chưa tồn tại
- Retrain tự động sau mỗi 7 ngày
- Retrain tự động khi có dữ liệu mới

## Monitoring

Kiểm tra logs container:

```bash
docker logs -f weather-prediction
```

## API Documentation

Sau khi chạy ứng dụng, truy cập:

- Swagger UI: http://localhost:8181/docs
- ReDoc: http://localhost:8181/redoc

## Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## License

[MIT License](LICENSE)