from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import pandas as pd
import schedule
import time
import threading
from handle import Handle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="API Dự Báo Lượng Mưa",
    description="API dự đoán lượng mưa cho các tỉnh thành Việt Nam sử dụng mô hình LSTM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WeatherData(BaseModel):
    """
    Schema cho một bản ghi dữ liệu thời tiết
    
    Thuộc tính:
        province (str): Tên tỉnh/thành phố
        date (str): Ngày theo định dạng YYYY-MM-DD
        rain (float): Lượng mưa (đơn vị: mm)
    """
    province: str
    date: str
    rain: float

class WeatherDataBatch(BaseModel):
    """
    Schema cho việc cập nhật dữ liệu thời tiết hàng loạt
    
    Thuộc tính:
        data (List[WeatherData]): Danh sách các bản ghi thời tiết
    """
    data: List[WeatherData]

@app.on_event("startup")
async def startup_event():
    """
    Khởi tạo Handle và train các model khi ứng dụng khởi động
    Hàm này sẽ chạy khi ứng dụng FastAPI bắt đầu
    """
    global handle
    print("Initializing Handle and training models...")
    handle = Handle()
    print("Initialization completed")

@app.post("/predict", 
    response_model=dict,
    tags=["Dự Báo"],
    summary="Dự đoán lượng mưa cho một tỉnh và ngày cụ thể",
    response_description="Kết quả dự đoán lượng mưa"
)
async def predict_rainfall(province: str, date: str):
    """
    Dự đoán lượng mưa cho một tỉnh và ngày cụ thể.

    Tham số:
    - province (str): Tên tỉnh/thành phố
    - date (str): Ngày cần dự báo (định dạng YYYY-MM-DD)

    Trả về:
    - dict: Chứa tỉnh, ngày và lượng mưa dự đoán
        {
            "province": str,
            "date": str,
            "predicted_rainfall": float
        }

    Lỗi có thể xảy ra:
    - HTTPException(400): Lỗi validation đầu vào
    - HTTPException(404): Không tìm thấy tỉnh
    - HTTPException(500): Lỗi khi dự đoán
    
    Ví dụ:    ```
    POST /predict?province=HaNoi&date=2024-03-25    ```
    """
    try:
        prediction_date = datetime.strptime(date, "%Y-%m-%d")
        rainfall = handle.predict_rainfall(province, prediction_date)
        return {
            "province": province,
            "date": date,
            "predicted_rainfall": rainfall
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain", 
    response_model=dict,
    tags=["Cập Nhật Dữ Liệu"],
    summary="Force retrain model",
    response_description="Thông báo kết quả retrain model"
)
async def force_retrain():
    """
    Bắt buộc train lại tất cả các model cho các tỉnh.

    Endpoint này sẽ kích hoạt việc training lại ngay lập tức cho tất cả các model,
    không phụ thuộc vào lịch training định kỳ.

    Trả về:
    - dict: Thông báo kết quả training
        {
            "message": "Đã train lại các model thành công"
        }

    Lỗi có thể xảy ra:
    - HTTPException(500): Lỗi khi training lại model
    """
    try:
        handle.retrain_models()
        return {"message": "Models retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-weather-data", 
    response_model=dict,
    tags=["Cập Nhật Dữ Liệu"],
    summary="Cập nhật dữ liệu thời tiết mới",
    response_description="Thông báo cập nhật dữ liệu thời tiết"
)
async def update_weather_data(weather_batch: WeatherDataBatch):
    """
    Cập nhật dữ liệu thời tiết và train lại model cho các tỉnh bị ảnh hưởng.

    Tham số:
    - weather_batch (WeatherDataBatch): Batch dữ liệu thời tiết cần cập nhật

    Body Request:    ```json
    {
        "data": [
            {
                "province": "HaNoi",
                "date": "2024-03-20",
                "rainfall": 25.5
            },
            {
                "province": "HoChiMinh",
                "date": "2024-03-20",
                "rainfall": 30.2
            }
        ]
    }    ```

    Trả về:
    - dict: Thông báo kết quả cập nhật
        {
            "message": "Đã cập nhật dữ liệu cho X tỉnh"
        }

    Lỗi có thể xảy ra:
    - HTTPException(400): Lỗi validation dữ liệu
    - HTTPException(500): Lỗi khi cập nhật hoặc training
    """
    try:
        # Chuyển đổi dữ liệu mới thành DataFrame
        new_data = []
        for item in weather_batch.data:
            new_data.append({
                'province': item.province,
                'date': item.date,
                'rainfall': item.rainfall
            })
        new_df = pd.DataFrame(new_data)
        
        # Đọc file hiện tại
        try:
            existing_df = pd.read_csv('data/weather.csv')
            # Gộp dữ liệu mới với dữ liệu cũ
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except FileNotFoundError:
            combined_df = new_df
            
        # Loại bỏ các bản ghi trùng lặp (nếu có)
        combined_df = combined_df.drop_duplicates(subset=['province', 'date'])
        
        # Sắp xếp theo tỉnh và ngày
        combined_df = combined_df.sort_values(['province', 'date'])
        
        # Lưu lại file
        combined_df.to_csv('data/weather.csv', index=False)
        
        # Train lại model cho các tỉnh có dữ liệu mới
        affected_provinces = set(new_df['province'].unique())
        for province in affected_provinces:
            handle.train_province_model(province)
            
        return {"message": f"Updated data for {len(affected_provinces)} provinces"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
