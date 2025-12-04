# Credit Scoring Web App

Ứng dụng này giúp phân loại rủi ro tín dụng của khách hàng dựa trên các thông tin đầu vào, sử dụng mô hình XGBoost đã huấn luyện.

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường

- Cài Python 3.8 trở lên.
- (Khuyến nghị) Tạo môi trường ảo:
    ```
    python -m venv venv
    venv\Scripts\activate
    ```
- Cài đặt các thư viện cần thiết:
    ```
    pip install -r requirements.txt
    ```

### 2. Chạy ứng dụng

- Chạy lệnh sau trong thư mục chứa `app.py`:
    ```
    streamlit run app.py
    ```
- Truy cập địa chỉ hiển thị trên terminal (thường là http://localhost:8501).

### 3. Nhập thông tin khách hàng

- Nhập các trường thông tin ở sidebar bên trái.
- Nhấn nút **Dự đoán** để xem kết quả phân loại rủi ro tín dụng.

---

**Lưu ý:**  
- Đảm bảo file `pipeline.pkl` (mô hình đã huấn luyện) nằm cùng thư mục với `app.py`.
- Nếu gặp lỗi thiếu thư viện, kiểm tra lại các bước cài đặt.

---

Chúc bạn sử dụng hiệu quả!