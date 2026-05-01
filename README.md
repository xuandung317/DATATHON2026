# [DATATHON2026] - NHÓM INSIGHTX4

## 1. Xác định bài toán
* Lĩnh vực: Tài chính kinh doanh.
* Loại bài toán chính: Dự đoán doanh thu.
* Input: Dữ liệu đầu vào bao gồm dữ liệu về sản phẩm, khách hàng, đơn hàng,...
* Output: Dữ liệu đầu ra - `target` là `revenue` và `cogs`.

## 2. Giới thiệu Dataset
Bộ dữ liệu mô phỏng hoạt động của một doanh nghiệp thời trang thương mại điện tử tại Việt Nam trong giai đoạn từ 04/07/2012 đến 31/12/2022. Dữ liệu bao gồm 15 file CSV, được chia thành 4 lớp: Master (dữ liệu tham chiếu), Transaction (giao dịch), Analytical (phân tích) và Operational (vận hành).

## 3. Cấu trúc 
```
project/
├── data/                           (Dữ liệu đầu vào)                
|
├── outputs/
|   └── submission.csv              (File csv dự đoán) 
|
├── notebook/
|   ├── EDA.ipynb                   (Notebook EDA)
│   └── main.ipynb                  (Notebook chính)
|
├── src/
│   ├── config.py                   (Cấu hình)
│   ├── fe.py                       (Features engineering)
│   └── train.py                    (Pipeline mô hình)
|
├── main.py                         (python script)                          
├── README.md                 
└── requirements.txt              
```                  
## 4. Hướng dẫn cài đặt
```python
python --version
python -m venv venv
pip install -r requirements.txt
```
## 5. Hướng dẫn chạy
_**Chạy file main.py**_

Chạy lệnh sau trên **terminal**:
```
python main.py
```
_**Chạy file EDA.ipynb để khám phá dữ liệu và file main.ipynb để xem quá trình huấn luyện mô hình**_