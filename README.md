# ClimateChangeSeries
> Phân tích chuỗi thời gian trong dự báo thời tiết

## 1. TimeSeries Forecasting: 
Dự báo chuỗi thời gian là quá trình dự đoán các giá trị trong tương lai dựa trên dữ liệu lịch sử được sắp xếp theo thứ tự thời gian từ trước. Quá trình này sử dụng các quan sát trong quá khứ để đưa ra những dự đoán có cở sở về các điểm dữ liệu trong tương lai.

Mục tiêu dự đoán: 
+ Dự đoán xu hướng và biến động thời tiết theo thời gian thực tế.
+ Áp dụng đa dạng các phương pháp và mô hình dự báo chuỗi thời gian.

## 2. Autoregressive Integrated Moving Average (ARIMA):
Mô hình ARIMA là mô hình tự hồi quy tích hợp trung bình trượt, là phương pháp phổ biến để dự báo chuỗi thời gian có tính dừng.

Ký hiệu: $ARIMA(p, d, q)$ với $p, d, q > 0$

Cơ sở lý thuyết:
+ AR (Autoregressive - Tự hồi quy - p): dùng các giá trị quan sát trong quá khứ ($y_{t - 1}, y_{t - 2}, ...$) để dự báo giá trị hiện tại $y_t$
+ I (Integrated - Tích hợp / Sai phân - d): trừ giá trị hiện tại $y_t$ cho giá trị $y_{t - 1}$ ngay trước đó để loại bỏ xu hướng => đạt tính dừng (trung bình và phương sai không đổi) 

$$\Delta y_t = y_t - y_{t-1}$$

+ MA (Moving Average - Trung bình trượt - q): dùng sai số quá khứ ($\epsilon_{t-1}, \epsilon_{t-2}, \dots$) để điều chỉnh giá trị dự báo $y_t$ hiện tại

Kỹ thuật xử lý của ARIMA:
+ Xác định các thành phần tự hồi quy (AR) và trung bình trượt (MA).
+ Áp dụng kỹ thuật lấy sai phân (differencing) để biến đổi dữ liệu thành chuỗi dừng.

## 3. Seasonal Autoregressive Integrated Moving-Average (SARIMA):
SARIMA là mô hình tự hồi quy tích hợp trung bình trượt mùa vụ, là phiên bản mở rộng của ARIMA có tính đến thành phần có tính chất chu kỳ mùa vụ (seasonal components) trong bộ dữ liệu.