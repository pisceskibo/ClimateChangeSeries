# ClimateChangeSeries
> Phân tích chuỗi thời gian trong dự báo thời tiết

## 1. TimeSeries Forecasting: 
Dự báo chuỗi thời gian là quá trình dự đoán các giá trị trong tương lai dựa trên dữ liệu lịch sử được sắp xếp theo thứ tự thời gian từ trước. Quá trình này sử dụng các quan sát trong quá khứ để đưa ra những dự đoán có cở sở về các điểm dữ liệu trong tương lai.

Mục tiêu dự đoán: 
+ Dự đoán xu hướng và biến động thời tiết theo thời gian thực tế.
+ Áp dụng đa dạng các phương pháp và mô hình dự báo chuỗi thời gian.

## 2. Autoregressive Integrated Moving Average (ARIMA):
Mô hình ARIMA là mô hình tự hồi quy tích hợp trung bình động, là phương pháp phổ biến để dự báo chuỗi thời gian có tính dừng. ARIMA kết hợp các thành phần tự hồi quy (autoregression), lấy sai phân (differencing), trung bình động (moving average) để mô hình hóa và dự báo các giá trị mới trong tương lai.

Ký hiệu: $ARIMA(p, d, q)$ với $p, d, q > 0$

Kỹ thuật xử lý của ARIMA:
+ Xác định các thành phần tự hồi quy (AR) và trung bình trượt (MA).
+ Áp dụng kỹ thuật lấy sai phân (differencing) để biến đổi dữ liệu thành chuỗi dừng.

### 2.1. Thành phần $I(d)$ - Integration (Tích hợp / Sai phân):
I (Integrated - Tích hợp / Sai phân - d): trừ giá trị hiện tại $y_t$ cho giá trị $y_{t - 1}$ ngay trước đó để loại bỏ xu hướng => đạt tính dừng (trung bình và phương sai không đổi) 
+ Sai phân bậc 1 (d = 1): $w_t = \Delta y_t = y_t - y_{t-1}$
+ Sai phân bậc d tổng quát: $w_t = \Delta^d y_t = (1 - B)^d y_t$ với $By_t = y_{t - 1}$

### 2.2. Thành phần $AR(p)$ - AutoRegressive (Tự hồi quy)
AR (Autoregressive - Tự hồi quy - p): dùng các giá trị quan sát trong quá khứ ($y_{t - 1}, y_{t - 2}, ...$) để dự báo giá trị hiện tại $y_t$

$$w_t = c + \phi_1 w_{t-1} + \phi_2 w_{t-2} + \dots + \phi_p w_{t-p} + \epsilon_t = c + \sum_{i=1}^{p} \phi_i w_{t-i} + \epsilon_t$$

trong đó: 
+ c là hằng số. 
+ $\phi_1, \phi_2, \dots, \phi_p$ là các hệ số hồi quy cần ước lượng.
+ $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ là nhiễu trắng tại thời điểm t.

### 2.3. Thành phần $MA(q)$ - Moving Average (Trung bình trượt):
MA (Moving Average - Trung bình trượt - q): dùng sai số quá khứ ($\epsilon_{t-1}, \epsilon_{t-2}, \dots$) để điều chỉnh giá trị dự báo $y_t$ hiện tại.

$$w_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q} = c + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

trong đó:
+ $\theta_1, \theta_2, \dots, \theta_q$ là các hệ số trung bình trượt cần ước lượng
+ $\epsilon_{t-j}$ là sai số dự báo tại thời điểm $t - j$

### 2.4. Phương trình tổng quát $ARIMA(p, d, q)$:
Phương trình $ARIMA(p, d, q)$ kết hợp cả 3 thành phần $I(d), AR(p), MA(q)$ với $w_t = \Delta^d y_t$ có dạng:

$$w_t = c + \sum_{i=1}^{p} \phi_i w_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

Dạng toán tử lùi thời gian (Backshift Operator $B$):

$$\left(1 - \sum_{i=1}^{p} \phi_i B^i\right) (1 - B)^d y_t = c + \left(1 + \sum_{j=1}^{q} \theta_j B^j\right) \epsilon_t$$

Quy trình thực hiện:
1. Kiểm định ADF và xử lý tính dừng (Identifying $d$).
2. Xác định bậc $p$ và $q$ (Model Identification) thông qua hàm đồ thị ACF và PACF.
3. Ước lượng tham số (MLE/OLS) và đánh giá chỉ AIC/BIC càng nhỏ càng tốt với $\hat{L}$ là Maximum Likelihood, k = p + q + 1 (nếu chứa hằng số).

$$\text{AIC} = 2k - 2\ln(\hat{L}), \text{BIC} = k\ln(n) - 2\ln(\hat{L})$$

4. Chẩn đoán sai số của mô hình bằng cách kiểm tra phần dư $\epsilon_t = y_t - \hat{y}_t$ sao cho phần dư nhiễu trắng và có phân bố chuẩn.

## 3. Seasonal Autoregressive Integrated Moving-Average (SARIMA):
Seasonal Autoregressive Integrated Moving-Average (SARIMA) là mô hình tự hồi quy tích hợp trung bình trượt mùa vụ, là phiên bản mở rộng của ARIMA có tính đến thành phần có tính chất chu kỳ mùa vụ (seasonal components) trong bộ dữ liệu.

Ký hiệu: $SARIMA(p, d, q) \times (P, D, Q)_m$

### 3.1. Thành phần phi mùa vụ $(p, d, q)$:
Giống với mô hình chuỗi thời gian ARIMA phía trên.

### 3.2. Thành phần mùa vụ $(P, D, Q)_m$:
+ $P$ (Seasonal AR): là bậc tự hồi quy mùa vụ, mối quan hệ giữa thời điểm hiện tại và cùng thời điểm đó ở các chu kỳ trước ($y_t$ phụ thuộc vào $y_{t - m}, y_{t - 2m}, ...$).
+ $D$ (Seasonal Difference): bậc lấy sai phân mùa vụ ($\Delta_m y_t = y_t - y_{t-m}$) để triệt tiêu tính mùa vụ lặp đi lặp lại.
+ $Q$ (Seasonal MA): Bậc trung bình động mùa vụ. Dùng sai số dự báo từ các chu kỳ trước ($\epsilon_{t-m}, \epsilon_{t-2m}, ...$) để điều chỉnh.
+ $m$ (Seasonal Period): Độ dài của một chu kỳ mùa vụ (số bước thời gian để hình thành 1 chu kỳ lặp lại).
    + $m = 7$: Dữ liệu theo ngày, tính chu kỳ theo tuần (Thứ Hai tuần này so với Thứ Hai tuần trước). => ACTIVE 
    + $m = 12$: Dữ liệu theo tháng, tính chu kỳ theo năm (Tháng 1 năm nay so với Tháng 1 năm ngoái).

### 3.3. Phương trình tổng quát $SARIMA(p, d, q) \times (P, D, Q)_m$:
Sử dụng Toán tử lùi thời gian (Backshift Operator $B$), trong đó $B y_t = y_{t-1}$ và $B^m y_t = y_{t-m}$:

$$\Phi_P(B^m) \, \phi_p(B) \, (1 - B)^d \, (1 - B^m)^D \, y_t = c + \Theta_Q(B^m) \, \theta_q(B) \, \epsilon_t$$

trong đó:
+ $\phi_p(B) = 1 - \phi_1 B - \phi_2 B^2 - \dots - \phi_p B^p$: Đa thức AR phi mùa vụ.
+ $\Phi_P(B^m) = 1 - \Phi_1 B^m - \Phi_2 B^{2m} - \dots - \Phi_P B^{Pm}$: Đa thức AR mùa vụ.
+ $(1 - B)^d$: Sai phân phi mùa vụ bậc $d$.
+ $(1 - B^m)^D$: Sai phân mùa vụ bậc $D$.
+ $\theta_q(B) = 1 + \theta_1 B + \theta_2 B^2 + \dots + \theta_q B^q$: Đa thức MA phi mùa vụ.
+ $\Theta_Q(B^m) = 1 + \Theta_1 B^m + \Theta_2 B^{2m} + \dots + \Theta_Q B^{Qm}$: Đa thức MA mùa vụ.
+ $c$: Hằng số (Intercept / Drift).
+ $\epsilon_t$: Nhiễu trắng (White noise) có phân bố chuẩn $\mathcal{N}(0, \sigma^2)$.

Quy trình thực hiện:
1. Kiểm định ADF và xử lý tính dừng (Identifying $d$) theo chu kỳ.
2. Xác định bậc $p$ và $q$ (Model Identification) thông qua hàm đồ thị ACF và PACF.
3. Ước lượng tham số (MLE/OLS) và đánh giá chỉ AIC/BIC càng nhỏ càng tốt với $\hat{L}$ là Maximum Likelihood, k = p + q + P + Q + 1 (nếu chứa hằng số).

$$\text{AIC} = 2k - 2\ln(\hat{L}), \text{BIC} = k\ln(n) - 2\ln(\hat{L})$$

4. Chẩn đoán sai số của mô hình bằng cách kiểm tra phần dư $\epsilon_t = y_t - \hat{y}_t$ sao cho phần dư nhiễu trắng và có phân bố chuẩn.

## 4. Một số mô hình chuỗi thời gian khác (tham khảo):
+ Exponential Smoothing (ETS) là phương pháp san phẳng mũ bao gồm các mô hình như San phẳng mũ đơn (Simple Exponential Smoothing), San phẳng mũ kép (Double Exponential Smoothing) và San phẳng mũ ba (Triple Exponential Smoothing - hay phương pháp Holt-Winters).
+ Mô hình Prophet là mô hình dự báo được phát triển bởi Facebook, được thiết kế để xử lý dữ liệu chuỗi thời gian có tính mùa vụ (seasonality) và ảnh hưởng của các ngày lễ (holiday effects).

## 5. Quy trình phân tích chuỗi thời gian dự báo thời tiết:
1. Tiền xử lý dữ liệu (Data Preprocessing): làm sạch và biến đổi dữ liệu để xử lý các giá trị bị khuyết, giá trị ngoại lai và các điểm bất thường. Trong đó, dữ liệu có thể được phân rã (decomposition) để tách các thành phần chính có xu hướng theo mùa vụ.
2. Lựa chọn mô hình (Model Selection): chọn mô hình phù hợp dựa trên đặc tính của dữ liệu chuỗi thời gian.
3. Huấn luyện mô hình (Model Training): sử dụng dữ liệu quá khứ để ước lượng các tham số và tối ưu hóa hiệu suất mô hình.
4. Đánh giá mô hình (Model Evaluation): đánh giá độ chính xác của các mô hình dự báo và chỉ số đánh giá phù hợp.
5. Dự báo (Forecasting): đưa ra dự đoán cho các mốc thời gian trong tương lai bằng mô hình đã huấn luyện.


| STT | Mô hình                             | Có code thực hành |
| --- | ----------------------------------- | ----------------- |
| 1   | **ARIMA, SARIMA**                           | ✅                 |
| 2   | **Linear Regression**               | ✅                 |
| 3   | **SVR – Support Vector Regression** | ✅                 |
| 4   | **Random Forest Regression**        | ✅                 |
| 5   | **KNN Regression**                  | ✅                 |
| 6   | **Decision Tree Regression**        | ✅                 |

> Tài liệu tham khảo: https://www.kaggle.com/code/aniketkadam702030/daily-climate-change-time-series/notebook