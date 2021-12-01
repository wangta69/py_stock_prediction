import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA # pip install statsmodels==0.12.2
# from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# step1_setting_pyplot.py Start
# 한글폰트
import platform

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
elif platform.system() == 'Linux':
    path = "/usr/share/fonts/NanumFont/NanumGothicBold.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    plt.rc('font', family=font_name)
else:
    pass
# step1_setting_pyplot.py End


#step2_주가가져오기.py start
eh_df_1 = yf.download('EH',
                      start='2019-01-01',
                      end='2021-04-22',
                      progress=False)
#step2_주가가져오기.py end

eh_df_1 = eh_df_1.reset_index()
eh_df_1.columns = ['day', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

eh_df_1['day'] = pd.to_datetime(eh_df_1['day'])
eh_df_1.index = eh_df_1['day']
eh_df_1.set_index('day', inplace=True)

eh_df_vol = eh_df_1[["Volume"]]
eh_df_1 = eh_df_1[["Open", "High", "Low", "Close", "Adj Close"]]

# x일 행에 x+1일의 종가를 추가
eh_df_1['tomorrow Adj Close'] = eh_df_1['Adj Close'].shift(-1)

# 변동율 측정 => ((다음날종가)-(오늘종가))/오늘종가
eh_df_1['Fluctuation'] = eh_df_1['tomorrow Adj Close']-eh_df_1['Adj Close']
eh_df_1['Fluctuation Rate'] = eh_df_1['Fluctuation']/eh_df_1['Adj Close']
eh_df_1['price'] = eh_df_1['Adj Close']

# maN : N일전부터 현재까지의 수정종가 평균
ma5 = eh_df_1['Adj Close'].rolling(window=5).mean()
ma20 = eh_df_1['Adj Close'].rolling(window=20).mean()
ma60 = eh_df_1['Adj Close'].rolling(window=60).mean()
ma120 = eh_df_1['Adj Close'].rolling(window=120).mean()

#이동평균선 추가
eh_df_1.insert(len(eh_df_1.columns), "MA5", ma5)
eh_df_1.insert(len(eh_df_1.columns), "MA20", ma20)
eh_df_1.insert(len(eh_df_1.columns), "MA60", ma60)
eh_df_1.insert(len(eh_df_1.columns), "MA120", ma120)

# 거래량 추가
vma5 = eh_df_vol['Volume'].rolling(window=5).mean()
eh_df_vol.insert(len(eh_df_vol.columns), "VMA5", vma5)

# 변동율 추가
# x일 행에 x+1일의 종가를 추가
eh_df_1['tomorrow Adj Close'] = eh_df_1['Adj Close'].shift(-1)

# 변동율 측정 => ((다음날종가)-(오늘종가))/오늘종가
eh_df_1['Fluctuation'] = eh_df_1['tomorrow Adj Close']-eh_df_1['Adj Close']
eh_df_1['Fluctuation Rate'] = eh_df_1['Fluctuation']/eh_df_1['Adj Close']
eh_df_1['price'] = eh_df_1['Adj Close']


# # 추가로 먼저 Train/Test set으로 나눌것인데 마지막 날짜 5일을 제외한 나머지를 Train으로 지정 해주고 마지막 날짜 5일을 Test으로 지정 해주었다.
eh_train_df = eh_df_1[:335]
# print(eh_train_df['price'])
# print('eh_train_df End =============')
#
eh_test_df = eh_df_1[335:]
# eh_test_df
#  https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMAResults.html
# model = ARIMA(eh_train_df.price.values, order=(2, 1, 2))
# model_fit = model.fit(trend='c', full_output=True, disp=True)
# print(model_fit.summary())
model = ARIMA(eh_train_df.price.values, order=(2, 1, 2))
# model_fit = model.fit()
model_fit = sm.tsa.ARMA(eh_train_df.price.values, (3, 0)).fit()
print(model_fit.summary())

# fig = model_fit.plot_predict()
# plt.show()

# 잔차의 변동을 시각화
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# plt.title('ARIMA 모델의 잔차 그래프')
# plt.show()

# 이항주가의 변동률과, ARIMA 모델의 잔차 그래프
# plt.figure(figsize=(12,8))
# plt.plot(eh_df_1['Fluctuation Rate'], color = 'lightblue')
# plt.axhline(y=0, color='red', ls='--') #기준선추가 , axhline() 함수의 첫번째 인자는 y 값으로서 수평선의 위치가 됩니다.
# plt.title('이항 주가 변동률')
# plt.show()

forecast_data = model_fit.forecast(steps=5) # 학습 데이터넷으로부터 5일 뒤를 예측한다.
# print(eh_test_df)
# print(forecast_data)

# 마지막 5일의 예측 데이터 (2021-04-15 ~ 2021-04-21)
pred_arima_y = forecast_data[0].tolist()

# 실제 5일의 데이터 (2021-04-15 ~ 2021-04-21)
test_y = eh_test_df.price.values

# 마지막 5일의 예측 데이터 최소값
pred_y_lower = []
# 마지막 5일의 예측 데이터 최대값
pred_y_upper = []

for lower_upper in forecast_data[2]:
    lower = lower_upper[0]
    upper = lower_upper[1]
    pred_y_lower.append(lower)
    pred_y_upper.append(upper)


print('pred_arima_y', pred_arima_y)
print('pred_y_lower', pred_y_lower)
print('pred_y_upper', pred_y_upper)
print('test_y', test_y)
# 시각화
# 모델이 예측한 가격 그래프
plt.clf()
plt.plot(pred_arima_y, color='gold')
# 모델이 예측한 최저 가격 그래프
plt.plot(pred_y_lower, color='red')
# 모델이 예측한 최고 가격 그래프
plt.plot(pred_y_upper, color='blue')
# 실제 가격 그래프
plt.plot(test_y, color='green')
plt.legend(['예측 가격', '예측 최저 가격', '예측 최고 가격', '실제 가격'])
plt.show()


# 상한가와 하한가를 제외한 그래프
plt.clf()
plt.plot(pred_arima_y, color="gold") # 모델이 예상한 가격 그래프
plt.plot(test_y, color = "green") # 실제 가격 그래프
plt.legend(['모델이 예상한 가격 그래프', '실제 가격 그래프'])
plt.show()
    




