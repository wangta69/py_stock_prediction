import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import yfinance as yf
import pandas as pd
from fbprophet import Prophet  # https://daewonyoon.tistory.com/266

# step1_setting_pyplot.py Start
# 한글폰트
import platform


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
eh_df = yf.download('EH',
                      start='2019-01-01',
                      end='2021-04-22',
                      progress=False)
eh_df = eh_df[["Close"]]
#step2_주가가져오기.py end

# eh_df = eh_df[["Close"]]
eh_df = eh_df.reset_index()
eh_df.columns = ['ds', 'y']
eh_train_df = eh_df[:335]
eh_test_df = eh_df[335:]
# print(eh_train_df, eh_test_df)
# seasonality_mode : 연간,월간,주간,일간 등의 트렌드성을 반영하는 것의 의미하는 파라미터이다.
# changepoint_prior_scale: 트렌드가 변경되는 문맥을 반영하는 파라미터이다. 수치가 높을수록 모델은 과적합에 가까워 진다.
prophet = Prophet(seasonality_mode='multiplicative',
                 yearly_seasonality=True,
                 weekly_seasonality=True,
                 daily_seasonality=True,
                 changepoint_prior_scale=0.5)

prophet.fit(eh_train_df)
print(eh_train_df)
"""
            ds          y
0   2019-12-13  12.900000
1   2019-12-16  12.500000
2   2019-12-17  11.070000
3   2019-12-18   9.520000
4   2019-12-19   9.400000
..         ...        ...
330 2021-04-08  34.070000
331 2021-04-09  34.340000
332 2021-04-12  29.200001
333 2021-04-13  29.020000
334 2021-04-14  29.400000

[335 rows x 2 columns]
"""
future_data = prophet.make_future_dataframe(periods=5, freq='d')
forecast_data = prophet.predict(future_data)
forecast_data = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
# forecast_data = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# print(forecast_data)
# print(forecast_data.shape)
# print(forecast_data.info())
# print(forecast_data.columns)


# print('ds:', forecast_data['ds'])
# print('yhat:', forecast_data['yhat'])
# print('yhat_lower:', forecast_data['yhat_lower'])
# print('yhat_upper:', forecast_data['yhat_upper'])
# print('-- forecast_data --')
# print(forecast_data)
"""
            ds       yhat  yhat_lower  yhat_upper
335 2021-04-15  32.967509   26.808574   38.187686
336 2021-04-16  33.068849   27.482798   39.058786
337 2021-04-17  31.120162   25.208730   37.383985
338 2021-04-18  31.493229   25.489314   36.984746
339 2021-04-19  33.519117   27.750270   39.321226
"""
# fig1 = prophet.plot(forecast_data)

# plt.clf()
# not work
# fig2 = prophet.plot_components(forecast_data) # KeyError: 'trend
# plt.show()

# Testset 평가 이번에는 테스트셋을 평가 해보자. 다음 코드의 실행 결과를 보자
# plt.clf()
# plt.figure(figsize=(15, 10))
#
# # 마지막 5일의 예측 데이터 (2021-04-15 ~ 2021-04-19)
# pred_fbprophet_y = forecast_data.yhat.values[-5:]
#
# # 실제 5일의 데이터 (2021-04-15 ~ 2021-04-19)
# test_y = eh_test_df.y.values
#
# # 마지막 5일의 예측 데이터 최소값
# pred_y_lower = forecast_data.yhat_lower.values[-5:]
# # 마지막 5일의 예측 데이터 최대값
# pred_y_upper = forecast_data.yhat_upper.values[-5:]
#
# # 모델이 예측한 가격 그래프
# plt.plot(pred_fbprophet_y, color = 'gold')
#
# # 모델이 예측한 최저 가격 그래프
# plt.plot(pred_y_lower, color = 'red')
#
# # 모델이 예측한 최고 가격 그래프
# plt.plot(pred_y_upper, color = 'blue')
#
# # 실제 가격 그래프
# plt.plot(test_y, color = 'green')
#
# plt.legend(['예측값', '최저값','최대값','실제값'])
# plt.title("값 비교")
# plt.show()

# Step 3 활용: 더나은 결과를 위한 방법
"""
이번 분석 단계에서는 모델의 성능을 조금 더 향상시킬 수 있는 방법들에 대해 알아보겠다. 
첫 번재로 고려해볼 방법은 상한값 혹은 하한값을 설정해 주는 것이다. 
바닥과 천장이 없는 주가 데이터의 경우에는 의미가 없을 수 있지만 일반적인 시계열 데이터에서는 상한값 혹은 하한값을 설정해 주는 것이 모델의 성능을 높여줄 수 있는 방법 중 하나이다. 
Prophet 모델에서는 future_data['cap']=130을 통해 데이터셋에 상한선을 설정할 수 있다. 다음 코드와 실행 결과는 상한선을 적용한 학습 결과를 시각화한 것이다. 
겉으로 보기에는 원래의 결과와 별 차이가 없어 보인다
"""
# plt.clf()
# # 상한가 설정
# eh_train_df['cap'] = 130
#
# # 상한가 적용을 위한 파라미터를 다음과 같이 설정
# prophet = Prophet(seasonality_mode = 'multiplicative',
#                  growth = 'logistic',
#                  yearly_seasonality = True,
#                  weekly_seasonality = True,
#                  daily_seasonality = True,
#                  changepoint_prior_scale = 0.5)
#
# prophet.fit(eh_train_df)
#
# # 5일 예측
# future_data = prophet.make_future_dataframe(periods = 5, freq = 'd')
#
# # 상한가 설정
# future_data['cap'] = 130
# forecast_data = prophet.predict(future_data)
#
# fig = prophet.plot(forecast_data)
# plt.show()


#이상치 제거
"""
이번에는 모델의 성능을 향상시키는 다른 방법중 하나인 이상치 제거 기법을 살펴보자. 이상치란 평균적인 수치에 비해 지나치게 높거나 낮은 수치의 데이터를 의미한다. fbprophet 모델이 이상치를 제거한 데이터로 학습하려면 이상치에 해당하는 데이터를 None로 설정해주면 된다. 이번에는 100이 넘는 데이터를 이상치라고 설정 해주었다.

이상치 100넘는 값 제거
우선 이상치를 100으로 해놓고 100이상인 것들을 제거 해보자.
"""
# plt.clf()
eh_train_df.loc[eh_train_df['y'] > 100, 'y'] = None

# prophet 모델 학습
prophet = Prophet(seasonality_mode='multiplicative',
                 yearly_seasonality = True,
                 weekly_seasonality = True,
                 daily_seasonality = True,
                 changepoint_prior_scale = 0.5)

prophet.fit(eh_train_df)

# 5일 예측
future_data = prophet.make_future_dataframe(periods=5, freq='d')

forecast_data = prophet.predict(future_data)
pred_fbprophet_y_60 = forecast_data.yhat.values[-5:]
# fig = prophet.plot(forecast_data)
# plt.show()

"""
이번에는 이상치를 60이라고 지정 해주고 이를 제거 해보자.
"""
plt.clf()
eh_train_df.loc[eh_train_df['y'] > 60, 'y'] = None

# prophet 모델 학습
prophet = Prophet(seasonality_mode='multiplicative',
                 yearly_seasonality = True,
                 weekly_seasonality = True,
                 daily_seasonality = True,
                 changepoint_prior_scale = 0.5)

prophet.fit(eh_train_df)

# 5일 예측
future_data = prophet.make_future_dataframe(periods=5, freq='d')

forecast_data = prophet.predict(future_data)

# fig = prophet.plot(forecast_data)
# plt.show()

# 이값을 예상치에 넣어 둔다.
pred_fbprophet_y_60 = forecast_data.yhat.values[-5:]



## 전체 graph 이것은 이전의 arima도 포함되므로 나중에 별도로 통합 문서를 만들예정
# df = pd.DataFrame({'ARIMA 예측값':pred_arima_y,
#                    'FBprophet 예측값':  pred_fbprophet_y,
#                    'FBprophet 이상치 제거 후(100) 예측값':pred_fbprophet_y_100,
#                    'FBprophet 이상치 제거 후(60) 예측값':pred_fbprophet_y_60,
#                    '실제값':test_y})
#
# from sklearn.metrics import mean_squared_error, r2_score
# from math import sqrt
#
# plt.figure(figsize=(15, 10))
#
# # arima 모델의  rmse
# rmse_arima = sqrt(mean_squared_error(pred_arima_y, test_y))
#
# # fbprophet 모델의  rmse
# rmse_fbprophet = sqrt(mean_squared_error(pred_fbprophet_y, test_y))
#
# # 전처리 진행한 fbprophet (이상치 100) 모델의  rmse
# rmse_fbprophet_100 = sqrt(mean_squared_error(pred_fbprophet_y_100, test_y))
#
# # 전처리 진행한 fbprophet (이상치 60) 모델의  rmse
# rmse_fbprophet_60 = sqrt(mean_squared_error(pred_fbprophet_y_60, test_y))
#
#
# # 모델이 예측한 가격 그래프
# plt.plot(df[['ARIMA 예측값']], color = 'gold')
#
# # 모델이 예측한 최저 가격 그래프
# plt.plot(df[['FBprophet 예측값']], color = 'red')
#
# # 모델이 예측한 최고 가격 그래프
# plt.plot(df[['FBprophet 이상치 제거 후(100) 예측값']], color = 'blue')
#
# # 모델이 예측한 최고 가격 그래프
# plt.plot(df[['FBprophet 이상치 제거 후(60) 예측값']], color = 'purple')
#
# # 실제 가격 그래프
# plt.plot(test_y, color = 'green')
#
# plt.rc('legend', fontsize=16)
# plt.legend(['ARIMA 예측값 \n -RMSE:' + str(round(rmse_arima,0)),
#             'FBprophet 예측값  \n -RMSE:' + str(round(rmse_fbprophet,0)),
#             'FBprophet 이상치 제거 후(100) 예측값 \n -RMSE:' + str(round(rmse_fbprophet_100,0)),
#             'FBprophet 이상치 제거 후(60) 예측값 \n -RMSE:' + str(round(rmse_fbprophet_60,0)),
#            '실제가격그래프' ])
# plt.title("이항 주가의 예측값 실제값 비교")
