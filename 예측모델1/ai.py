# https://dacon.io/codeshare/2588

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# 한글폰트
import platform

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
    print('Mac version')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
    print('Windows version')
elif platform.system() == 'Linux':
    path = "/usr/share/fonts/NanumFont/NanumGothicBold.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    plt.rc('font', family=font_name)
    print('Linux version')
else:
    print('Unknown system... sorry~~~~')

# Step1 탐색: 시간 정보가 포함된 데이터 살펴보기
# 데이타 가져오기
eh_df = yf.download('EH',
                      start='2019-01-01',
                      end='2021-04-22',
                      progress=False)

eh_df = eh_df[["Close"]]

eh_df = eh_df.reset_index()

eh_df.columns = ['day', 'price']

eh_df['day'] = pd.to_datetime(eh_df['day'])

eh_df.index = eh_df['day']
eh_df.set_index('day', inplace=True)

print(eh_df)
"""
day         price
2019-12-13  12.900000
...         ...
2021-04-14  29.400000
"""
# 처음 335 까지는 계산에 포함할 거도
eh_train_df = eh_df[:335]
print(eh_train_df)

# 마지막 5개는 결과 확인용으로 위의 계산 결과와 얼마나 차이 나는지 확인용
eh_test_df  = eh_df[335:]
print(eh_test_df)


# # 그래프 그리기
# fig, ax = plt.subplots(figsize=(15, 8))
# eh_df.plot(ax=ax)
#
# # 주가 최고점
# ax.annotate('', xy=('2021-02-12', 124.089996), xytext=('2021-01-01', 120),
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3,rad=-0.2"),
#            )
#
# plt.text('2020-11-01', 110, "주가 최고점 \n-날짜: 2021-02-12 \n-종가: 124.089996", fontsize=13)
#
#
# # 공매도 리포트 직후 (울프팩리서치의 보고서)
# ax.annotate('', xy=('2021-02-16', 46.299999), xytext=('2021-03-15', 60),
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3,rad=-0.2"),
#            )
# plt.text('2021-03-05',62, "울프팩리서치의 보고서 \n-날짜: 2021-02-16 \n-종가: 46.299999",fontsize=11)
#
#
# # 다음날의 반등
# ax.annotate('', xy=('2021-02-17' ,77.73), xytext=('2021-03-20', 100),
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3,rad=0.2"),
#            )
# plt.text('2021-03-15',103, "다음날 반등 \n-날짜: 2021-02-17 \n-종가: 77.73",fontsize=11)
#
# # Scatter plot 추가
# y1 = ['2021-02-12', '2021-02-16', '2021-02-17']
# y2 = [124.089996, 46.299999, 77.73 ]
#
#
# plt.scatter(y1,y2,s=30,color='r')
#
#
# plt.title("이항 주가")
# plt.show()


# 이항 데이터셋의 기본 정보 구하기
# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

eh_df_1 = yf.download('EH',
                      start='2019-01-01',
                      end='2021-04-22',
                      progress=False)


eh_df_1 = eh_df_1.reset_index()

eh_df_1.columns = ['day', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

eh_df_1['day'] = pd.to_datetime(eh_df_1['day'])

eh_df_1.index = eh_df_1['day']
eh_df_1.set_index('day', inplace=True)

print(eh_df_1)
"""
        Open       High        Low      Close  Adj Close   Volume
day   
2019-12-13  12.450000  13.700000  12.120000  12.900000  12.900000   290800
...               ...        ...        ...        ...        ...      ...
2021-04-21  24.260000  26.680000  23.309999  25.820000  25.820000  2430500
"""

eh_df_vol = eh_df_1[["Volume"]]
print(eh_df_vol)
"""
        Volume
day 
2019-12-13   290800
...             ...
2021-04-21  2430500
"""

eh_df_1 = eh_df_1[["Open", "High", "Low", "Close", "Adj Close"]]
print(eh_df_1)
"""
        Open       High        Low      Close  Adj Close
day 
2019-12-13  12.450000  13.700000  12.120000  12.900000  12.900000
...               ...        ...        ...        ...        ...
2021-04-21  24.260000  26.680000  23.309999  25.820000  25.820000
"""

## 1. 주가 데이터 시각화
# 1) 기본 시각화
# eh_df_1.plot()
# plt.title("이항 주가")
# plt.show()

# 2) 캔들 차트를 사용한 시각화
# from mpl_finance import candlestick2_ohlc # WARNING: `mpl_finance` is deprecated:
from mplfinance.original_flavor import candlestick2_ohlc


import matplotlib.pyplot as plt

#matplotlib에서 x축과 y축에 표시되는 값을 ticker라 함
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import numpy as np
#
# #차트와 피규어 그리기 준비 및 크기 설정
# fig, ax = plt.subplots(figsize=(15,7))
#
#
# # 시고저종 데이터를 통해 캔들 차트를 그리기 (순서를 반드시 시고저종으로 입력)
# candlestick2_ohlc(ax, eh_df_1['Open'], eh_df_1['High'], eh_df_1['Low'], eh_df_1['Close'], width=0.6, colorup='r', colordown='b')
#

# 차트 타이틀 설정
# plt.title('Candle stick S elec.')
# plt.show()


# 3) 봉차트
# 이제 적은기간으로 보기쉽게하기 위한 봉차트를 알아보자.
# 필요한 라이브러리 다운로드
from datetime import datetime
import pandas_datareader.data as wb


# start = datetime(2021,1,1) #시작날짜
# end = datetime(2021,3,2) #끝 날짜
# eh_df2 = yf.download('EH', start,end, progress=False)
#
# # 시간포멧 보기좋게 바꾸는 방법
# #날짜 포멧을 일, 또는 연-월-일로 바꾸기
# print(eh_df2.index[0])
# print(eh_df2.index[0].strftime('%d'))
# print(eh_df2.index[0].strftime('%Y-%m-%d'))

#다시 차트 그리기
# fig, ax = plt.subplots(figsize=(13,7))

#x축의 눈금을 설정
# day_list = []
# name_list = []
# for i, day in enumerate(eh_df2.index):
#     day_list.append(i) # 고정 축 list 생성
#     name_list.append(day.strftime('%Y-%m-%d')) # 고정 축에 적힐 날짜 list 생성
# ax.xaxis.set_major_locator(ticker.FixedLocator(day_list)) #day_list를 받아서 리스트 안의 값의 위치에 고정 축 생성
# ax.xaxis.set_major_formatter(ticker.FixedFormatter(name_list)) #설정한 고정 축에 name_list 안의 값을 축에 출력
#
# # Open시가, Hihg고가, Low저가, Close종가
# candlestick2_ohlc(ax, eh_df2['Open'],eh_df2['High'],eh_df2['Low'],eh_df2['Close'], width=0.5, colorup='r', colordown='b')

# plt.xticks(rotation = 90) #x축 각도 회전
# #fig.autofmt_xdate() #날짜 형식으로 자동 format 지정
# plt.title('Candle stick S elec')
# plt.grid()
# plt.show()


# 2. 데이터확인
# print(eh_df_1.shape)
# print(eh_df_1.info())
# eh_df_1.tail()
"""
(340, 5)
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 340 entries, 2019-12-13 to 2021-04-21
Data columns (total 5 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Open       340 non-null    float64
 1   High       340 non-null    float64
 2   Low        340 non-null    float64
 3   Close      340 non-null    float64
 4   Adj Close  340 non-null    float64
dtypes: float64(5)
memory usage: 15.9 KB
None
"""

# att확인
# print(eh_df_1.columns)
"""
Index(['Open', 'High', 'Low', 'Close', 'Adj Close'], dtype='object')
High: 고가
Low: 저가
Open: 시가
Close: 종가
Volume: 거래량
Adj Close: 수정 종가
"""
# # type확인
# print(eh_df_1.dtypes)
#
# # describe() 메소드로 기본정보 확인
# print(eh_df_1.describe())
"""
             Open        High         Low       Close   Adj Close
count  340.000000  340.000000  340.000000  340.000000  340.000000
mean    20.289003   21.525779   18.815729   20.158100   20.158100
std     19.676354   21.758806   17.447580   19.728020   19.728020
min      7.980000    8.035000    7.590000    7.700000    7.700000
25%     10.007500   10.530000    9.707500   10.037500   10.037500
50%     12.035000   12.300000   11.535000   11.910000   11.910000
75%     20.313499   21.280001   19.251250   19.962499   19.962499
max    123.497002  129.800003  103.800003  124.089996  124.089996
"""

# 3.이동평균선 계산해서 att추가하기
# maN : N일전부터 현재까지의 수정종가 평균
ma5 = eh_df_1['Adj Close'].rolling(window=5).mean()
ma20 = eh_df_1['Adj Close'].rolling(window=20).mean()
ma60 = eh_df_1['Adj Close'].rolling(window=60).mean()
ma120 = eh_df_1['Adj Close'].rolling(window=120).mean()

#새로운 데이터를 삽입하는 코드
eh_df_1.insert(len(eh_df_1.columns), "MA5", ma5)
eh_df_1.insert(len(eh_df_1.columns), "MA20", ma20)
eh_df_1.insert(len(eh_df_1.columns), "MA60", ma60)
eh_df_1.insert(len(eh_df_1.columns), "MA120", ma120)

# (2) 거래량을 이용한 이동평균선 계산, 추가
vma5 = eh_df_vol['Volume'].rolling(window=5).mean()
eh_df_vol.insert(len(eh_df_vol.columns), "VMA5", vma5)

#이동평균값 plot 그리기
# plt.plot(eh_df_1.index,eh_df_1['MA5'], label="MA5")
# plt.legend(loc='best') #라벨 위치 설정
# plt.xticks(rotation=45) #x축 좌표각도
# plt.grid() #격자
# plt.show() #현재까지 그려진 그래프 보여주기



#차트에 표현할 요소 설정
plt.plot(eh_df_1['Adj Close'], label="Adj Close")
plt.plot(eh_df_1['MA5'], label="MA5")
plt.plot(eh_df_1['MA20'], label="MA20")
plt.plot(eh_df_1['MA60'], label="MA60")
plt.plot(eh_df_1['MA120'], label="MA120")

#'best'를 인자로 주어 가장 적절한 자리에 위치하게 함
plt.legend(loc='best')

#격자 그리기
# plt.grid()
# plt.show()

# 4.주가변동율 추가 및 시각화
# (1)변동율 추가
# x일 행에 x+1일의 종가를 추가
eh_df_1['tomorrow Adj Close']= eh_df_1['Adj Close'].shift(-1)

# 변동율 측정 => ((다음날종가)-(오늘종가))/오늘종가
eh_df_1['Fluctuation']= eh_df_1['tomorrow Adj Close']-eh_df_1['Adj Close']
eh_df_1['Fluctuation Rate'] = eh_df_1['Fluctuation']/eh_df_1['Adj Close']
eh_df_1['price']= eh_df_1['Adj Close']
# eh_df_1

eh_train_df = eh_df_1[:335]
print(eh_train_df)
"""
                 Open       High  ...  Fluctuation Rate      price
day                               ...                             
2019-12-13  12.450000  13.700000  ...         -0.031008  12.900000
...               ...        ...  ...               ...        ...
2021-04-14  29.520000  31.000000  ...         -0.015986  29.400000
"""

eh_test_df = eh_df_1[335:]
print(eh_test_df)

# (2) 추가한 데이터로 시각화
# plt.figure(figsize=(12,8))
# plt.plot(eh_df_1.index, eh_df_1['Fluctuation Rate'], color='lightblue')
# plt.axhline(y=0, color='red', ls='--') #기준선추가 , axhline() 함수의 첫번째 인자는 y 값으로서 수평선의 위치가 된다.
# plt.show()


## Step2 예측: ARIMA 예측 모델
# https://m.blog.naver.com/sigmagil/221502504340  : 시계열 예측 모델에 대한 간략한 설명

# from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.arima_model as smt
import statsmodels
# (AR = 2, 차분 =1, MA=2) 파라미터로 ARIMA 모델을 학습한다.
# model = ARIMA(eh_train_df.price.values, order=(2, 1, 2))
# model_fit = model.fit(trend='c', full_output=True, disp=True)

# model_fit = ARIMA(eh_train_df.price.values, order=(2, 1, 2), trend='c').fit()

model = ARIMA(eh_train_df.price.values, order=(2, 1, 2))
model_fit = model.fit()
# model_fit = model.fit(trend='c', full_output=True, disp=True)

print('statsmodels.__version__:', statsmodels.__version__)
print('1111111111111111111111')
print(model_fit.summary())
print('12121212121212121212121211212')
print(model_fit.forecast(steps=len(eh_train_df)))
print('2222222222222222222222')
# 학습 데이터에 대한 예측 결과
# fig = model_fit.plot_predict()
print('3333333333333333333333')
# 잔차의 변동을 시각화
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
# 이항주가의 변동률과, ARIMA 모델의 잔차 그래프
plt.plot(eh_df_1['Fluctuation Rate'], color = 'lightblue')
plt.axhline(y=0, color='red', ls='--') #기준선추가 , axhline() 함수의 첫번째 인자는 y 값으로서 수평선의 위치가 됩니다.
plt.title('이항 주가 변동률')
plt.show()

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title('ARIMA 모델의 잔차 그래프')