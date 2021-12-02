import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from mplfinance.original_flavor import candlestick2_ohlc
from datetime import datetime

#matplotlib에서 x축과 y축에 표시되는 값을 ticker라 함
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import numpy as np


# step1_setting_pyplot.py Start
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
# step1_setting_pyplot.py End


#step2_주가가져오기.py start
start = datetime(2021,1,1) #시작날짜
end = datetime(2021,3,2) #끝 날짜
eh_df2 = yf.download('EH', start,end, progress=False)

#step2_주가가져오기.py end

eh_df_2 = eh_df2.reset_index()

eh_df_2.columns = ['day', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

eh_df_2['day'] = pd.to_datetime(eh_df_2['day'])

eh_df_2.index = eh_df_2['day']
eh_df_2.set_index('day', inplace=True)

eh_df_2 = eh_df_2[["Open", "High", "Low", "Close", "Adj Close"]]

#날짜 포멧을 일, 또는 연-월-일로 바꾸기
# print(eh_df2.index[0]) # 2021-01-04 00:00:00
# print(eh_df2.index[0].strftime('%d')) # 04
# print(eh_df2.index[0].strftime('%Y-%m-%d')) # 2021-01-04

#차트와 피규어 그리기 준비 및 크기 설정
fig, ax = plt.subplots(figsize=(13, 7))

#x축의 눈금을 설정
day_list = []
name_list = []
for i, day in enumerate(eh_df2.index):
    day_list.append(i) # 고정 축 list 생성
    name_list.append(day.strftime('%Y-%m-%d'))  # 고정 축에 적힐 날짜 list 생성
ax.xaxis.set_major_locator(ticker.FixedLocator(day_list))  # day_list를 받아서 리스트 안의 값의 위치에 고정 축 생성
ax.xaxis.set_major_formatter(ticker.FixedFormatter(name_list))  # 설정한 고정 축에 name_list 안의 값을 축에 출력

# Open시가, Hihg고가, Low저가, Close종가
candlestick2_ohlc(ax,eh_df2['Open'],eh_df2['High'],eh_df2['Low'],eh_df2['Close'], width=0.5, colorup='r', colordown='b')

plt.xticks(rotation = 90) #x축 각도 회전
#fig.autofmt_xdate() #날짜 형식으로 자동 format 지정
plt.title('Candle stick S elec')
plt.grid()
plt.show()

