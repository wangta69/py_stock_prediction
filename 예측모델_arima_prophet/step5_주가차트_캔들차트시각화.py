import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from mplfinance.original_flavor import candlestick2_ohlc

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

eh_df_1 = eh_df_1[["Open", "High", "Low", "Close", "Adj Close"]]

#차트와 피규어 그리기 준비 및 크기 설정
fig, ax = plt.subplots(figsize=(15, 7))


# 시고저종 데이터를 통해 캔들 차트를 그리기 (순서를 반드시 시고저종으로 입력)
candlestick2_ohlc(
    ax,
    eh_df_1['Open'],
    eh_df_1['High'],
    eh_df_1['Low'],
    eh_df_1['Close'],
    width=0.6,
    colorup='r',
    colordown='b'
)


# 차트 타이틀 설정
plt.title('Candle stick S elec.')
plt.show()

# # 2. 데이터확인
# print(eh_df_1.shape)
# print(eh_df_1.info())
# eh_df_1.tail()
#
# # att확인
# eh_df_1.columns
#
# # type확인
# eh_df_1.dtypes
#
# #describe() 메소드로 기본정보 확인
# eh_df_1.describe()

# 3.이동평균선 계산해서 att추가하기



