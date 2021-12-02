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

#차트에 표현할 요소 설정
plt.plot(eh_df_1['Adj Close'], label="Adj Close")
plt.plot(eh_df_1['MA5'], label="MA5")
plt.plot(eh_df_1['MA20'], label="MA20")
plt.plot(eh_df_1['MA60'], label="MA60")
plt.plot(eh_df_1['MA120'], label="MA120")

#'best'를 인자로 주어 가장 적절한 자리에 위치하게 함
plt.legend(loc='best')

#격자 그리기
plt.grid()
plt.show()



