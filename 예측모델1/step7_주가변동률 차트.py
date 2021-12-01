import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

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



# x일 행에 x+1일의 종가를 추가
eh_df_1['tomorrow Adj Close'] = eh_df_1['Adj Close'].shift(-1)

# 변동율 측정 => ((다음날종가)-(오늘종가))/오늘종가
eh_df_1['Fluctuation'] = eh_df_1['tomorrow Adj Close']-eh_df_1['Adj Close']
eh_df_1['Fluctuation Rate'] = eh_df_1['Fluctuation']/eh_df_1['Adj Close']
eh_df_1['price'] = eh_df_1['Adj Close']

plt.figure(figsize=(12,8))
plt.plot(eh_df_1.index,eh_df_1['Fluctuation Rate'], color = 'lightblue')
plt.axhline(y=0, color='red', ls='--')  # 기준선추가 , axhline() 함수의 첫번째 인자는 y 값으로서 수평선의 위치가 된다.
plt.show()

# # 추가로 먼저 Train/Test set으로 나눌것인데 마지막 날짜 5일을 제외한 나머지를 Train으로 지정 해주고 마지막 날짜 5일을 Test으로 지정 해주었다.
# eh_train_df = eh_df_1[:335]
# eh_train_df
#
# eh_test_df = eh_df_1[335:]
# eh_test_df



