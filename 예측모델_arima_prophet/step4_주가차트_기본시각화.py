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

eh_df_1.plot()
plt.title("이항 주가")
plt.show()

