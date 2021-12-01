import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from fbprophet import Prophet  # https://daewonyoon.tistory.com/266

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
print(eh_train_df, eh_test_df)
prophet = Prophet(seasonality_mode='multiplicative',
                 yearly_seasonality=True,
                 weekly_seasonality=True,
                 daily_seasonality=True,
                 changepoint_prior_scale=0.5)

prophet.fit(eh_train_df)

