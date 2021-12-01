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
#step2_주가가져오기.py end


fig, ax = plt.subplots(figsize=(15, 8))
eh_df.plot(ax=ax)

# plt.text(x, y, 'your texxt') : x, y 위치에 text 출력
# ax.annotate(text, xy, xytext, xycoords)
#   - text(str): The text of the annotation
#   - xy(float, float) : The point (x, y) to annotate. The coordinate system is determined by xycoords.
#   - xytext(float, float), default: xy : The position (x, y) to place the text at. The coordinate system is determined by textcoords.
#   - xycoordsstr or Artist or Transform or callable or (float, float), default: 'data'
#       The coordinate system that xy is given in. The following types of values are supported:
# 주가 최고점
ax.annotate(
            '',
            xy=('2021-02-12', 124.089996),
            xytext=('2021-01-01', 120),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
           )
plt.text('2020-11-01', 110, "주가 최고점 \n-날짜: 2021-02-12 \n-종가: 124.089996", fontsize=13)


# 공매도 리포트 직후 (울프팩리서치의 보고서)
ax.annotate(
            '',
            xy=('2021-02-16', 46.299999),
            xytext=('2021-03-15', 60),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
           )
plt.text('2021-03-05', 62, "울프팩리서치의 보고서 \n-날짜: 2021-02-16 \n-종가: 46.299999", fontsize=11)


# 다음날의 반등
ax.annotate(
            '',
            xy=('2021-02-17', 77.73),
            xytext=('2021-03-20', 100),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
           )

plt.text('2021-03-15', 103, "다음날 반등 \n-날짜: 2021-02-17 \n-종가: 77.73", fontsize=11)

# Scatter plot 추가
y1 = ['2021-02-12', '2021-02-16', '2021-02-17']
y2 = [124.089996, 46.299999, 77.73 ]

plt.scatter(y1, y2, s=30, color='r')
plt.title("이항 주가")
plt.show()