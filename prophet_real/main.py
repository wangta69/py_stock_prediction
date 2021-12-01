import matplotlib.pyplot as plt
from fbprophet import Prophet
import platform
from matplotlib import font_manager, rc
from prophet_real.connMysql import Mysql

class Predictor():
    def __init__(self):
        super().__init__()
        self.mysql = Mysql()
        self.plt = self.init_plt()
        print(self.plt)

    def init_plt(self):
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

        return plt

    def byProphet(self, code=None):
        getCount = 120
        if code:
            prices = self.mysql.prices(code, getCount)
            print(prices)

            prophet = Prophet(seasonality_mode='multiplicative',
                              yearly_seasonality=True,
                              weekly_seasonality=True,
                              daily_seasonality=True,
                              changepoint_prior_scale=0.5)

        pass

if __name__ == "__main__":
    pred = Predictor()
    pred.byProphet('005930')