import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from fbprophet import Prophet
import platform

from real_prophet.connMysql import Mysql


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
        period = 10
        if code:
            df = self.mysql.prices(code, getCount)
            df = df.sort_values(by=['ymd'])
            """
                        ymd   open   high    low  close
            206  2021-02-01  81700  83400  81000  83000
            205  2021-02-02  84100  86400  83700  84400
            204  2021-02-03  84800  85400  83400  84600
            203  2021-02-04  83500  83800  82100  82500
            202  2021-02-05  83100  84000  82500  83500
            ..          ...    ...    ...    ...    ...
            4    2021-11-25  75100  75100  73600  73700
            3    2021-11-26  73500  74100  72000  72300
            2    2021-11-29  71700  73000  71400  72300
            1    2021-11-30  73200  73900  70500  71300
            0    2021-12-01  72000  74800  71600  74400
            """
            df = df.set_index('ymd')
            """
                         open   high    low  close
            ymd                                   
            2021-02-01  81700  83400  81000  83000
            2021-02-02  84100  86400  83700  84400
            2021-02-03  84800  85400  83400  84600
            2021-02-04  83500  83800  82100  82500
            2021-02-05  83100  84000  82500  83500
            ...           ...    ...    ...    ...
            2021-11-25  75100  75100  73600  73700
            2021-11-26  73500  74100  72000  72300
            2021-11-29  71700  73000  71400  72300
            2021-11-30  73200  73900  70500  71300
            2021-12-01  72000  74800  71600  74400
            """

            df = df[["close"]]
            df = df.reset_index()
            df.columns = ['ds', 'y']
            """
                         ds      y
            0    2021-02-01  83000
            1    2021-02-02  84400
            2    2021-02-03  84600
            3    2021-02-04  82500
            4    2021-02-05  83500
            ..          ...    ...
            202  2021-11-25  73700
            203  2021-11-26  72300
            204  2021-11-29  72300
            205  2021-11-30  71300
            206  2021-12-01  74400
            """
            print(len(df))
            sampleCount = len(df) - period  # 5개는
            train_df = df[:sampleCount]
            real_df = df[sampleCount:]
            print(train_df)
            print(real_df)

            prophet = Prophet(seasonality_mode='multiplicative',
                              yearly_seasonality=True,
                              weekly_seasonality=True,
                              daily_seasonality=True,
                              changepoint_prior_scale=0.5)
            prophet.fit(train_df)
            # make_future_dataframe(m, periods, freq = "day", include_history = TRUE)
            # m : Prophet model object.
            # periods : Int number of periods to forecast forward. (예측할 기간)
            # freq : 'day', 'week', 'month', 'quarter', 'year', 1(1 sec), 60(1 minute) or 3600(1 hour).
            # include_history : Boolean to include the historical dates in the data frame for predictions.

            future_data = prophet.make_future_dataframe(periods=period, freq='d')
            forecast_data = prophet.predict(future_data)
            forecast_data = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period)

            real_y = real_df.y.values
            fbprophet_y = forecast_data.yhat.values[-period:]

            print(real_y)
            # print(df[real_y])
            plt.plot(real_y, color='gold')
            plt.plot(fbprophet_y, color='red')
            plt.legend(['실제값', '추정치'])
            plt.show()

        pass


if __name__ == "__main__":
    pred = Predictor()
    pred.byCnn('005930')  # 삼성전자 (005930)
    # pred.byProphet('253450')
    # pred.byProphet('011210')
    # pred.byProphet('000660') # sk 하이닉스
    # pred.byProphet('272210')
    # pred.byProphet('278280') # 천보
