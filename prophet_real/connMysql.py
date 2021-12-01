import os

import pymysql
import pandas as pd
import numpy as np
from dotenv import load_dotenv

class Mysql:
    def __init__(self):
        load_dotenv()

    def corporations(self):
        """
        기업리스트 가져오기
        :return: rows
        """
        conn = self.connect()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as curs:
                sql = "select id, code, code_krx, investing_comp_name, common_stocks from corporations where status = 0"
                curs.execute(sql)

                rs = curs.fetchall()
                return rs
        except Exception as e:
            print('I got a Exception  - reason "%s"' % str(e))
            print(curs._last_executed)
            raise

    def prices(self, code, limit):
        """
        업체별 가격 리스트를 가져온다.
        :return:
        """
        conn = self.connect()
        try:
            sql = "select ymd, open, high, low, close from market_prices where code = %s order by ymd desc limit 0, %s"
            return pd.read_sql_query(sql, conn, params=(code, limit))

            # with conn.cursor(pymysql.cursors.DictCursor) as curs:
            #     sql = "select ymd, open, high, low, close from market_prices where code = %s order by ymd desc limit 0, %s"
            #     curs.execute(sql, (code, limit))
            #
            #     rs = curs.fetchall()
            #     return rs
        except Exception as e:
            print('I got a Exception  - reason "%s"' % str(e))
            # print(curs._last_executed)
            raise


    def close(self):
        self.conn.close()

    def connect(self):
        host = os.getenv('DB_HOST')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        db = os.getenv('DB_DATABASE')
        port = int(os.getenv('DB_PORT'))
        # print(host, port, user, password, db)
        return pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset='utf8')