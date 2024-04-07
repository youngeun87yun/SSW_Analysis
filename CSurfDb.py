import sqlite3 as lite
import pandas as pd
import datetime

database_name = 'C:\\DATA\\sqlite\\PINO_DB.db'

surf_interest_tbl_name = 'surf_interest_tbl'
surf_min1_tbl_name = 'surf_min1_tbl'
surf_min5_tbl_name = 'surf_min5_tbl'
surf_min60_tbl_name = 'surf_min60_tbl'
surf_daily_tbl_name = 'surf_daily_tbl'
surf_weekly_tbl_name = 'surf_weekly_tbl'
surf_monthly_tbl_name = 'surf_monthly_tbl'
surf_code_tbl_name = 'surf_code_tbl'
surf_tick_tbl_name = 'surf_tick_tbl'
surf_real_min5_tbl_name = 'surf_real_min5_tbl'

surf_interest_temp_tbl_name = 'surf_interest_temp_tbl'
surf_daily_temp_tbl_name = 'surf_daily_temp_tbl'
surf_weekly_temp_tbl_name = 'surf_weekly_temp_tbl'
surf_monthly_temp_tbl_name = 'surf_monthly_temp_tbl'
surf_min_temp_tbl_name = 'surf_min_temp_tbl'
surf_trading_v_tbl_name = 'surf_trading_v_tbl'
surf_trading_v_result_tbl_name = 'surf_trading_v_result_tbl'

min_columns = ['Code', 'D', 'T', 'O', 'H', 'L', 'C', 'V', 'VS',
               'MACD', 'SIGNAL', 'OSC',
               'MV_5', 'MV_10', 'MV_20', 'MV_60', 'MV_120', 'MV_300',
               'MV_Vol_5', 'MV_Vol_20',
               'PER_MVA_5', 'PER_MVA_10', 'PER_MVA_20', 'PER_MVA_60', 'PER_MVA_120', 'PER_MVA_300',
               'VAR_24', 'VAR_60',
               'DT_MACD', 'DT_SIGNAL', 'DT_OSC',
               'DT_MV_5', 'DT_MV_10', 'DT_MV_20', 'DT_MV_60', 'DT_MV_120', 'DT_MV_300',
               'DT_MV_Vol_5', 'DT_MV_Vol_20',
               'CR', 'HR',
               'C_Bef', 'V_Bef', 'MV_Vol_5_Bef', 'MV_Vol_20_Bef',
               'RSI_1', 'RSI_2'
               ]

day_columns = ['Code', 'D', 'O', 'H', 'L', 'C', 'V', 'VS',
               'MACD', 'SIGNAL', 'OSC',
               'MV_5', 'MV_10', 'MV_20', 'MV_60', 'MV_120', 'MV_300',
               'MV_Vol_5', 'MV_Vol_20',
               'PER_MVA_5', 'PER_MVA_10', 'PER_MVA_20', 'PER_MVA_60', 'PER_MVA_120', 'PER_MVA_300',
               'VAR_24', 'VAR_60',
               'DT_MACD', 'DT_SIGNAL', 'DT_OSC',
               'DT_MV_5', 'DT_MV_10', 'DT_MV_20', 'DT_MV_60', 'DT_MV_120', 'DT_MV_300',
               'DT_MV_Vol_5', 'DT_MV_Vol_20',
               'CR', 'HR',
               'C_Bef', 'V_Bef', 'MV_Vol_5_Bef', 'MV_Vol_20_Bef',
               'RSI_1', 'RSI_2'
               ]

day_columns_with_month = \
              ['Code', 'D', 'O', 'H', 'L', 'C', 'V', 'VS',
               'MACD', 'SIGNAL', 'OSC',
               'MV_5', 'MV_10', 'MV_20', 'MV_60', 'MV_120', 'MV_300',
               'MV_Vol_5', 'MV_Vol_20',
               'PER_MVA_5', 'PER_MVA_10', 'PER_MVA_20', 'PER_MVA_60', 'PER_MVA_120', 'PER_MVA_300',
               'VAR_24', 'VAR_60',
               'DT_MACD', 'DT_SIGNAL', 'DT_OSC',
               'DT_MV_5', 'DT_MV_10', 'DT_MV_20', 'DT_MV_60', 'DT_MV_120', 'DT_MV_300',
               'DT_MV_Vol_5', 'DT_MV_Vol_20',
               'CR', 'HR',
               'C_Bef', 'V_Bef', 'MV_Vol_5_Bef', 'MV_Vol_20_Bef',
               'RSI_1', 'RSI_2',
               'prevD', 'prevM', 'pM_MV_5', 'pM_MV_20', 'pM_MV_60', 'pM_MV_120', 'pM_MV_300']

class CSurfDB:
    def __init__(self):
        # print('CSurfDB:init()')
        self.conn = lite.connect(database_name)
        self.cur = self.conn.cursor()
        return

    def InterestTable_Insert(self, codes_list):
        # 1. first drop table
        col_name = ['Code']
        codes_df = pd.DataFrame(codes_list, columns=col_name)
        codes_df.to_sql(surf_interest_tbl_name, self.conn, if_exists='replace', index=False)
        return

    def InterestTable_Get(self):
        codes_list = []
        sql = "SELECT DISTINCT(Code ) FROM " + surf_interest_tbl_name + " WHERE D >= 20220901"
        self.cur.execute(sql)
        stock_codes = self.cur.fetchall()
        codes_list = [datum[0] for datum in stock_codes]
        return codes_list

    def InterestTable_Insert2(self, df):
        # print('InterestTable_Insert2')
        target_tbl_name = surf_interest_tbl_name
        target_temp_tbl_name = surf_interest_temp_tbl_name

        df.to_sql(surf_interest_temp_tbl_name, self.conn, if_exists='replace', index=False)

        sql = "INSERT INTO " + target_tbl_name + \
              " SELECT * " + \
              " FROM " + surf_interest_temp_tbl_name + " p " + \
              " WHERE NOT EXISTS " + \
              "   (SELECT * FROM " + target_tbl_name + " sub " + \
              "    WHERE (sub.Code = p.Code) AND (sub.D = p.D) AND (sub.T1 = p.T1)  AND (sub.T2 = p.T2));"

        # print(sql)
        self.cur.execute(sql)
        self.conn.commit()
        return

    def InsertToSurfDaily(self, df):
        # print('InsertToSurfDaily')
        target_tbl_name = surf_daily_tbl_name
        target_temp_tbl_name = surf_daily_temp_tbl_name

        df.to_sql(target_temp_tbl_name, self.conn, if_exists='replace', index=False)

        sql = "INSERT INTO " + target_tbl_name + \
              " SELECT * " + \
              " FROM " + target_temp_tbl_name + " p " + \
              " WHERE NOT EXISTS " + \
              "   (SELECT * FROM " + target_tbl_name + " sub " + \
              "    WHERE (sub.Code = p.Code) AND (sub.D = p.D));"

        # print(sql)
        self.cur.execute(sql)
        self.conn.commit()
        return

    def InsertToSurfWeekly(self, df):
        # print('InsertToSurfWeekly')
        target_tbl_name = surf_weekly_tbl_name
        target_temp_tbl_name = surf_weekly_temp_tbl_name

        df.to_sql(target_temp_tbl_name, self.conn, if_exists='replace', index=False)

        sql = "INSERT INTO " + target_tbl_name + \
              " SELECT * " + \
              " FROM " + target_temp_tbl_name + " p " + \
              " WHERE NOT EXISTS " + \
              "   (SELECT * FROM " + target_tbl_name + " sub " + \
              "    WHERE (sub.Code = p.Code) AND (sub.D = p.D));"

        # print(sql)
        self.cur.execute(sql)
        self.conn.commit()
        return

    def InsertToSurfMonthly(self, df):
        # print('InsertToSurfMonthly')
        target_tbl_name = surf_monthly_tbl_name
        target_temp_tbl_name = surf_monthly_temp_tbl_name

        df.to_sql(target_temp_tbl_name, self.conn, if_exists='replace', index=False)

        sql = "INSERT INTO " + target_tbl_name + \
              " SELECT * " + \
              " FROM " + target_temp_tbl_name + " p " + \
              " WHERE NOT EXISTS " + \
              "   (SELECT * FROM " + target_tbl_name + " sub " + \
              "    WHERE (sub.Code = p.Code) AND (sub.D = p.D));"

        # print(sql)
        self.cur.execute(sql)
        self.conn.commit()
        return

    def DailyTable_Get_by_Code(self, code):
        sql = "SELECT * FROM " + surf_daily_tbl_name + " WHERE Code='" + code + "' ORDER BY D"
        self.cur.execute(sql)
        data_tuples = self.cur.fetchall()
        # print(data_tuples)
        data_list = list(data_tuples)
        df = pd.DataFrame(data_list, columns=day_columns)
        return df

    def DailyTable_Get_by_Code_with_Month_Info(self, code):
        # sql = "SELECT * FROM " + surf_daily_tbl_name + " WHERE Code='" + code + "' ORDER BY D"
        sql = "WITH surf_daily2_tbl " + \
              "    AS ( SELECT sdt. *, LAG(D, 1) OVER(PARTITION BY Code ORDER BY D) AS prevD, " + \
              "         (LAG(D, 1) OVER(PARTITION BY Code ORDER BY D) - LAG(D, 1) OVER(PARTITION BY Code ORDER BY D) % 100) AS prevM " + \
              "         FROM surf_daily_tbl sdt " \
              "         WHERE Code='" + code + "' ORDER BY D ) " + \
              " SELECT sdt2. *, smt.MV_5 AS pM_MV_5, smt.MV_20 AS pM_MV_20, smt.MV_60 AS pM_MV_60, smt.MV_120  AS pM_MV_120, smt.MV_300  AS pM_MV_300 " + \
              "    FROM surf_daily2_tbl sdt2 " + \
              "    LEFT JOIN surf_monthly_tbl smt ON(sdt2.Code = smt.Code) AND(sdt2.prevM = smt.D) "
        # print(sql)
        self.cur.execute(sql)
        data_tuples = self.cur.fetchall()
        # print(data_tuples)
        data_list = list(data_tuples)
        df = pd.DataFrame(data_list, columns=day_columns_with_month)
        return df

    def InsertToSurfmin60(self, df):
        # print('InsertToSurfmin60')
        target_tbl_name = surf_min60_tbl_name
        target_temp_tbl_name = surf_min_temp_tbl_name

        df.to_sql(target_temp_tbl_name, self.conn, if_exists='replace', index=False)

        sql = "INSERT INTO " + target_tbl_name + \
              " SELECT * " + \
              " FROM " + target_temp_tbl_name + " p " + \
              " WHERE NOT EXISTS " + \
              "   (SELECT * FROM " + target_tbl_name + " sub " + \
              "    WHERE (sub.Code = p.Code) AND (sub.D = p.D) AND (sub.T = p.T));"

        # print(sql)
        self.cur.execute(sql)
        self.conn.commit()
        return

    def InsertToSurfmin5(self, df):
        # print('InsertToSurfmin5')
        target_tbl_name = surf_min5_tbl_name
        target_temp_tbl_name = surf_min_temp_tbl_name

        df.to_sql(target_temp_tbl_name, self.conn, if_exists='replace', index=False)

        sql = "INSERT INTO " + target_tbl_name + \
              " SELECT * " + \
              " FROM " + target_temp_tbl_name + " p " + \
              " WHERE NOT EXISTS " + \
              "   (SELECT * FROM " + target_tbl_name + " sub " + \
              "    WHERE (sub.Code = p.Code) AND (sub.D = p.D) AND (sub.T = p.T));"

        # print(sql)
        self.cur.execute(sql)
        self.conn.commit()
        return

    def InsertToSurfmin1(self, df):
        # print('InsertToSurfmin1')
        target_tbl_name = surf_min1_tbl_name
        target_temp_tbl_name = surf_min_temp_tbl_name

        df.to_sql(target_temp_tbl_name, self.conn, if_exists='replace', index=False)

        sql = "INSERT INTO " + target_tbl_name + \
              " SELECT * " + \
              " FROM " + target_temp_tbl_name + " p " + \
              " WHERE NOT EXISTS " + \
              "   (SELECT * FROM " + target_tbl_name + " sub " + \
              "    WHERE (sub.Code = p.Code) AND (sub.D = p.D) AND (sub.T = p.T));"

        # print(sql)
        self.cur.execute(sql)
        self.conn.commit()
        return

    def CodeList_Insert(self, df):
        # print('CodeList_Insert')
        target_tbl_name = surf_code_tbl_name
        df.to_sql(target_tbl_name, self.conn, if_exists='replace', index=False)
        return

    def CodeList_Get(self):
        codes_list = []
        sql = "SELECT Code FROM " + surf_code_tbl_name
        self.cur.execute(sql)
        stock_codes = self.cur.fetchall()
        codes_list = [datum[0] for datum in stock_codes]
        # codes_list = list(stock_codes)
        return codes_list

    def Delete_All_Tables_Records(self):
        # print('Delete_All_Tables_Records')

        # 1. Delete records of surf_code_table
        sql = "DELETE FROM " + surf_code_tbl_name + " WHERE TRUE; "
        self.cur.execute(sql)
        self.conn.commit()

        # 2. Delete records of surf_monthly_table
        sql = "DELETE FROM " + surf_monthly_tbl_name + " WHERE TRUE; "
        self.cur.execute(sql)
        self.conn.commit()

        # 3. Delete records of surf_weekly_table
        sql = "DELETE FROM " + surf_weekly_tbl_name + " WHERE TRUE; "
        self.cur.execute(sql)
        self.conn.commit()


        # 4. Delete records of surf_daily_tbl
        sql = "DELETE FROM " + surf_daily_tbl_name + " WHERE TRUE; "
        self.cur.execute(sql)
        self.conn.commit()

        # 5. Delete records of surf_min60_tbl
        sql = "DELETE FROM " + surf_min60_tbl_name + " WHERE TRUE; "
        self.cur.execute(sql)
        self.conn.commit()

        # 6. Delete records of surf_min5_tbl
        sql = "DELETE FROM " + surf_min5_tbl_name + " WHERE TRUE; "
        self.cur.execute(sql)
        self.conn.commit()

        # 7. Delete records of surf_min1_tbl
        sql = "DELETE FROM " + surf_min1_tbl_name + " WHERE TRUE; "
        self.cur.execute(sql)
        self.conn.commit()

        return

    # 일 / 주 / 월 table 에서 특정 일자 부터 data 를 삭제한다.
    # DWM 은 'D':일, 'W':주, 'M':월 을 의미하며, 문자열내에 모두 포함할 수 있다. ex "DWM" " 일/주/월 모두를 의미
    def Delete_DWM_From(self, DWM, from_date):

        if DWM.find('D') >= 0:
            # print("daily record delete from ", from_date)
            sql = "DELETE FROM " + surf_daily_tbl_name + " WHERE D='" + from_date + "'"
            self.cur.execute(sql)
            self.conn.commit()

        if DWM.find('W') >= 0:
            # print("weekly record delete from ", from_date)
            sql = "DELETE FROM " + surf_weekly_tbl_name + " WHERE D='" + from_date + "'"
            self.cur.execute(sql)
            self.conn.commit()

        if DWM.find('M') >= 0:
            # print("monthly record delete from ", from_date)
            sql = "DELETE FROM " + surf_monthly_tbl_name + " WHERE D='" + from_date + "'"
            self.cur.execute(sql)
            self.conn.commit()
        return

    def GetDailyData(self, code):

        sql = "SELECT COUNT(*) FROM " + surf_daily_tbl_name + " WHERE Code='" + code + "'"
        self.cur.execute(sql)
        record_count_tuples = self.cur.fetchall()
        record_count = [datum[0] for datum in record_count_tuples][0]

        # 최소 200 개의 Record 가 있어야 MACD 가 제대로 계산됨
        if record_count > 200:
            record_offset = record_count - 200
        else:
            record_offset = record_count

        sql = "SELECT D, O, H, L, C, V FROM surf_daily_tbl WHERE Code=? ORDER BY D ASC LIMIT 200 OFFSET ?"

        self.cur.execute(sql, (code, record_offset))
        record_tuples = self.cur.fetchall()

        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        vols = []
        times = []

        for datum in record_tuples:
            dates.append(datum[0])  # D
            opens.append(datum[1])
            highs.append(datum[2])
            lows.append(datum[3])
            closes.append(datum[4])
            vols.append(datum[5])

        return dates, opens, highs, lows, closes, vols, times

    def GetDailyData_all(self, code):

        sql = "SELECT COUNT(*) FROM " + surf_daily_tbl_name + " WHERE Code='" + code + "'"
        self.cur.execute(sql)
        record_count_tuples = self.cur.fetchall()
        record_count = [datum[0] for datum in record_count_tuples][0]

        record_offset = 0

        sql = "SELECT D, O, H, L, C, V FROM surf_daily_tbl WHERE Code=? ORDER BY D ASC LIMIT 10000 OFFSET ? "

        self.cur.execute(sql, (code, record_offset))
        record_tuples = self.cur.fetchall()

        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        vols = []
        times = []

        for datum in record_tuples:
            dates.append(datum[0])  # D
            opens.append(datum[1])
            highs.append(datum[2])
            lows.append(datum[3])
            closes.append(datum[4])
            vols.append(datum[5])

        return dates, opens, highs, lows, closes, vols, times

    def TickTable_Insert(self, df):
        # print('InsertToCodeList')
        target_tbl_name = surf_tick_tbl_name
        df.to_sql(target_tbl_name, self.conn, if_exists='append', index=False)
        return

    def Real5MinTable_Insert(self, df):
        # print('InsertToCodeList')
        target_tbl_name = surf_real_min5_tbl_name
        df.to_sql(target_tbl_name, self.conn, if_exists='append', index=False)
        return

    def Min60_Table_Get(self, code):

        sql = "SELECT * FROM " + surf_min60_tbl_name + " WHERE Code='" + code + "' ORDER BY D, T"
        self.cur.execute(sql)
        data_tuples = self.cur.fetchall()
        data_list = list(data_tuples)
        df = pd.DataFrame(data_list, columns=min_columns)
        return df

    def Min5_Table_Get(self, code):

        sql = "SELECT * FROM " + surf_min5_tbl_name + " WHERE Code='" + code + "' ORDER BY D, T"
        self.cur.execute(sql)
        data_tuples = self.cur.fetchall()
        data_list = list(data_tuples)
        df = pd.DataFrame(data_list, columns=min_columns)
        return df

    def Min1_Table_Get(self, code):

        sql = "SELECT * FROM " + surf_min1_tbl_name + " WHERE Code='" + code + "' ORDER BY D, T"
        self.cur.execute(sql)
        data_tuples = self.cur.fetchall()
        data_list = list(data_tuples)
        df = pd.DataFrame(data_list, columns=min_columns)
        return df

    def MonthlyTable_Get_by_Code(self, code):

        sql = "SELECT * FROM " + surf_monthly_tbl_name + " WHERE Code='" + code + "' ORDER BY D"
        self.cur.execute(sql)
        data_tuples = self.cur.fetchall()
        data_list = list(data_tuples)
        df = pd.DataFrame(data_list, columns=day_columns)
        return df

    def WeeklyTable_Get_by_Code(self, code):

        sql = "SELECT * FROM " + surf_weekly_tbl_name + " WHERE Code='" + code + "' ORDER BY D"
        self.cur.execute(sql)
        data_tuples = self.cur.fetchall()
        data_list = list(data_tuples)
        df = pd.DataFrame(data_list, columns=day_columns)
        return df

    def Trading_V_Table_Insert(self, stock_code, date, time, price, vol, flag, Al_type, msg):
        col_name = ['datetime', 'Code', 'D', 'T', 'C', 'V', 'F', 'Al_T', 'MSG']
        a_raw = [(datetime.datetime.now(), stock_code, date, time, price, vol, flag, Al_type, msg)]
        codes_df = pd.DataFrame(a_raw, columns=col_name)
        codes_df.to_sql(surf_trading_v_tbl_name, self.conn, if_exists='append', index=False)
        return

    def Trading_V_Table_Get_BY_Code_Date(self, stock_code, date, Al_type):

        # Trading V Table 에서 일자의 Code 와 Type 에 해당하는 List 조회

        sql = "SELECT * FROM " + surf_trading_v_tbl_name + " WHERE (Code=?) AND (D=?) AND (Al_T=?) ORDER BY datetime"
        self.cur.execute(sql, (stock_code, date, Al_type))
        trading_v_lists = self.cur.fetchall()

        return trading_v_lists

    def Trading_V_Table_Get_Codes_BY_Date_AlType(self, date, Al_type):

        # Trading V Table 에서 일자의 Code 와 Type 에 해당하는 List 조회

        sql = "SELECT DISTINCT(Code) FROM " + surf_trading_v_tbl_name + " WHERE (D=?) AND (Al_T=?)"
        self.cur.execute(sql, (date, Al_type))
        stock_codes = self.cur.fetchall()
        code_list = [datum[0] for datum in stock_codes]

        return code_list

    def Trading_V_Table_Get_AlTypes_BY_Date(self, q_date):

        # Trading V Table 에서 일자의 Code 와 Type 에 해당하는 List 조회

        sql = 'SELECT DISTINCT(Al_T) FROM {0}  WHERE D={1}'.format(surf_trading_v_tbl_name, q_date)
        self.cur.execute(sql)
        Al_Type_lists = self.cur.fetchall()
        Al_Type_list = [datum[0] for datum in Al_Type_lists]

        return Al_Type_list

    def Trading_V_Result_Table_Add(self, Al_T, Code, B_D, B_T, B_C, B_V, S_D, S_T, S_C, S_V):

        dtime = datetime.datetime.now()
        B_Money = B_C * B_V
        S_Money = S_C * S_V

        R_Remain = S_Money - B_Money
        R_tax = (S_Money * (0.015 + 0.24) + B_Money * 0.015) / 100.0
        R_Profit = (R_Remain - R_tax)
        R_Profit_P = R_Profit / B_Money * 100

        raw_data = {'datetime': [dtime],
                    'Al_T': [Al_T],
                    'Code': [Code],
                    'B_D': [B_D],
                    'B_T': [B_T],
                    'B_C': [B_C],
                    'B_V': [B_V],
                    'B_Money': [B_Money],
                    'S_D': [S_D],
                    'S_T': [S_T],
                    'S_C': [S_C],
                    'S_V': [S_V],
                    'S_Money': [S_Money],
                    'R_Remain': [R_Remain],
                    'R_tax': [R_tax],
                    'R_Profit': [R_Profit],
                    'R_Profit_P': [R_Profit_P]
                    }

        df = pd.DataFrame(raw_data)
        df = df.round({'R_tax': 1, 'R_Profit': 1, 'R_Profit_P': 1})

        target_tbl_name = surf_trading_v_result_tbl_name
        df.to_sql(target_tbl_name, self.conn, if_exists='append', index=False)

        return

    def daily4_table_test(self, df):
        # print('daily4_table_test')
        target_tbl_name = 'surf_daily4_tbl'
        df.to_sql(target_tbl_name, self.conn, if_exists='append', index=False)
        return

    def daily5_table_test(self, df):
        print('daily5_table_test')
        target_tbl_name = 'surf_daily5_tbl'
        df.to_sql(target_tbl_name, self.conn, if_exists='append', index=False)
        return