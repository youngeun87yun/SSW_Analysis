import sqlite3 as lite
from CSurfDb import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar

def quartic_curve_fit(x, y):
    """
    4차 곡선을 적합하고 계수를 반환합니다.

    :param x: x 값 배열
    :param y: y 값 배열
    :return: 4차 곡선의 계수 (a, b, c, d, e)
    """

    # 4차 다항식 함수 정의
    def quartic_func(x, a, b, c, d, e):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    # curve_fit을 사용하여 데이터에 4차 곡선 적합
    popt, _ = curve_fit(quartic_func, x, y)

    return popt


def calculate_error(original_y, fitted_y):
    """
    원본 데이터와 적합된 곡선 간의 오차율을 계산합니다.

    :param original_y: 원본 데이터의 y 값 배열
    :param fitted_y: 적합된 곡선의 y 값 배열
    :return: 오차율 (RMSE)
    """
    # Root Mean Square Error (RMSE) 계산
    rmse = np.sqrt(np.mean(((original_y - fitted_y)/original_y) ** 2))


    return rmse


def plot_coefficient(coefficient_name, coefficient_values):
    """
    주어진 곡선의 계수를 그래프로 그립니다.

    :param coefficient_name: 계수 이름 (예: 'a', 'b', 'c', 'd', 'e', 'f')
    :param coefficient_values: 계수 값들의 리스트
    """
    plt.figure(figsize=(10, 6))
    plt.plot(coefficient_values)
    plt.title(f'{coefficient_name} Coefficient')
    plt.xlabel('Index')
    plt.ylabel('Coefficient Value')
    plt.grid(True)
    plt.show()


def quartic_inflection_point(coefficients):
    """
    4차 함수의 변곡점 위치를 찾습니다.

    :param coefficients: 4차 다항식의 계수 (a, b, c, d, e)
    :return: 변곡점의 x 값
    """

    # 4차 다항식 함수 정의
    def quartic_func(x, a, b, c, d, e):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    # 4차 다항식의 미분 함수 정의
    def quartic_derivative(x, a, b, c, d, e):
        return 4 * a * x ** 3 + 3 * b * x ** 2 + 2 * c * x + d

    # 4차 다항식의 미분의 극소점을 찾습니다.
    result = minimize_scalar(quartic_derivative, args=coefficients)

    return result.x

def find_last_inflection_point(x, coefficients):
    """
    마지막 변곡점의 x 값을 찾습니다.

    :param x: x 값 배열
    :param coefficients: 4차 곡선의 계수 (a, b, c, d, e)
    :return: 마지막 변곡점의 x 값
    """

    # 4차 다항식 함수 정의
    def quartic_func(x, a, b, c, d, e):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    # 변곡점을 찾기 위해 4차 미분의 계수를 구합니다.
    # 변곡점은 4차 미분의 극소점이 됩니다.
    derivative_coefficients = np.polyder(coefficients, m=4)

    # 4차 미분 함수
    def fourth_derivative(x):
        return np.polyval(derivative_coefficients, x)

    # 4차 미분의 극소점을 찾습니다.
    inflection_points = np.roots(fourth_derivative)

    # 마지막 변곡점을 찾습니다.
    last_inflection_point = np.max(inflection_points.real)

    return last_inflection_point

def distance_to_last_inflection_point(x, coefficients):
    """
    마지막 변곡점으로부터 현재까지의 거리를 계산합니다.

    :param x: 현재 x 값
    :param coefficients: 4차 곡선의 계수 (a, b, c, d, e)
    :return: 마지막 변곡점으로부터 현재까지의 거리
    """

    # 4차 다항식 함수 정의
    def quartic_func(x, a, b, c, d, e):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    # 변곡점을 찾기 위해 4차 미분의 계수를 구합니다.
    # 변곡점은 4차 미분의 극소점이 됩니다.
    derivative_coefficients = np.polyder(coefficients, m=4)

    # 4차 미분 함수
    def fourth_derivative(x):
        return np.polyval(derivative_coefficients, x)

    # 4차 미분의 극소점을 찾습니다.
    inflection_points = np.roots(fourth_derivative)

    # 마지막 변곡점을 찾습니다.
    last_inflection_point = np.max(inflection_points.real)

    # 현재 x 값과 마지막 변곡점과의 거리를 구합니다.
    distance = np.abs(x - last_inflection_point)

    return distance


def find_last_inflection_point(x, coefficients):
    """
    마지막 변곡점의 x 값을 찾습니다.

    :param x: x 값 배열
    :param coefficients: 4차 곡선의 계수 (a, b, c, d, e)
    :return: 마지막 변곡점의 x 값
    """

    # 4차 다항식 함수 정의
    def quartic_func(x, a, b, c, d, e):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    # 변곡점을 찾기 위해 4차 미분의 계수를 구합니다.
    # 변곡점은 4차 미분의 극소점이 됩니다.
    derivative_coefficients = np.polyder(coefficients, m=4)

    # 4차 미분 함수
    def fourth_derivative(x):
        return np.polyval(derivative_coefficients, x)

    # 4차 미분의 극소점을 찾습니다.
    inflection_points = np.roots(derivative_coefficients)

    # 변곡점이 없는 경우 처리
    if len(inflection_points) == 0:
        return None

    # 마지막 변곡점을 찾습니다.
    last_inflection_point = np.max(inflection_points.real)

    return last_inflection_point


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    objMyDB = CSurfDB()

    # 1. DB의 모든 코드를 구하자
    # 1. 전체 종목 조회
    # code_list = objMyDB.CodeList_Get()
    code_list = ['A005880', 'A382900', 'A012860']

    n = 100
    x_data = list(range(1, n + 1))

    # 데이터프레임을 초기화합니다.
    df = pd.DataFrame(columns=['Code', 'Date', 'Close', 'MAX', 'A', 'B', 'C', 'D', 'Error'])

    for code in code_list:

        # 2. DB 에서 code 의 모든 데이터를 가져온다.
        dates, opens, highs, lows, closes, vols, times = objMyDB.GetDailyData_all(code)

        nLen = len(closes)
        print(code, nLen)

        for index in range(nLen-n-5):

            y_data = closes[index:index+n]

            # 4차 함수의 계수를 구합니다.
            coeffs = np.polyfit(x_data, y_data, 4)

            last_close = y_data[-1]

            # 100 이후 5개의 데이터 중에서 가장 큰 값을 구합니다.
            max_after_100 = np.max(closes[index+100:index+105])

            # 데이터를 4차 함수로 fitting 합니다.
            fitted_values = np.polyval(coeffs, range(n))

            # 오차율을 계산합니다.
            error = np.mean(np.abs((y_data - fitted_values) / y_data)) * 100

            # 결과를 데이터프레임에 추가합니다.
            df.loc[len(df)] = [code, dates[index], last_close, max_after_100, coeffs[0], coeffs[1], coeffs[2], coeffs[3], error]

    # 결과 출력
    print(df)
    objMyDB.daily5_table_test(df)