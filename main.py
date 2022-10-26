from flask import Flask, send_file, render_template, make_response

from io import BytesIO, StringIO
import pandas as pd
import numpy as np
import time
from functools import wraps, update_wrapper
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

global avg_value
avg_value = [10, 50] #SNR 평균값 [min, max] ===================================================================================================

global r_a #right ascension
r_a = []
global dec #declination
dec = []


sun_r = 12 #right ascension of sun
sun_d = 180 #declination of sun
int_r = 10 #interval of right ascension
int_d = 10 #interval of decliantion


sun_r = sun_r - int_r/2
sun_d = sun_d - int_d/2

for i in range(-4, 6):
    r_ = sun_r + i * int_r
    d_ = sun_d + i * int_d

    r_a.append(str(r_)+"h")
    dec.append(str(d_) + "deg")




def make_list(df):
    df_timestamp = df["Timestamp"].to_numpy()  # time만 빼와서 array 형성
    df_SNR = df["SNR"].to_numpy()
    df_timestamp_non = []
    df_timestamp_fi = []
    df_timestamp_hour = []  # 단위 h

    for _ in df_timestamp:
        df_timestamp_non.append(_.split()[1])  # 스페이스를 기준으로 오른쪽의 시간부분만 빼와서 list형성

    for __ in df_timestamp_non:
        list_ = []
        ___ = __.split(':')  # 시간 문자열에서 :을 기준으로 나누기
        for i in ___:
            list_.append(float(i))  # 문자열을 정수형 데이터로
        df_timestamp_fi.append(list_)  # [시간, 분, 초]를 각 데이터로 가지는 최종 데이터 생성

    for j in df_timestamp_fi:
        hour_ = j[0] + j[1] / 60 + j[2] / 3600  # 단위 h로 통일
        df_timestamp_hour.append(hour_)

    return [df_timestamp_hour, df_SNR, df_timestamp_fi]

def make_list_live(df):
    df_timestamp = df["Timestamp"].to_numpy()  # time만 빼와서 array 형성
    df_SNR = df["SNR"].to_numpy()
    df_timestamp_non = []
    df_timestamp_fi = []
    df_timestamp_hour = []  # 단위 h

    for _ in df_timestamp:
        df_timestamp_non.append(_.split()[1])  # 스페이스를 기준으로 오른쪽의 시간부분만 빼와서 list형성

    for __ in df_timestamp_non:
        list_ = []
        ___ = __.split(':')  # 시간 문자열에서 :을 기준으로 나누기
        for i in ___:
            list_.append(float(i))  # 문자열을 정수형 데이터로
        df_timestamp_fi.append(list_)  # [시간, 분, 초]를 각 데이터로 가지는 최종 데이터 생성

    for j in df_timestamp_fi:
        hour_ = j[0] + j[1] / 60 + j[2] / 3600  # 단위 h로 통일
        df_timestamp_hour.append(hour_)

    return [df_timestamp_hour[-10:], df_SNR[-10:], df_timestamp_fi[-10:]]

def make_timelim(data_list):
    time_lim = []
    for i in data_list:
        for j in i[2]:
            if j[0] not in time_lim:
                time_lim.append(j[0])

    return time_lim


def make_graph(data_list, time_lim, data_name, graph_name):
    for i in enumerate(data_list):
        plt.plot(i[1][0], i[1][1], label=data_name[i[0]])
    plt.xticks(time_lim, time_lim, fontsize=7, color='gray')
    plt.tick_params(axis='y', labelcolor='gray')
    plt.legend()
    plt.title(graph_name, fontsize=10)
    plt.ylabel("SNR[dB]", fontsize=8)

def data_preprocessing(df):
    global r_a
    global dec

    data_dict = {"right ascension": [], "declination": [], "value": []}  # 데이터 형식을 변형하기 위한 dictionary

    df_value = df["SNR"].to_numpy()  # heatmap pixel value
    df_value = df_value[:100]


    # 데이터 파일에서 value indexing해서 dict에 추가하기
    for index, value in enumerate(df_value):
        index_ = index % 20
        index__ = index // 10

        if index_ >= 10:
            _ = index_-10
            data_dict["right ascension"].append(r_a[_])  # 적경 추가
            data_dict["declination"].append(dec[index__])  # 적위 추가
            data_dict["value"].append(value)  # 그 때의 value 추가
        else:
            _= (index_+1)*(-1)
            data_dict["right ascension"].append(r_a[_])  # 적경 추가
            data_dict["declination"].append(dec[index__])  # 적위 추가
            data_dict["value"].append(value)  # 그 때의 value 추가

    return data_dict


def make_heatmap(pivot_data):
    plt.pcolor(pivot_data)
    plt.xticks(np.arange(0.5, len(pivot_data.columns), 1), pivot_data.columns, fontsize=7, rotation=45)
    plt.yticks(np.arange(0.5, len(pivot_data.index), 1), pivot_data.index, fontsize=7)
    plt.title('SUN SNR heatmap', fontsize=20)
    plt.xlabel('Right Ascension', fontsize=14)
    plt.ylabel('Declinaiton', fontsize=14)
    plt.colorbar()











@app.route("/")
def index():
    return render_template('index.html')

@app.route("/window1")
def window1():
    global avg_value

    _ = 0

    with open("1.txt") as f:
        content_1 = f.read().split('\n')

    df = pd.read_csv("SUN.csv")  # 실시간으로 갱신되는 csv 파일명 적기=========================================================================================
    # data preprocessing
    df_ = make_list_live(df)[1]

    for i in df_:
        if (i < avg_value[0]) or (i > avg_value[1]):
            _ = 1
            print("\a")
    return render_template('window1.html', content = content_1, TF = _)


#A의 비실시간 그래프
@app.route('/fig1')
def fig1():
    time.sleep(0.1) #그래프 충돌시 값 키우기=========================================================================================

    # data파일 불러오기
    df1 = pd.read_csv("SUN.csv")
    df2 = pd.read_csv("BACK.csv")

    # 이름 리스트화
    data_name_1 = ["SUN"]
    data_name_2 = ["BACK"]
    data_name_3 = ["SUN", "BACK"]

    # data preprocessing
    df_1 = make_list(df1)
    df_2 = make_list(df2)

    # 시간 범위 설정
    time_lim_1 = make_timelim([df_1])
    time_lim_2 = make_timelim([df_2])
    time_lim_3 = make_timelim([df_1, df_2])

    # 그래프 그리기
    plt.figure(figsize=(11, 8.5))

    plt.subplot(3, 1, 1)
    make_graph([df_1], time_lim_1, data_name_1, "SUN SNR")

    plt.subplot(3, 1, 2)
    make_graph([df_2], time_lim_2, data_name_2, "BACK SNR")

    plt.subplot(3, 1, 3)
    make_graph([df_1, df_2], time_lim_3, data_name_3, "SUN SNR, BACK SNR")

    plt.xlabel("time[h]", fontsize=8)
    ## file로 저장하는 것이 아니라 binary object에 저장해서 그대로 file을 넘겨준다고 생각하면 됨
    ## binary object에 값을 저장한다.
    ## svg로 저장할 수도 있으나, 이 경우 html에서 다른 방식으로 저장해줘야 하기 때문에 일단은 png로 저장해줌
    img = BytesIO()
    plt.savefig(img, format='png', dpi=500)
    ## object를 읽었기 때문에 처음으로 돌아가줌
    img.seek(0)
    return send_file(img, mimetype='image/png')

#A의 실시간 그래프
@app.route('/fig2')
def fig2():
    df = pd.read_csv("SUN.csv") #실시간으로 갱신되는 csv 파일명 적기=========================================================================================
    # 이름 리스트화
    data_name = ["name"] #그래프 이름 적기=================================================================================================
    # data preprocessing
    df_ = make_list_live(df)
    # 시간 범위 설정
    time_lim = make_timelim([df_])
    # 그래프 그리기
    plt.figure(figsize= (11, 6.5))
    make_graph([df_], time_lim, data_name, "SUN SNR in last 10s") #그래프 자체 사진의 이름=========================================================================================
    plt.xlabel("time[s]")
    ## file로 저장하는 것이 아니라 binary object에 저장해서 그대로 file을 넘겨준다고 생각하면 됨
    ## binary object에 값을 저장한다.
    ## svg로 저장할 수도 있으나, 이 경우 html에서 다른 방식으로 저장해줘야 하기 때문에 일단은 png로 저장해줌
    img = BytesIO()
    plt.savefig(img, format='png', dpi=500)
    ## object를 읽었기 때문에 처음으로 돌아가줌
    img.seek(0)
    return send_file(img, mimetype='image/png')



@app.route("/window2")
def window2():
    with open("2.txt") as f:
        content_2 = f.read().split('\n')

    return render_template('window2.html', content = content_2)

#B의 히트맵 그래프
@app.route('/fig3')
def fig3():
    df = pd.read_csv("SUN.csv") #처리하기 힘든 형식의 데이터 불러오기

    data_dict = data_preprocessing(df) #pivot 형태로 변환하기 위해 데이터 전처리

    plt.figure(figsize=(10, 8.5))

    df_preprocessing = pd.DataFrame(data_dict)
    df_preprocessing = df_preprocessing.pivot("declination", "right ascension", "value")  # data 피벗 테이블로 변형


    make_heatmap(df_preprocessing)


    ## file로 저장하는 것이 아니라 binary object에 저장해서 그대로 file을 넘겨준다고 생각하면 됨
    ## binary object에 값을 저장한다.
    ## svg로 저장할 수도 있으나, 이 경우 html에서 다른 방식으로 저장해줘야 하기 때문에 일단은 png로 저장해줌
    img = BytesIO()
    plt.savefig(img, format='png', dpi=200)
    ## object를 읽었기 때문에 처음으로 돌아가줌
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.after_request
def set_response_headers(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    return r


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=9900)
