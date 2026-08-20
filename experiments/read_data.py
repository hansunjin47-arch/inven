import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

import pandas as pd
import numpy as np
import datetime
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt # 시각화를 위해 추가
import pickle

# 파일 이름 리스트 (실제 파일이 같은 폴더에 있다고 가정)
file_names = ['P_hnb_2015', 'P_hnb_2016', 'P_hnb_2017', 'P_hnb_2018']

df_list = []
for file_name in file_names:
    try:
        df_temp = pd.read_csv(
            file_name,
            delimiter='|',
            encoding='cp949', # 파일 인코딩 (필요에 따라 'euc-kr' 등으로 변경)
            dtype={
                'GOODS_CD': str,
                'BIZPL_CD': str,
                'TRANS_STK_QTY': float,
                'BUY_QTY': float,
                'SHIP_QTY': float,
                'MV_QTY': float,
                'SALE_QTY': float, # float으로 타입 지정 확인
                'SALES_AMT': float,
                'SALES_VAT_AMT': float,
                'STK_QTY': float,
                'LINE_CD': str,
                'LINE_NM': str,
                'CLASS_NM': str
            },
            parse_dates=['STK_EVAL_DT']
        )
        df_list.append(df_temp)
    except FileNotFoundError:
        print(f"Warning: {file_name} not found. Skipping.")
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# 모든 DataFrame 합치기
if df_list:
    df = pd.concat(df_list, ignore_index=True)
else:
    print("No data frames were loaded. Please check file paths and encoding. Exiting.")
    exit()

# 정렬
df_ordered = df.sort_values(by=['BIZPL_CD', 'GOODS_CD', 'STK_EVAL_DT'])
df_change = df_ordered

#선크림
target_good1 = df_change.loc[
    (df_change['GOODS_CD'] == "8806133612347") &
    (df_change['BIZPL_CD'] == "W0090") & # W0090
    (df_change['STK_EVAL_DT'] >= datetime.datetime(2015, 1, 1)) &
    (df_change['STK_EVAL_DT'] <= datetime.datetime(2017, 1, 1))
]

# --- 2. 수요 시퀀스 추출 및 저장 ---
demand_data_2015_good1 = target_good1.loc[
    (target_good1['STK_EVAL_DT'] >= datetime.datetime(2015, 1, 1)) &
    (target_good1['STK_EVAL_DT'] <= datetime.datetime(2015, 12, 31))
]
demand_data_2015_good1 = demand_data_2015_good1.set_index('STK_EVAL_DT').sort_index()
daily_demand_2015_good1 = demand_data_2015_good1['SALE_QTY'].resample('D').sum().fillna(0)

# 수정된 부분: 음수와 이상치(>7)를 처리
daily_demand_2015_good1.loc[daily_demand_2015_good1 < 0] = 0
daily_demand_2015_good1.loc[daily_demand_2015_good1 > 10] = 0

fixed_seq_2015_good1 = daily_demand_2015_good1.values # NumPy 배열로 변환

# fixed_seq를 pickle 파일로 저장
pickle_file_path = 'data/daily_demand_2015_good1.pkl'
with open(pickle_file_path, 'wb') as f: # 'wb'는 쓰기 바이너리 모드
    pickle.dump(fixed_seq_2015_good1, f)

# 2015년 데이터 플롯
plt.figure(figsize=(12, 4))
plt.plot(fixed_seq_2015_good1, label='Demand(training period)')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.ylim(0, 10)   # ★ y축 고정
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("demand_plot_2015_good1.pdf")
plt.close()

# --- 2. 수요 시퀀스 추출 및 저장 ---
demand_data_2016_good1 = target_good1.loc[
    (target_good1['STK_EVAL_DT'] >= datetime.datetime(2016, 1, 1)) &
    (target_good1['STK_EVAL_DT'] <= datetime.datetime(2016, 12, 31))
]
demand_data_2016_good1 = demand_data_2016_good1.set_index('STK_EVAL_DT').sort_index()
daily_demand_2016_good1 = demand_data_2016_good1['SALE_QTY'].resample('D').sum().fillna(0)

# 수정된 부분: 음수와 이상치(>7)를 처리
daily_demand_2016_good1.loc[daily_demand_2016_good1 < 0] = 0
daily_demand_2016_good1.loc[daily_demand_2016_good1 > 10] = 10

fixed_seq_2015_good1 = daily_demand_2015_good1.values # NumPy 배열로 변환

fixed_seq_2016_good1 = daily_demand_2016_good1.values # NumPy 배열로 변환

# fixed_seq를 pickle 파일로 저장
pickle_file_path = 'data/daily_demand_2016_good1.pkl'
with open(pickle_file_path, 'wb') as f: # 'wb'는 쓰기 바이너리 모드
    pickle.dump(fixed_seq_2016_good1, f)

# 2016년 데이터 플롯
plt.figure(figsize=(12, 4))
plt.plot(fixed_seq_2016_good1, label='Demand(test period)')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.ylim(0, 10)   # ★ y축 고정
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("demand_plot_2016_good1.pdf")
plt.close()

#데오드란트
target_good2 = df_change.loc[
    (df_change['GOODS_CD'] == "4005808321735") &
    (df_change['BIZPL_CD'] == "W0090") & 
    (df_change['STK_EVAL_DT'] >= datetime.datetime(2015, 3, 1)) &
    (df_change['STK_EVAL_DT'] <= datetime.datetime(2018, 3, 1))
]

# --- 2. 수요 시퀀스 추출 및 저장 ---
demand_data_2015_good2 = target_good2.loc[
    (target_good2['STK_EVAL_DT'] >= datetime.datetime(2015, 3, 1)) &
    (target_good2['STK_EVAL_DT'] < datetime.datetime(2016, 3, 1))
    ]
demand_data_2015_good2 = demand_data_2015_good2.set_index('STK_EVAL_DT').sort_index()
daily_demand_2015_good2 = demand_data_2015_good2['SALE_QTY'].resample('D').sum().fillna(0)

fixed_seq_2015_good2 = daily_demand_2015_good2.values  # NumPy 배열로 변환

# 2015년 good 2데이터 플롯
plt.figure(figsize=(12, 4))
plt.plot(fixed_seq_2015_good2, label='Demand Sequence 2015 good 2')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.title('Demand Sequence 2015 good2')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("demand_plot_2015_good2.png")
plt.close()

# fixed_seq를 pickle 파일로 저장
pickle_file_path = './daily_demand_2015_good2.pkl'
with open(pickle_file_path, 'wb') as f:  # 'wb'는 쓰기 바이너리 모드
    pickle.dump(fixed_seq_2015_good2, f)

# --- 2. 수요 시퀀스 추출 및 저장 ---
demand_data_2016_good2 = target_good2.loc[
    (target_good2['STK_EVAL_DT'] >= datetime.datetime(2016, 3, 1)) &
    (target_good2['STK_EVAL_DT'] < datetime.datetime(2017, 3, 1))
    ]
demand_data_2016_good2 = demand_data_2016_good2.set_index('STK_EVAL_DT').sort_index()
daily_demand_2016_good2 = demand_data_2016_good2['SALE_QTY'].resample('D').sum().fillna(0)
fixed_seq_2016_good2 = daily_demand_2016_good2.values  # NumPy 배열로 변환

# fixed_seq를 pickle 파일로 저장
pickle_file_path = './daily_demand_2016_good2.pkl'
with open(pickle_file_path, 'wb') as f:  # 'wb'는 쓰기 바이너리 모드
    pickle.dump(fixed_seq_2016_good2, f)

# 2015년 good 2데이터 플롯
plt.figure(figsize=(12, 4))
plt.plot(fixed_seq_2016_good2, label='Demand Sequence 2016 good2')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.title('Demand Sequence 2016 good2')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("demand_plot_2016_good2.png")
plt.close()

# --- 2. 수요 시퀀스 추출 및 저장 ---
demand_data_2017_good2 = target_good2.loc[
    (target_good2['STK_EVAL_DT'] >= datetime.datetime(2017, 3, 1)) &
    (target_good2['STK_EVAL_DT'] < datetime.datetime(2018, 3, 1))
    ]
demand_data_2017_good2 = demand_data_2017_good2.set_index('STK_EVAL_DT').sort_index()
daily_demand_2017_good2 = demand_data_2017_good2['SALE_QTY'].resample('D').sum().fillna(0)

fixed_seq_2017_good2 = daily_demand_2017_good2.values  # NumPy 배열로 변환

# fixed_seq를 pickle 파일로 저장
pickle_file_path = './daily_demand_2017_good2.pkl'
with open(pickle_file_path, 'wb') as f:  # 'wb'는 쓰기 바이너리 모드
    pickle.dump(fixed_seq_2017_good2, f)

# 2015년 good 2데이터 플롯
plt.figure(figsize=(12, 4))
plt.plot(fixed_seq_2017_good2, label='Demand Sequence 2017 good2')
plt.xlabel('Day')
plt.ylabel('Demand')
plt.title('Demand Sequence 2017 good2')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("demand_plot_2017_good2.png")
plt.close()
