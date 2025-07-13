import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import calendar
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.ensemble import RandomForestRegressor

# === 字体设置 ===
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# === 简化天气类型映射 ===
weather_mapping = {
    '晴': '晴天', '多云': '多云', '阴': '阴天',
    '雨': '雨天', '小雨': '雨天', '中雨': '雨天', '大雨': '雨天', '阵雨': '雨天', '雷阵雨': '雨天',
    '雪': '雪天', '小雪': '雪天', '中雪': '雪天', '大雪': '雪天'
}

def simplify_weather(w):
    if pd.isna(w): return '未知'
    for k, v in weather_mapping.items():
        if k in w: return v
    return '其他'

def extract_wind_level(wind):
    if pd.isna(wind): return np.nan
    if '微风' in wind: return 2
    match = pd.to_numeric(pd.Series(wind).str.extract(r'(\d+)')[0], errors='coerce')
    if not pd.isna(match).all(): return match.max()
    return np.nan

# === 风力分档函数（新） ===
def wind_level_to_bin(level):
    try:
        if pd.isna(level): return np.nan
        nums = list(map(int, str(level).replace('~', '-').split('-')))
        avg = sum(nums) / len(nums)
        if avg <= 3:
            return '0-3级'
        elif avg <= 5:
            return '3-5级'
        elif avg <= 7:
            return '5-7级'
        elif avg <= 10:
            return '7-10级'
        else:
            return '10级以上'
    except:
        return np.nan

# === Selenium 爬虫函数 ===
def get_weather_selenium(url, retries=3):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    for attempt in range(retries):
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        try:
            driver.get(url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "weather-table")))
            table = driver.find_element(By.CLASS_NAME, "weather-table")
            rows = table.find_elements(By.TAG_NAME, "tr")
            data = []

            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 4: continue
                date = cols[0].text.strip()
                weather = cols[1].text.strip()
                temp_high_elem = cols[2].find_elements(By.CLASS_NAME, "temp-high")
                temp_low_elem = cols[2].find_elements(By.CLASS_NAME, "temp-low")
                if not temp_high_elem or not temp_low_elem: continue
                temp_high = temp_high_elem[0].text.strip()
                temp_low = temp_low_elem[0].text.strip()
                wind = cols[3].text.strip()

                data.append({
                    "日期": date,
                    "天气状况": weather,
                    "最高气温": temp_high,
                    "最低气温": temp_low,
                    "风力风向": wind
                })
            driver.quit()
            return pd.DataFrame(data)
        except Exception as e:
            print(f"加载失败（{attempt+1}/{retries}）：{e}")
            driver.quit()
    return pd.DataFrame()

# === 数据预处理 ===
def preprocess_data(df):
    df['日期'] = pd.to_datetime(df['日期'], format='%Y年%m月%d日', errors='coerce')
    df['年份'] = df['日期'].dt.year
    df['月份'] = df['日期'].dt.month
    df['最高气温'] = df['最高气温'].str.extract(r'(-?\d+)').astype(float)
    df['最低气温'] = df['最低气温'].str.extract(r'(-?\d+)').astype(float)

    df[['白天天气', '夜晚天气']] = df['天气状况'].str.split('/', expand=True)
    df['白天天气'] = df['白天天气'].str.strip()
    df['夜晚天气'] = df['夜晚天气'].str.strip()
    df['白天天气类型'] = df['白天天气'].apply(simplify_weather)
    df['夜晚天气类型'] = df['夜晚天气'].apply(simplify_weather)

    df[['白天风力风向', '夜晚风力风向']] = df['风力风向'].str.split('/', expand=True)
    df['白天风力等级'] = df['白天风力风向'].str.extract(r'(\d+[-~]?\d*)级')
    df['夜晚风力等级'] = df['夜晚风力风向'].str.extract(r'(\d+[-~]?\d*)级')
    return df

# === 绘图函数 ===
def plot_temperature_trend(df):
    monthly_temp = df.groupby(['年份', '月份']).agg({'最高气温': 'mean', '最低气温': 'mean'}).reset_index()
    plt.figure(figsize=(14, 7))
    years = sorted(df['年份'].unique())
    colors = ['blue', 'green', 'red']

    for i, year in enumerate(years):
        ydata = monthly_temp[monthly_temp['年份'] == year]
        plt.plot(ydata['月份'], ydata['最高气温'], 'o-', color=colors[i], label=f'{year}年最高气温')
        plt.plot(ydata['月份'], ydata['最低气温'], 's--', color=colors[i], label=f'{year}年最低气温')

    plt.title('近三年每月平均气温变化趋势')
    plt.xlabel('月份')
    plt.ylabel('温度 (°C)')
    plt.xticks(range(1, 13), [calendar.month_name[m][:3] for m in range(1, 13)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('temperature_trend.png')
    plt.close()
    print("✅ 已生成气温变化趋势图")

def plot_day_night_weather(df):
    for col, label in [("白天天气类型", "白天"), ("夜晚天气类型", "夜晚")]:
        stat = df.groupby(['年份', '月份'])[col].value_counts().unstack(fill_value=0).reset_index()
        plt.figure(figsize=(16, 8))
        for i, year in enumerate(sorted(df['年份'].unique())):
            ax = plt.subplot(1, 3, i + 1)
            year_data = stat[stat['年份'] == year]
            bottom = np.zeros(len(year_data))
            for weather in ['晴天', '多云', '阴天', '雨天', '雪天', '其他']:
                if weather in year_data.columns:
                    ax.bar(year_data['月份'], year_data[weather], bottom=bottom, label=weather, alpha=0.7)
                    bottom += year_data[weather].values
            ax.set_title(f"{year}年{label}天气类型分布")
            ax.set_xlabel("月份"); ax.set_ylabel("天数")
            ax.set_xticks(range(1, 13))
            ax.grid(True, linestyle='--', alpha=0.5)
        plt.suptitle(f"{label}天气状况分布图")
        plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.1), ncol=6)
        plt.tight_layout()
        plt.savefig(f"{label}天气类型_分布图.png")
        plt.close()

def plot_day_night_wind(df):
    for col, label in [('白天风力等级', '白天'), ('夜晚风力等级', '夜晚')]:
        df_cut = df.copy()
        df_cut['风力分档'] = df_cut[col].apply(wind_level_to_bin)
        df_cut = df_cut.dropna(subset=['风力分档'])
        wind_stat = df_cut.groupby(['年份', '月份', '风力分档']).size().unstack(fill_value=0).reset_index()
        bins_order = ['0-3级', '3-5级', '5-7级', '7-10级', '10级以上']
        wind_stat = wind_stat[['年份', '月份'] + [b for b in bins_order if b in wind_stat.columns]]

        plt.figure(figsize=(16, 8))
        for i, year in enumerate(sorted(df['年份'].unique())):
            ax = plt.subplot(1, 3, i + 1)
            year_data = wind_stat[wind_stat['年份'] == year]
            bottom = np.zeros(len(year_data))
            for b in bins_order:
                if b in year_data.columns:
                    ax.bar(year_data['月份'], year_data[b], bottom=bottom, label=b, alpha=0.75)
                    bottom += year_data[b].values
            ax.set_title(f"{year}年{label}风力等级分布")
            ax.set_xlabel("月份"); ax.set_ylabel("天数")
            ax.set_xticks(range(1, 13))
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f"{label}风力等级分布图")
        plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.1), ncol=5)
        plt.tight_layout()
        plt.savefig(f"{label}风力等级_分布图.png")
        plt.close()
        print(f"✅ 已生成 {label} 风力图")

def predict_temperature_rf(df):
    monthly_avg = df.groupby(['年份', '月份'])['最高气温'].mean().reset_index()
    X_train = monthly_avg[(monthly_avg['年份'] < 2025)][['年份', '月份']]
    y_train = monthly_avg[(monthly_avg['年份'] < 2025)]['最高气温']
    X_pred = pd.DataFrame({'年份': [2025]*6, '月份': list(range(1, 7))})

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_pred)
    return X_pred, y_pred

def crawl_real_2025_data():
    real_data = []
    for m in range(1, 7):
        url = f"https://www.tianqihoubao.com/lishi/dalian/month/2025{m:02d}.html"
        print(f"抓取：{url}")
        df = get_weather_selenium(url)
        if not df.empty:
            df = preprocess_data(df)
            real_data.append(df)
        time.sleep(1)
    if real_data:
        return pd.concat(real_data, ignore_index=True)
    return pd.DataFrame()

def plot_prediction_vs_real(X_pred, y_pred, real_df):
    real_monthly = real_df.groupby(['月份'])['最高气温'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(X_pred['月份'], y_pred, 'r-o', label='预测最高气温（RF）')
    plt.plot(real_monthly['月份'], real_monthly['最高气温'], 'g--s', label='真实最高气温')
    plt.fill_between(X_pred['月份'], y_pred, real_monthly['最高气温'], color='orange', alpha=0.1)
    plt.xlabel('月份')
    plt.ylabel('气温（°C）')
    plt.title('2025年1-6月 预测与真实最高气温对比')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("2025预测_vs_真实_最高气温.png")
    plt.close()
    print("✅ 已绘制 2025年预测与真实最高气温对比图")

# === 主程序 ===
if __name__ == "__main__":
    try:
        df = pd.read_csv("大连2022-2024天气.csv", encoding="utf-8-sig")
        print("✅ 已加载本地数据文件")
    except FileNotFoundError:
        print("⚠️ 未找到本地文件，开始爬取数据")
        all_data = []
        for year in range(2022, 2025):
            for month in range(1, 13):
                url = f"https://www.tianqihoubao.com/lishi/dalian/month/{year}{month:02d}.html"
                print(f"抓取中：{year}-{month:02d}")
                df_month = get_weather_selenium(url)
                if not df_month.empty:
                    all_data.append(df_month)
                time.sleep(1)
        df = pd.concat(all_data, ignore_index=True)
        df.to_csv("大连2022-2024天气.csv", index=False, encoding="utf-8-sig")
        print("✅ 已保存爬取数据")

    processed_df = preprocess_data(df)
    plot_temperature_trend(processed_df)
    plot_day_night_weather(processed_df)
    plot_day_night_wind(processed_df)
    X_pred, y_pred = predict_temperature_rf(processed_df)
    real_2025_df = crawl_real_2025_data()
    if not real_2025_df.empty:
        plot_prediction_vs_real(X_pred, y_pred, real_2025_df)
    else:
        print("❌ 无法获取2025年真实数据")
