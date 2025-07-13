import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import re
from collections import Counter
import os

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', text.lower())

# 各会议与对应URL模板
conference_config = {
    'neurips': {
        'years': list(range(2020, 2025)),
        'url_tpl': "https://dblp.org/db/conf/nips/neurips{}.html",
        'predict_years': [2025, 2026]
    },
    'aaai': {
        'years': list(range(2020, 2026)),
        'url_tpl': "https://dblp.org/db/conf/aaai/aaai{}.html",
        'predict_years': [2026]
    },
    'kdd': {
        'years': list(range(2020, 2025)),
        'url_tpl': "https://dblp.org/db/conf/kdd/kdd{}.html",
        'predict_years': [2025, 2026]
    }
}

def fetch_papers(conf_name, url_tpl, years):
    all_papers = []
    valid_years = []
    for year in years:
        url = url_tpl.format(year)
        print(f"📥 Fetching {conf_name.upper()} {year} from {url}")
        res = requests.get(url)
        if res.status_code != 200:
            print(f"⚠️ 请求失败 - HTTP {res.status_code}: {url}")
            continue
        soup = BeautifulSoup(res.text, 'html.parser')
        papers = soup.find_all('cite', class_='data')
        for p in papers:
            title_tag = p.find('span', class_='title')
            if not title_tag:
                continue
            title = title_tag.text.strip()
            authors = [a.text for a in p.find_all('span', itemprop='author')]
            link_tag = p.find('a', href=True)
            link = link_tag['href'] if link_tag else ''
            all_papers.append({
                'title': title,
                'authors': ', '.join(authors),
                'year': year,
                'conference': f"{conf_name.upper()} {year}",
                'conf': conf_name,
                'url': link
            })
        valid_years.append(year)
    return all_papers, valid_years

csv_file = "all_conference_papers_2020_2024.csv"

if os.path.exists(csv_file):
    print(f"✅ 文件 {csv_file} 存在，读取数据，跳过爬取")
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
else:
    print(f"📥 文件 {csv_file} 不存在，开始爬取数据")
    all_data = []
    valid_years_map = {}

    for conf, cfg in conference_config.items():
        data, valid_years = fetch_papers(conf, cfg['url_tpl'], cfg['years'])
        all_data.extend(data)
        valid_years_map[conf] = valid_years

    df = pd.DataFrame(all_data)
    if df.empty:
        print("❌ 未获取任何论文数据")
        exit()

    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

# 逐会议预测与可视化
for conf, cfg in conference_config.items():
    sub_df = df[df['conf'] == conf].copy()
    if sub_df.empty:
        continue
    sub_df['year'] = sub_df['year'].astype(int)
    counts = sub_df.groupby('year').size()
    X = np.array(counts.index).reshape(-1, 1)
    y = counts.values

    # 判断是否是AAAI，使用多项式回归
    if conf == 'aaai':
        degree = 2
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        # 预测未来年份
        X_pred_years = np.array(cfg['predict_years']).reshape(-1, 1)
        X_pred_poly = poly.transform(X_pred_years)
        preds = model.predict(X_pred_poly)

        # 曲线拟合数据
        X_curve = np.linspace(X.min(), max(cfg['predict_years']), 300).reshape(-1, 1)
        X_curve_poly = poly.transform(X_curve)
        y_curve = model.predict(X_curve_poly)
    else:
        # 使用线性回归
        model = LinearRegression().fit(X, y)
        X_pred_years = np.array(cfg['predict_years']).reshape(-1, 1)
        preds = model.predict(X_pred_years)

        # 曲线拟合数据
        X_curve = np.linspace(X.min(), max(cfg['predict_years']), 300).reshape(-1, 1)
        y_curve = model.predict(X_curve)

    # 输出预测结果
    for year, pred in zip(cfg['predict_years'], preds):
        print(f"🔮 预测 {conf.upper()} {year} 年论文数量约为：{int(pred)} 篇")

    # 可视化结果
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values, label='实际')
    plt.plot(cfg['predict_years'], preds, 'ro--', label='预测')
    plt.plot(X_curve, y_curve, 'g--', label='拟合曲线')  # 绿色虚线拟合曲线
    plt.title(f'{conf.upper()} 年度论文数量趋势')
    plt.xlabel('年份')
    plt.ylabel('论文数')
    plt.xticks(list(counts.index) + cfg['predict_years'])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{conf}_trend_prediction.png')
    plt.show()


# 生成每年统一词云（2020–2024）
for year in range(2020, 2025):
    titles = df[df['year'] == year]['title']
    if titles.empty:
        print(f"⚠️ {year} 没有可用标题，跳过词云生成。")
        continue
    words = ' '.join(titles.map(clean_text)).split()
    word_freq = Counter(words)
    if not word_freq:
        print(f"⚠️ {year} 词频统计为空，无法生成词云。")
        continue
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{year} 年研究热点词云')
    plt.tight_layout()
    plt.savefig(f'wordcloud_{year}.png')
    plt.show()
