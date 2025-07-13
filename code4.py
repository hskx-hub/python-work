import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False   # 负号正常显示

# ===================== 大乐透数据处理 =====================

def fetch_dlt_before_july1(limit=200, save_path="dlt_before_july1.csv"):
    url = f"https://datachart.500.com/dlt/history/newinc/history.php?limit={limit}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://datachart.500.com/dlt/history.shtml",
        "Accept-Language": "zh-CN,zh;q=0.9"
    }
    try:
        print("🔄 正在请求大乐透历史数据...")
        response = requests.get(url, headers=headers, timeout=20)
        response.encoding = "gb2312"
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("tbody tr.t_tr1")

        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 15:
                issue = cols[0].text.strip()
                reds = [cols[i].text.strip() for i in range(1, 6)]
                blues = [cols[6].text.strip(), cols[7].text.strip()]
                jackpot = cols[8].text.strip()
                first_prize_count = cols[9].text.strip()
                first_prize_money = cols[10].text.strip()
                second_prize_count = cols[11].text.strip()
                second_prize_money = cols[12].text.strip()
                total_sales = cols[13].text.strip().replace(',', '').replace('元', '')
                draw_date = cols[14].text.strip()
                data.append([
                    issue, *reds, *blues, jackpot,
                    first_prize_count, first_prize_money,
                    second_prize_count, second_prize_money,
                    total_sales, draw_date
                ])

        columns = [
            "期号", "红1", "红2", "红3", "红4", "红5",
            "蓝1", "蓝2", "奖池奖金(元)",
            "一等奖注数", "一等奖奖金(元)",
            "二等奖注数", "二等奖奖金(元)",
            "总投注额(元)", "开奖日期"
        ]
        df = pd.DataFrame(data, columns=columns)

        cutoff_date = datetime(2025, 7, 1)
        df["开奖日期"] = pd.to_datetime(df["开奖日期"], format="%Y-%m-%d", errors="coerce")
        df["总投注额(元)"] = pd.to_numeric(df["总投注额(元)"], errors="coerce")
        df = df[df["开奖日期"] < cutoff_date]
        df = df.sort_values("开奖日期", ascending=False).head(100)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"✅ 成功保存大乐透数据到 {save_path}")
        return df
    except Exception as e:
        print(f"❌ 抓取大乐透数据出错：{e}")
        return None

def analyze_and_visualize(df):
    print("\n📊 开始分析大乐透数据...")
    df = df.sort_values("开奖日期")
    plt.figure(figsize=(10, 6))
    plt.plot(df["开奖日期"], df["总投注额(元)"], marker='o')
    plt.title("大乐透总销售额随开奖日期变化")
    plt.xlabel("开奖日期")
    plt.ylabel("总销售额")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 添加期数列
    df["期数"] = np.arange(len(df))

    # 设置特征和目标
    X = df[["期数"]]
    y = df["总投注额(元)"]

    # 生成多项式特征（如 2 次或 3 次）
    poly = PolynomialFeatures(degree=2, include_bias=False)  # 二次多项式
    X_poly = poly.fit_transform(X)

    # 拟合多项式回归模型
    model = LinearRegression().fit(X_poly, y)

    # 预测下一期的期数
    next_period = np.array([[df["期数"].max() + 1]])
    next_period_poly = poly.transform(next_period)
    predicted_sales = model.predict(next_period_poly)

    print(f"🔮 [多项式回归] 预测下一期总销售额：{int(predicted_sales[0])} 元")

    front_numbers = []
    back_numbers = []
    for _, row in df.iterrows():
        front_numbers += [int(row[f"红{i}"]) for i in range(1, 6)]
        back_numbers += [int(row["蓝1"]), int(row["蓝2"])]

    front_counts = Counter(front_numbers)
    plt.figure(figsize=(12, 5))
    plt.bar(front_counts.keys(), front_counts.values())
    plt.title("前区号码出现频率")
    plt.xlabel("号码")
    plt.ylabel("次数")
    plt.show()

    back_counts = Counter(back_numbers)
    plt.figure(figsize=(8, 4))
    plt.bar(back_counts.keys(), back_counts.values(), color='orange')
    plt.title("后区号码出现频率")
    plt.xlabel("号码")
    plt.ylabel("次数")
    plt.show()

    front_recommend = [num for num, _ in front_counts.most_common(5)]
    back_recommend = [num for num, _ in back_counts.most_common(2)]
    print(f"🎯 推荐投注号码：前区 {front_recommend}，后区 {back_recommend}")

def analyze_draw_days(df):
    print("\n📅 分析开奖日模式...")

    df["星期"] = df["开奖日期"].dt.dayofweek
    df["星期中文"] = df["星期"].map({0: "周一", 2: "周三", 5: "周六"})
    filtered = df[df["星期"].isin([0, 2, 5])]
    grouped = filtered.groupby("星期中文")

    all_front_counts = {}
    sales_stats = {}

    for name, group in grouped:
        print(f"\n{name}：开奖次数 {len(group)}，平均投注额：{int(group['总投注额(元)'].mean())} 元")

        nums = []
        for _, row in group.iterrows():
            nums += [int(row[f"红{i}"]) for i in range(1, 6)]
        counts = Counter(nums)
        all_front_counts[name] = counts
        sales_stats[name] = {
            "平均销售额": group["总投注额(元)"].mean(),
            "销售额标准差": group["总投注额(元)"].std()
        }

        plt.figure(figsize=(8, 4))
        plt.bar(counts.keys(), counts.values())
        plt.title(f"{name} 前区号码分布")
        plt.xlabel("号码")
        plt.ylabel("频率")
        plt.show()

    print("\n📊 不同开奖日号码分布对比（前区）")
    all_keys = sorted(set(k for d in all_front_counts.values() for k in d))
    width = 0.25
    plt.figure(figsize=(12, 6))
    for i, (name, counts) in enumerate(all_front_counts.items()):
        values = [counts.get(k, 0) for k in all_keys]
        plt.bar([x + i * width for x in range(len(all_keys))], values, width=width, label=name)
    plt.xticks([x + width for x in range(len(all_keys))], all_keys)
    plt.legend()
    plt.title("不同开奖日的前区号码分布对比")
    plt.xlabel("号码")
    plt.ylabel("频率")
    plt.tight_layout()
    plt.show()

    print("\n📊 不同开奖日总销售额对比柱状图")
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(sales_stats.keys()), y=[v["平均销售额"] for v in sales_stats.values()], palette="Set2")
    plt.title("不同开奖日平均销售额")
    plt.xlabel("星期")
    plt.ylabel("平均销售额")
    plt.show()

    print("\n🔵 后区号码分布对比（按开奖日）")
    all_back_counts = {}
    for name, group in grouped:
        back_nums = []
        for _, row in group.iterrows():
            back_nums += [int(row["蓝1"]), int(row["蓝2"])]
        counts = Counter(back_nums)
        all_back_counts[name] = counts

    all_back_keys = sorted(set(k for d in all_back_counts.values() for k in d))
    plt.figure(figsize=(10, 5))
    for i, (name, counts) in enumerate(all_back_counts.items()):
        values = [counts.get(k, 0) for k in all_back_keys]
        plt.bar([x + i * width for x in range(len(all_back_keys))], values, width=width, label=name)
    plt.xticks([x + width for x in range(len(all_back_keys))], all_back_keys)
    plt.legend()
    plt.title("不同开奖日后区号码分布对比")
    plt.xlabel("号码")
    plt.ylabel("频率")
    plt.tight_layout()
    plt.show()

# ===================== 专家数据处理 =====================

def extract_number(text):
    if not isinstance(text, str):
        return None
    m = re.search(r'\d+', text)
    return int(m.group()) if m else None

def fetch_expert_from_cmzj(limit=20):
    url = "https://i.cmzj.net/expert/hotExpertList?lottery=2"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        print("🔄 正在获取专家数据...")
        res = requests.get(url, headers=headers)
        data = res.json().get("data", [])

        experts = []
        for exp in data[:limit]:
            expert_id = exp.get("expertId")
            detail = get_expert_info(expert_id)

            if detail:
                experts.append({
                    "专家ID": expert_id,
                    "姓名": detail.get("name"),
                    "彩龄": detail.get("cai_ling"),
                    "文章数量": detail.get("articles"),
                    "双色球一等奖": detail.get("双色球一等奖"),
                    "双色球二等奖": detail.get("双色球二等奖"),
                    "双色球三等奖": detail.get("双色球三等奖"),
                    "大乐透一等奖": detail.get("大乐透一等奖"),
                    "大乐透二等奖": detail.get("大乐透二等奖"),
                    "大乐透三等奖": detail.get("大乐透三等奖"),
                })

        df = pd.DataFrame(experts)
        df.to_csv("expert_list.csv", index=False, encoding="utf-8-sig")
        print("✅ 专家数据保存完毕")
        return df
    except Exception as e:
        print(f"❌ 抓取专家数据失败：{e}")
        return None



import requests


def get_expert_info(expert_id):
    url = f"https://i.cmzj.net/expert/queryExpertById?expertId={expert_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") == 0 and data.get("data"):
            expert = data["data"]
            return {
                "name": expert.get("name"),
                "grade": expert.get("gradeName"),
                "cai_ling": expert.get("age"),
                "articles": expert.get("articles"),
                "双色球一等奖": expert.get("ssqOne", 0),
                "双色球二等奖": expert.get("ssqTwo", 0),
                "双色球三等奖": expert.get("ssqThree", 0),
                "大乐透一等奖": expert.get("dltOne", 0),
                "大乐透二等奖": expert.get("dltTwo", 0),
                "大乐透三等奖": expert.get("dltThree", 0)
            }
        else:
            print(f"接口返回错误或无数据: {data.get('msg')}")
            return None
    except Exception as e:
        print(f"请求失败: {e}")
        return None

def analyze_expert_data(df):
    print("\n📊 开始分析专家数据...")

    # 类型转换
    for col in ["彩龄", "文章数量", "双色球一等奖", "双色球二等奖", "双色球三等奖",
                "大乐透一等奖", "大乐透二等奖", "大乐透三等奖"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["双色球总奖"] = df["双色球一等奖"] + df["双色球二等奖"] + df["双色球三等奖"]
    df["大乐透总奖"] = df["大乐透一等奖"] + df["大乐透二等奖"] + df["大乐透三等奖"]
    df["总中奖数"] = df["双色球总奖"] + df["大乐透总奖"]

    # ========== 1. 彩龄 vs 发文量 ==========
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="彩龄", y="文章数量", size="总中奖数", hue="总中奖数", palette="viridis", sizes=(20, 200))
    plt.title("彩龄 vs 文章数量（气泡大小=总中奖数）")
    plt.xlabel("彩龄（年）")
    plt.ylabel("文章数量")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ========== 2. 彩龄与中奖数关系 ==========
    plt.figure(figsize=(10, 5))
    sns.regplot(data=df, x="彩龄", y="总中奖数", scatter_kws={'s': 60})
    plt.title("彩龄对中奖总数的影响")
    plt.xlabel("彩龄（年）")
    plt.ylabel("总中奖数")
    plt.tight_layout()
    plt.show()

    # ========== 3. 发文量与中奖总数 ==========
    plt.figure(figsize=(10, 5))
    sns.regplot(data=df, x="文章数量", y="总中奖数", scatter_kws={'s': 60}, color='orange')
    plt.title("发文量对中奖总数的影响")
    plt.xlabel("文章数量")
    plt.ylabel("总中奖数")
    plt.tight_layout()
    plt.show()

    # ========== 4. 聚类分析 ==========
    print("\n🤖 正在进行专家聚类分析（KMeans）...")
    features = df[["彩龄", "文章数量", "总中奖数"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["聚类标签"] = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="彩龄", y="文章数量", hue="聚类标签", size="总中奖数", palette="Set1", sizes=(50, 250))
    plt.title("专家聚类分布（基于彩龄、文章、中奖）")
    plt.tight_layout()
    plt.show()

    # ========== 5. PCA 降维可视化 ==========
    print("🧬 使用PCA对专家进行二维可视化...")
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df["PCA1"] = components[:, 0]
    df["PCA2"] = components[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="聚类标签", style="聚类标签", s=100)
    for i in range(len(df)):
        plt.text(df["PCA1"][i]+0.1, df["PCA2"][i], df["姓名"][i], fontsize=9)
    plt.title("专家聚类（PCA降维可视化）")
    plt.tight_layout()
    plt.show()

    # ========== 6. 排名前几的专家 ==========
    top_experts = df.sort_values("总中奖数", ascending=False).head(5)
    print("\n🏆 中奖最多的前5位专家：")
    print(top_experts[["姓名", "彩龄", "文章数量", "总中奖数"]])

# ===================== 主程序入口 =====================

if __name__ == "__main__":
    # 大乐透数据处理与分析
    df_dlt = fetch_dlt_before_july1()
    if df_dlt is not None:
        analyze_and_visualize(df_dlt)
        analyze_draw_days(df_dlt)

    # 专家数据抓取与分析
    df_expert = fetch_expert_from_cmzj()
    if df_expert is not None:
        analyze_expert_data(df_expert)


