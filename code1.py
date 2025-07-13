import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re

plt.rcParams["font.family"] = ["SimHei", "STXihei", "STSong"]
plt.rcParams["axes.unicode_minus"] = False
print(f"已设置中文字体: {plt.rcParams['font.family'][0]}")

def fetch_and_save_html_pages(url, max_pages=50):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(180)

    try:
        print(f"正在加载页面: {url}")
        driver.get(url)
        time.sleep(10)
    except Exception as e:
        print(f"页面加载失败: {e}")
        driver.quit()
        return []

    try:
        print("正在等待页面加载完成...")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table#table'))
        )
        print("页面加载完成")
    except Exception as e:
        print(f"等待超时或元素未加载: {e}")
        driver.quit()
        return []

    with open("debug_hurun_firstpage.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    driver.save_screenshot("debug_hurun_firstpage.png")
    print("HTML 和截图已保存")

    pages_html = [driver.page_source]
    for page in range(1, max_pages):
        try:
            print(f"正在抓取第 {page} 页...")
            next_btn = driver.find_element(By.CSS_SELECTOR, 'a.page-link[aria-label="下一页"]')
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(4)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            pages_html.append(driver.page_source)
        except Exception as e:
            print(f"抓取到第 {page} 页后停止，未发现下一页按钮或错误：{e}")
            break

    driver.quit()
    return pages_html

def parse_all_pages(pages_html):
    data = []
    for html in pages_html:
        soup = BeautifulSoup(html, "html.parser")
        for row in soup.select("table#table > tbody > tr"):
            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            rank = cols[0].get_text(strip=True).replace("No.", "")
            wealth = cols[1].get_text(strip=True).replace("\u00a0", "").replace("￥", "").replace("亿", "").strip()
            change = cols[2].get_text(strip=True)
            name = cols[3].select_one(".hs-index-list-name span")
            age = cols[3].select_one(".hs-index-list-gender")
            company = cols[4].select_one(".company")
            industry = cols[4].select_one(".industry")
            birthplace = cols[4].select_one(".birthplace")

            data.append({
                "排名": rank,
                "财富（亿）": wealth,
                "排名变化": change,
                "姓名": name.get_text(strip=True) if name else "",
                "年龄": age.get_text(strip=True) if age else "",
                "公司": company.get_text(strip=True) if company else "",
                "行业": industry.get_text(strip=True).replace("行业：", "") if industry else "",
                "出生地": birthplace.get_text(strip=True) if birthplace else "",
            })

    df = pd.DataFrame(data)
    print("抓取的列名: ", df.columns)
    print("数据预览: ", df.head())

    if "行业" not in df.columns:
        print("未成功抓取到 '行业' 列")
        return pd.DataFrame()

    df['性别'] = df['年龄'].str.extract(r'(先生|女士)')
    df['年龄'] = df['年龄'].str.extract(r'(\d+)').astype(float)
    df['财富（亿）'] = pd.to_numeric(df['财富（亿）'], errors='coerce')
    df.to_csv("hurun_richlist_2024.csv", index=False, encoding='utf-8-sig')
    print(f"共提取 {len(df)} 条数据")
    return df

def analyze_by_industry(df):
    # 清理字段（统一列名，去除空格）
    df.columns = df.columns.str.strip()
    if '行业' not in df.columns or '财富（亿）' not in df.columns:
        raise ValueError("找不到 '行业' 或 '财富（亿）' 列，请检查列名。")

    # 定义行业整合的规则（关键词匹配）
    def classify_industry(industry):
        industry = str(industry).strip().lower()

        # 关键词匹配
        if re.search(r'(房地产|地产)', industry):
            return '房地产'
        elif re.search(r'(金融|投资)', industry):
            return '金融'
        elif re.search(r'(医疗|健康)', industry):
            return '医疗健康'
        elif re.search(r'(互联网|技术|网络)', industry):
            return '互联网'
        elif re.search(r'(制造|生产|工业)', industry):
            return '制造业'
        elif re.search(r'(食品|饮料)', industry):
            return '食品饮料'
        elif re.search(r'(能源|电力)', industry):
            return '能源'
        elif re.search(r'(交通|物流)', industry):
            return '交通物流'
        elif re.search(r'(文娱|传媒)', industry):
            return '文娱传媒'
        elif re.search(r'(零售|电商)', industry):
            return '零售'
        elif re.search(r'(环保|绿色)', industry):
            return '环保'
        elif re.search(r'(农业)', industry):
            return '农业'
        elif re.search(r'(军工)', industry):
            return '军工'
        else:
            return '其他'

    # 应用整合规则
    df["整合行业"] = df["行业"].apply(classify_industry)

    # 将财富（亿）列转换为数字格式
    df["财富（亿）"] = pd.to_numeric(df["财富（亿）"], errors='coerce')

    # 去除'其他'行业的数据
    df_filtered = df[df["整合行业"] != "其他"]

    # 按照整合行业对财富进行汇总
    industry_wealth = df_filtered.groupby("整合行业")["财富（亿）"].sum().sort_values(ascending=False)

    # 统计各行业富豪人数
    industry_counts = df_filtered["整合行业"].value_counts().sort_values(ascending=False)

    # 绘制第一个柱状图：行业财富总和
    plt.figure(figsize=(12, 8))
    industry_wealth.plot(kind='bar', color='skyblue')
    plt.xlabel("行业")
    plt.ylabel("财富总和（亿）")
    plt.title("行业发展趋势图")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 绘制第二个柱状图：行业富豪人数
    plt.figure(figsize=(12, 8))
    industry_counts.plot(kind='bar', color='lightcoral')
    plt.xlabel("行业")
    plt.ylabel("富豪人数")
    plt.title("各行业富豪人数分布")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_by_age(df):
    plt.figure(figsize=(12, 8))
    sns.histplot(df['年龄'], bins=20, kde=True, color='skyblue')
    plt.title('富豪年龄分布')
    plt.savefig("age_distribution.png", bbox_inches='tight')
    print("富豪年龄分布图已保存为 'age_distribution.png'")
    plt.show()

def analyze_by_gender(df):
    if df.empty or '性别' not in df.columns:
        print("数据为空或缺少 '性别' 列，无法进行分析。")
        return

    gender_data = df['性别'].value_counts()

    plt.figure(figsize=(8, 8))
    gender_data.plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])
    plt.title('富豪性别分布')
    plt.ylabel("")
    plt.savefig("gender_distribution.png", bbox_inches='tight')
    print("富豪性别分布图已保存为 'gender_distribution.png'")
    plt.show()

if __name__ == "__main__":
    url = "https://www.hurun.net/zh-CN/Rank/HsRankDetails?pagetype=rich"
    pages = fetch_and_save_html_pages(url)
    df = parse_all_pages(pages)
    if not df.empty:
        analyze_by_industry(df)
        analyze_by_age(df)
        analyze_by_gender(df)
