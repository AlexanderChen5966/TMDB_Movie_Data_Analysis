# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:01:01 2025

@author: USER
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# 載入資料
# file_path = 'tmdb_5000_movies.csv'
file_path = 'tmdb_data/tmdb_5000_movies.csv'

movies_df = pd.read_csv(file_path)

# 保留需要的欄位
columns_needed = ['release_date', 'title', 'budget', 'revenue', 'genres', 'vote_average', 'vote_count']

# 複製DataFrame
movies_copy_df = movies_df[columns_needed].copy()

# 使用apply()函數解析genres欄位
genre_ids_apply = movies_copy_df['genres'].apply(lambda genre_str: [genre['id'] for genre in json.loads(genre_str)])
genre_names_apply = movies_copy_df['genres'].apply(lambda genre_str: [genre['name'] for genre in json.loads(genre_str)])

# 使用for迴圈解析genres欄位
genre_names = []
for genre_str in movies_copy_df['genres']:
    genre_list = json.loads(genre_str)  # 將 JSON 字串轉換為 Python list
    names = [genre['name'] for genre in genre_list]  # 提取所有 genre 的 'name'
    genre_names.append(names)

# 儲存結果到新的欄位
movies_copy_df['genres_ids_apply'] = genre_ids_apply
movies_copy_df['genres_names_apply'] = genre_names_apply



# 資料清洗---------------------------------------------
# 1. 檢查遺漏值
print("檢查遺漏值：")
print(movies_copy_df.isnull().sum())  # 顯示每個欄位的遺漏值數量

# ---------------------------------------------
# 2. 填補或刪除遺漏值
# 若有些欄位的遺漏值可以填補，用平均數或其他方式填補
movies_copy_df['budget'].fillna(movies_copy_df['budget'].median(), inplace=True)
movies_copy_df['revenue'].fillna(movies_copy_df['revenue'].median(), inplace=True)
movies_copy_df['vote_average'].fillna(movies_copy_df['vote_average'].median(), inplace=True)
movies_copy_df['vote_count'].fillna(movies_copy_df['vote_count'].median(), inplace=True)

# 若某些欄位遺漏值太多，您可以選擇刪除這些列（如 'release_date', 'title' 等為必填欄位）
movies_copy_df.dropna(subset=['release_date', 'title'], inplace=True)

# ---------------------------------------------
# 3. 資料格式處理
# 確保數字欄位是數字類型
movies_copy_df['budget'] = pd.to_numeric(movies_copy_df['budget'], errors='coerce')  # 若無法轉換會變為 NaN
movies_copy_df['revenue'] = pd.to_numeric(movies_copy_df['revenue'], errors='coerce')
movies_copy_df['vote_average'] = pd.to_numeric(movies_copy_df['vote_average'], errors='coerce')
movies_copy_df['vote_count'] = pd.to_numeric(movies_copy_df['vote_count'], errors='coerce')

# 轉換 'release_date' 為日期型別
movies_copy_df['release_date'] = pd.to_datetime(movies_copy_df['release_date'], errors='coerce')

# ---------------------------------------------
# 4. 刪除重複資料
movies_copy_df['genres_ids_apply'] = movies_copy_df['genres_ids_apply'].apply(tuple)
movies_copy_df['genres_names_apply'] = movies_copy_df['genres_names_apply'].apply(tuple)

movies_copy_df.drop_duplicates(inplace=True)

# ---------------------------------------------
# 5. 移除字串欄位的空白字符
movies_copy_df['title'] = movies_copy_df['title'].str.strip()
movies_copy_df['genres_names_apply'] = movies_copy_df['genres_names_apply'].apply(lambda x: [name.strip() for name in x])

# ---------------------------------------------
# 6. 處理異常值
# 處理異常值，例如當 'budget' 或 'revenue' 是 0 或負值時，我們可以將其設為 NaN 或中位數
movies_copy_df['budget'] = movies_copy_df['budget'].apply(lambda x: x if x > 0 else None)
movies_copy_df['revenue'] = movies_copy_df['revenue'].apply(lambda x: x if x > 0 else None)

# 重新填補異常值
movies_copy_df['budget'].fillna(movies_copy_df['budget'].median(), inplace=True)
movies_copy_df['revenue'].fillna(movies_copy_df['revenue'].median(), inplace=True)

# ---------------------------------------------
# 最後檢查處理後的資料
print("處理後的資料：")
print(movies_copy_df.head())  # 顯示前幾筆資料

# 顯示資料清理後的遺漏值情況
print(movies_copy_df.isnull().sum())

#新增一個release_year欄位，只擷取release_date年份的資料
movies_copy_df['release_year'] = pd.to_datetime(movies_copy_df['release_date'], errors='coerce').dt.year
print(movies_copy_df.info)

# ---------------------------------------------


'''
分析標的
1.票房最高前10電影
2.每年上映的電影數量、電影總票房
3.電影預算、評分對票房的影響
4.什麼電影風格出現最多？
5.電影風格和年度的趨勢
'''

#-----------------
# 分析資料準備
# 以revenue為排序單位，由大到小進行排序
top_10_revenue_movies = movies_copy_df.sort_values(by='revenue',ascending= False).head(10)
# print(top_10_revenue_movies.head(10))#取前10名

# 計算各個年度的電影數量
print(movies_copy_df['release_year'].value_counts()) #統計年份出現的次數，會排序


# 預算、評分對票房的影響
budget_revenue_corr = movies_copy_df[['budget', 'vote_average', 'revenue']].dropna()
print(budget_revenue_corr.corr)

# 統計每個風格出現的次數
genre_counts = movies_copy_df['genres_names_apply'].explode().value_counts().reset_index()
genre_counts.columns = ['genre', 'count']
print(genre_counts)


#----------
#圖表顯示
# 設定中文
# plt.rcParams["font.family"] = "Microsoft JhengHei" #Windows 適用字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode Ms'] #mac os 適用字型
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False

# 統計每年上映的電影數量，折線圖
plt.figure(num=1)
movies_year_count=movies_copy_df.groupby(['release_year'])['title'].count()
movies_year_count.plot(figsize=(10,5),marker='.')
plt.title('每年上映電影數量統計',fontsize=22)
plt.xlabel('年份',fontsize=15)
plt.ylabel('上映電影數量',fontsize=15)
plt.show()

# 統計每年總票房，折線圖
plt.figure(num=2)
movies_year_gross=movies_copy_df.groupby(['release_year'])['revenue'].sum()
movies_year_gross.plot(figsize=(10,5),marker='.')
plt.title('每年總票房統計',fontsize=22)
plt.xlabel('年份',fontsize=15)
plt.ylabel('總票房',fontsize=15)
plt.show()



# 預算對票房的影響，散點圖
plt.figure(num=3,figsize=(10,6))
sns.scatterplot(data=budget_revenue_corr, x='budget', y='revenue')
plt.title('Budget vs Revenue')
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.show()


# 評分對票房的影響，散點圖
plt.figure(num=4,figsize=(10,6))
sns.scatterplot(data=budget_revenue_corr, x='vote_average', y='revenue')
plt.title('Rating vs Revenue')
plt.xlabel('Rating')
plt.ylabel('Revenue')
plt.show()




#%%
# 文字雲

text = " ".join(genre_counts['genre'])

wordcloud = WordCloud().generate(text)

# 繪圖
# 創建文字雲
wordcloud = WordCloud(width=800, height=400,max_words=2000, background_color='white').generate(text)

# 顯示文字雲
plt.figure(num=5,figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 移除坐標軸
plt.show()


#%%


# 統計每年電影風格的出現次數
genres_per_year = movies_copy_df['genres_names_apply'].explode().groupby([movies_copy_df['release_year'], movies_copy_df['genres_names_apply'].explode().rename('genre')]).size().reset_index(name='count')

genres_per_year.columns = ['release_year', 0, 'count']
genres_per_year = genres_per_year.rename(columns = {0:'genre'})


# 顯示結果
print(genres_per_year.head())

# 繪製每年電影風格的趨勢
plt.figure(num=6,figsize=(14,8))
sns.lineplot(data=genres_per_year, x='release_year', y='count', hue='genre', marker='o')

# 設定圖表標題與標籤
plt.title('年度電影風格趨勢')
plt.xlabel('年份')
plt.ylabel('電影風格數量')
plt.legend(title='電影風格', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 顯示圖表
plt.show()



top_10_genres = genre_counts.head(10)
print("前十名電影風格數量:",top_10_genres)
# 2. 繪製橫條直方圖
plt.figure(num=7,figsize=(10, 6))

# 使用 seaborn 的 barplot() 繪製橫條直方圖
sns.barplot(data=top_10_genres, y='genre', x='count', palette='Set2', orient='h')
bars = sns.barplot(data=top_10_genres, y='genre', x='count', palette='Set2', orient='h').patches #Edited line

# 使用 matplotlib 的 barh() 繪製橫條直方圖
# bars = plt.barh(top_10_genres['genre'],top_10_genres['count'], color=sns.color_palette("Set2", len(top_10_genres['genre'])))


# 自定義顏色
# sns.barplot(data=top_10_genres, y='genre', x='count', palette=['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#D2691E', '#8A2BE2', '#7FFF00', '#FF4500', '#FF1493', '#8B0000'], orient='h')

for bar in bars:
    # `bar.get_width()` 是條形的長度，`bar.get_y()` 是條形的起始點
    # 顯示數字的水平位置設置為條形的前端，稍微向右偏移一點
    plt.text(10,  # `bar.get_width()` 是條形的長度，`+ 20` 是向外偏移
          bar.get_y() + bar.get_height() / 2,  # 中間顯示
          f'{bar.get_width()}',  # 顯示條形的數字（即該條形的長度）
          va='center', ha='left', fontsize=12, color='black')  # 設置顏色和對齊方式

    # 顯示數字的水平位置設置為條形的尾端，稍微向外偏移一點
    # plt.text(bar.get_width() + 20,  # `bar.get_width()` 是條形的長度，`+ 20` 是向外偏移
    #       bar.get_y() + bar.get_height() / 2,  # 中間顯示
    #       f'{bar.get_width()}',  # 顯示條形的數字（即該條形的長度）
    #       va='center', ha='left', fontsize=12, color='black')  # 設置顏色和對齊方式


# 設定標題和軸標籤
plt.title('前十大電影風格', fontsize=16)
plt.xlabel('電影數量', fontsize=12)
plt.ylabel('電影風格', fontsize=12)


# 顯示圖表
plt.show()



#%%
# 統計每年上映的電影數量和總票房
yearly_movies = movies_copy_df.groupby('release_year').agg(
    num_movies=('title', 'count'),
    total_revenue=('revenue', 'sum')
).reset_index()

print("yearly_movies:",yearly_movies)

yearly_movies["percentages"] = (yearly_movies["num_movies"] / yearly_movies["num_movies"].sum()) * 100
print("yearly_movies:",yearly_movies)

# 從 yearly_stats DataFrame 中選取 'release_year' 和 'percentages' 欄位
yearly_movies_subset = yearly_movies[['release_year', 'percentages']]

# 顯示新 DataFrame
print(yearly_movies_subset.head())

# 設定百分比的閾值（小於 1%）
threshold = 1.0

# 篩選出百分比小於 1% 的年份
other_years = yearly_movies_subset[yearly_movies_subset['percentages'] < threshold]

# 計算這些年份的百分比總和
others_percentage = other_years['percentages'].sum()

# 移除小於 1% 的年份
yearly_movies_subset = yearly_movies_subset[yearly_movies_subset['percentages'] >= threshold]

# 將 'Others' 年份的資料加入到 DataFrame
others_row = pd.DataFrame({'release_year': ['Others'], 'percentages': [others_percentage]})
yearly_movies_subset = pd.concat([yearly_movies_subset, others_row], ignore_index=True)

# 顯示結果
print(yearly_movies_subset)

# 繪製圓餅圖
plt.figure(num=10,figsize=(10, 8))

# 將 'release_year' 欄位轉換為數值型態，並將 'Others' 轉換為一個特定的數值 (例如：0)
yearly_movies_subset['release_year'] = pd.to_numeric(yearly_movies_subset['release_year'], errors='coerce').fillna(0)


# 使用 'percentages' 欄位作為繪製圓餅圖的依據
wedges, texts, autotexts = plt.pie(yearly_movies_subset['percentages'], labels=yearly_movies_subset['release_year'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

# 調整標籤字體大小
for text in texts + autotexts:
    text.set_fontsize(10)  # 設定標籤字體大小

# 設定標題
plt.title('每年電影上映數量及其百分比', fontsize=22)

# 顯示圖表
plt.show()


#%%

# 計算每年電影數量
# yearly_movie_count = movies_copy_df['release_year'].value_counts().sort_index()

# # 計算每年電影數量的百分比
# total_movies = yearly_movie_count.sum()
# percentages = yearly_movie_count / total_movies * 100

# # 將小於 1% 的年份合併為 "Others"
# threshold = 1  # 設置百分比閾值
# other_years = percentages[percentages < threshold].index
# yearly_movie_count['Others'] = yearly_movie_count[other_years].sum()

# # 去掉小於 1% 百分比的年份
# yearly_movie_count = yearly_movie_count.drop(other_years)

# # 繪製圓餅圖
# plt.figure(num=10,figsize=(10, 8))

# # 繪製圓餅圖
# wedges, texts, autotexts = plt.pie(yearly_movie_count, labels=yearly_movie_count.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

# # 調整標籤字體大小
# for text in texts + autotexts:
#     text.set_fontsize(10)  # 設定標籤字體大小

# # 設定標題
# plt.title('每年電影上映數量及其百分比', fontsize=22)

# # 顯示圖表
# plt.show()


#%%
# 創建雙軸圖
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # 畫電影數量的折線圖
# ax1.plot(yearly_stats['release_year'], yearly_stats['num_movies'], color='b', label='Number of Movies')
# ax1.set_xlabel('Year')
# ax1.set_ylabel('Number of Movies', color='b')
# ax1.tick_params(axis='y', labelcolor='b')

# # 創建第二個 y 軸，並畫總票房收益的折線圖
# ax2 = ax1.twinx()
# ax2.plot(yearly_stats['release_year'], yearly_stats['total_revenue'], color='g', label='Total Revenue')
# ax2.set_ylabel('Total Revenue', color='g')
# ax2.tick_params(axis='y', labelcolor='g')

# # 添加標題
# plt.title('Number of Movies and Total Revenue Over the Years')

# # 顯示圖表
# plt.tight_layout()
# plt.show()