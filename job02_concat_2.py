import pandas as pd
import glob

data_paths = glob.glob('crawling_data/reviews_20162017.csv')
data_paths = data_paths+glob.glob('crawling_data/reviews_2016(30_59page).csv')
df = pd.DataFrame()

for path in data_paths:
    df_temp = pd.read_csv(path)
    df_temp.dropna(inplace=True)
    df_temp.drop_duplicates(inplace=True)
    df = pd.concat([df, df_temp], ignore_index=True)
df.drop_duplicates(inplace=True)
df.info()
print(len(df.titles.value_counts()))
df.to_csv('./crawling_data/review_20162017_2.csv', index=False)
