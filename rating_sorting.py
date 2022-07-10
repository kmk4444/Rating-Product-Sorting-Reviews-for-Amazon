import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###################################### TASK 1 ##########################################################
df = pd.read_csv("WEEK_4/ÖDEVLER/rating_sorting_amazon/amazon_review.csv")

#Step 1: Estimate average point of the product

df["overall"].value_counts()

df.groupby("asin").agg({"overall":"mean"})

# Step 2: Estimate time-Based Weighted Average

df.info()

# type of reviewtime convert to datetime
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

#find current_date according to maximum date of reviewtime
df["reviewTime"].max()
current_date = pd.to_datetime("2014-12-07")

#we found the difference between the dates
df["days"] = (current_date - df["reviewTime"]).dt.days

# we find quartiles so as to seperate groups.
df["days"].quantile([.25, .5, .75])

# as we see below, in recent days, the point of the product increased.
df.loc[df["days"] <= 280, "overall"].mean()
df.loc[(df["days"] > 280) & (df["days"] <= 430), "overall"].mean()
df.loc[(df["days"] > 430) & (df["days"] <= 600), "overall"].mean()
df.loc[df["days"] > 600,"overall"].mean()


# we estimate mean time based weighted average.
df.loc[df["days"] <= 280, "overall"].mean() * 28/100 + \
    df.loc[(df["days"] > 280) & (df["days"] <= 430), "overall"].mean() * 26/100 + \
    df.loc[(df["days"] > 430) & (df["days"] <= 600), "overall"].mean() * 24/100 + \
    df.loc[df["days"] > 600,"overall"].mean() * 22/100

#function

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
   return   dataframe.loc[dataframe["days"] <= 280, "overall"].mean() * w1 / 100 + \
            dataframe.loc[(dataframe["days"] > 280) & (dataframe["days"] <= 430), "overall"].mean() * w2 / 100 + \
            dataframe.loc[(dataframe["days"] > 430) & (dataframe["days"] <= 600), "overall"].mean() * w3 / 100 + \
            dataframe.loc[dataframe["days"] > 600, "overall"].mean() * w4 / 100


time_based_weighted_average(df)

###################################### TASK 2 ##########################################################
# Specify 20 reviews to be displayed on the product detail page for the product.

#Step 1: create helpful_no variable

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

#Step 2: estimate score_pos_neg_diff, score_average_rating ve wilson_lower_bound
#and add dataframe.

def score_pas_neg_diff(up, down):
   return up - down

df["score_pas_neg_diff"] = df.apply(lambda x:score_pas_neg_diff(x["helpful_yes"],
                                                                x["helpful_no"]), axis =1)

def score_average_rating(up, down):
   if up + down == 0:
      return 0
   else:
      return up / (up +down)

df["score_average_rating"] = df.apply(lambda x:score_average_rating(x["helpful_yes"],
                                                                    x["helpful_no"]), axis=1)

def wilson_lower_bound(up, down, confidence=0.95):
   """
   Wilson Lower Bound Score hesapla

   - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
   - Hesaplanacak skor ürün sıralaması için kullanılır.
   - Not:
   Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
   Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

   Parameters
   ----------
   up: int
       up count
   down: int
       down count
   confidence: float
       confidence

   Returns
   -------
   wilson score: float

   """
   n = up + down
   if n == 0:
      return 0
   z = st.norm.ppf(1 - (1 - confidence) / 2)
   phat = 1.0 * up / n
   return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

df[["reviewerID","wilson_lower_bound","days","helpful_yes","helpful_no","total_vote","score_pas_neg_diff","score_average_rating"]].sort_values(by = "wilson_lower_bound", ascending=False).head(20)

