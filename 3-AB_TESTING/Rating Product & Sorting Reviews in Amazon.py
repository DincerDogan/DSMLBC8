
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#######################################################################################
# Görev 1: Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
#######################################################################################

df_a = pd.read_csv("0_DATASETS/amazon_review.csv")
df = df_a.copy()
df.head(10)
df_a.head()
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(5))
    print("####### Generate descriptive statistics##########")
    print(dataframe.describe().T)
    print("##################### Tail #####################")
    print(dataframe.tail(5))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
print("---------------Dataset Summary------------------------")
check_df(df)

# Adım 1: Ürünün ortalama puanını hesaplayınız.

df["overall"].mean()

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.
# • reviewTime değişkenini tarih değişkeni olarak tanıtmanız
# • reviewTime'ın max değerini current_date olarak kabul etmeniz
# • her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturmanız ve
# gün cinsinden ifade edilen değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar)
# çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir. Örneğin q1 = 12 ise ağırlıklandırırken
# 12 günden az süre önce yapılan yorumların ortalamasını alıp bunlara yüksek ağırlık vermek gibi.

df["reviewTime"] = pd.to_datetime(df["reviewTime"], dayfirst=True)
df["reviewTime"].max()
# Out[36]: Timestamp('2014-12-07 00:00:00')
current_date = pd.to_datetime("2014-12-07")
df["days"] = (current_date - df["reviewTime"]).dt.days
df.head()

a = df["days"].quantile(0.25)
b = df["days"].quantile(0.50)
c = df["days"].quantile(0.75)

df.loc[df["days"] <= a, "overall"].mean()
# Out[55]: 4.6957928802588995
df.loc[(df["days"] > a) & (df["days"] <= b), "overall"].mean()
# Out[56]: 4.636140637775961
df.loc[(df["days"] > b) & (df["days"] <= c), "overall"].mean()
# Out[57]: 4.571661237785016
df.loc[(df["days"] > c), "overall"].mean()
# Out[58]: 4.4462540716612375

df.loc[df["days"] <= a, "overall"].mean() * 28 / 100 + \
    df.loc[(df["days"] > a) & (df["days"] <= b), "overall"].mean() * 26 / 100 + \
    df.loc[(df["days"] > b) & (df["days"] <= c), "overall"].mean() * 24 / 100 + \
    df.loc[(df["days"] > c), "overall"].mean() * 22 / 100
# Out[59]: 4.595593165128118
# Yorum: Ilk donemki puanlama 4.695 iken , son donemde bu deger 4.446 dusmustur.
#Puanlamadaki dusus sonraki donemleerede yansimistir. Agirlikli ortalama ise 4.595 olarak gozlenmistir.


#######################################################################################
# Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
#######################################################################################

# Adım 1: helpful_no değişkenini üretiniz.
# • total_vote bir yoruma verilen toplam up-down sayısıdır.
# • up, helpful demektir.
# • Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
# • Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy
# sayılarını (helpful_no) bulunuz.

df.head()
df["helpful"].value_counts()
new_features = df["helpful"].str.split(",",expand=True)
new_features = new_features.astype("string")
helpful_yes = new_features[0].str.lstrip("[")
helpful_yes = helpful_yes.astype("int64")

total_vote = new_features[1].str.rstrip("]")
total_vote = total_vote.astype("int64")

helpful_no = total_vote - helpful_yes
helpful_no.head()

df["helpful_yes"] = helpful_yes
df["helpful_no"] = helpful_no
df["total_vote"] = total_vote
#
# helpful_no = df["total_vote"] - df["helpful_yes"]
# helpful_no.head()
#
# df["helpful_no"] = helpful_no


# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.
# score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff,
# score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.
# • score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.
# • score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
# • wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.

def score_pos_neg_diff(pos, neg):
    return pos - neg

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]),axis=1)

def score_average_rating(pos, neg):
    if pos - neg == 0:
        return 0
    return pos / (pos + neg)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()

def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla
    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.
    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı
    Returns
    -------
    wilson score: float
    """
    import scipy.stats as st
    import math
    n = pos + neg
    if (pos-neg) == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],x["helpful_no"]),axis=1)

# Adım 3: 20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.
# • wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.
# • Sonuçları yorumlayınız.

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# Yorum: Eger sadece wilson lower bound skor hesaplamak icin kullanilirsa yorun degiskenlerinin etkileri dikkate alinmamis
# olur. Bu yuzden total_score ortalama oylama degiskeni etkiside dikkate alinarak sorting islmi daha etkin saglanmis olur.


df["total_score"] = (df["score_average_rating"] * 40 / 100 + df["wilson_lower_bound"] * 60 / 100)

df["score_average_rating"].sort_values(ascending=False).head()
df["wilson_lower_bound"].sort_values(ascending=False).head()
df["score_pos_neg_diff"].sort_values(ascending=False).head()
df["helpful"][df["total_score"].sort_values(ascending=False).head(20).index]

df.columns