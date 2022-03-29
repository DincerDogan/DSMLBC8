import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

###################################################################################
# Görev 1: Veriyi Anlama ve Hazırlama
###################################################################################
# Adım1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("0_DATASETS/flo_data_20k.csv")
df_.head()
df_.describe().T #"order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" degerlerinde max da cok yuksek degeler gozendi 200-109 yuzde 75 lik degerler 4-2 olmasına karsın
df_.head()
df_.isnull().sum()  #df.dropna(inplace=True) na degerleri olmadıgı icin gerek yok
df=df_.copy()
type(df)


# Adım2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Adım3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")
df.describe().T  #max degerlerinde aykırı degerlerin silinmesiyle bir nebze duzelme oldu.

# Adım4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["total_order_num"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["total_customer_value"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]
df.head()

# Adım5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
df.dtypes
df.loc[:, df.columns.str.contains("date")] = df.loc[:, df.columns.str.contains("date")].apply(pd.to_datetime)

###################################################################################
# Görev 2: CLTV Veri Yapısının Oluşturulması
###################################################################################
# Adım1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df.head()
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)
# Adım2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
#recency:Son atın alma uzerınden gecen zaman
#T:Musterinin yası
#frequency:tekrar eden tolam satın alma sayısı
#monetary:Satın alma basına ortalama kazanc

df['recency_day']=(df["last_order_date"]- df["first_order_date"])
type(df['recency_day'][0])
df['recency_day_2'] = df['recency_day']/ pd.to_timedelta(1, unit='D')
# df['master_id'].nunique()
# cltv_df = df.groupby('master_id').agg({'recency_day': lambda receny_day: receny_day})
cltv_df = df.groupby('master_id').agg({'last_order_date':[lambda last_order_date: (today_date - last_order_date.max()).days] ,
                                       'recency_day_2': lambda receny_day: receny_day,
                                       'total_order_num': lambda total_order_num: total_order_num,
                                       'total_customer_value': lambda total_customer_value: total_customer_value.sum()})

# cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency_cltv_weekly', 'T_weekly', 'frequency', 'monetary_cltv_avg']
cltv_df["monetary_cltv_avg"] = cltv_df["monetary_cltv_avg"] / cltv_df["frequency"]
cltv_df.describe().T
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df = cltv_df[cltv_df['recency_cltv_weekly'] < cltv_df['T_weekly'] ]

cltv_df["recency_cltv_weekly"] = cltv_df["recency_cltv_weekly"] / 7
cltv_df["T_weekly"] = cltv_df["T_weekly"] / 7
cltv_df['frequency']=cltv_df['frequency'].astype(int)

###################################################################################
# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
###################################################################################
# Adım1: BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])
# • 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
# dataframe'ine ekleyiniz.
bgf.predict(4*3,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']).sort_values(ascending=False).head(10)

cltv_df["expected_3_month"] = bgf.predict(4*3,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])

cltv_df["expected_3_month"].sort_values(ascending=False).head(10)
# • 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
# dataframe'ine ekleyiniz.
bgf.predict(4*6,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']).sort_values(ascending=False).head(10)

cltv_df["expected_6_month"] = bgf.predict(4*6,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])

cltv_df["expected_6_month"].sort_values(ascending=False).head(10)


# Adım2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

# Adım3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
# Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary_cltv_avg']).sort_values(ascending=False).head(20)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="master_id", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

###################################################################################
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
###################################################################################


# Adım1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
cltv_final

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])


# Adım2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.
cltv_final.groupby("segment").agg({"count", "mean", "sum"})

cltv_final.groupby("segment").agg({"recency_cltv_weekly":["count", "mean", "sum"],"monetary_cltv_avg":["sum"]})