import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###################################################################################
# Görev 1: Veriyi Anlama ve Hazırlama
###################################################################################
# Adım1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("0_DATASETS/flo_data_20k.csv")
df_.head()

###################################################################################
# Adım2: Veri setinde.
# a. İlk 10 gözlem,
df_.head(10)
# b. Değişken isimleri,
df_.columns
# c. Betimsel istatistik,
df_.describe().T
# d. Boş değer,
df_.isnull().sum()
# e. Değişken tipleri, incelemesi yapınız.
df_.dtypes

###################################################################################
# Adım3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df=df_.copy()
df.head()
df["total_order_num"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["total_customer_value"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]

# Adım4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
df.dtypes
df.loc[:, df.columns.str.contains("date")] = df.loc[:, df.columns.str.contains("date")].apply(pd.to_datetime)

# Adım5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.

df_deneme = df.groupby(["order_channel"]).agg({"master_id": ["count"],
                                               "total_order_num": ["count"],
                                               "total_customer_value": ["sum"]})


         

# Adım6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

temp_df = df.sort_values(["total_customer_value"], ascending=False)[:10]
temp_df[["master_id","total_customer_value"]]
temp_df.shape



# Adım7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
temp_df = df.sort_values(["total_order_num"], ascending=False)[:10]
temp_df[["master_id","total_order_num"]]
temp_df.shape



# Adım8: Veri ön hazırlık sürecini fonksiyonlaştırınız.
def create_rfm(dataframe, csv=False):
    # VERIYI OKUMA


    # VERIYI HAZIRLAMA
    dataframe.dropna(inplace=True)
    dataframe["total_order_num"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_customer_value"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    # DEGISKEN AYARLARININ YAPILMASI
    dataframe.loc[:, dataframe.columns.str.contains("date")] = dataframe.loc[:, dataframe.columns.str.contains("date")].apply(pd.to_datetime)

    #EN FAZLA KAZANCI GETIREN  ILK 10 FIRMA
    temp_df = dataframe.sort_values(["total_customer_value"], ascending=False)[:10]
    print("Ilk 10 firma kazanci fazla olan\n\n\n ")
    print(temp_df[["master_id", "total_customer_value"]])
    print("########################################\n\n\n ")

    # EN FAZLA SIPARISI VEREN ILK 10 MUSTERI
    temp_df2 = dataframe.sort_values(["total_order_num"], ascending=False)[:10]
    temp_df2[["master_id", "total_order_num"]]
    print("Ilk 10 firma siparisi fazla olan\n\n\n ")
    print(temp_df2[["master_id", "total_order_num"]])
    print("########################################\n\n\n ")

    return dataframe

create_rfm(df)

###################################################################################
# Görev 2: RFM Metriklerinin Hesaplanması
###################################################################################
# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.

    #Recency:musterinin en bizden ne zaman alisveris yaptigi durumunu ifade etmektedir
    # Frequency:musterinin toplam yaptigi alisveris sayisidir.Islem sayisidir
    # Monetary:musterilerin bize biraktigi parasal degeri ifade(eder)

# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
df.head()

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)
df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'total_order_num': lambda total_order_num: total_order_num,
                                     'total_customer_value': lambda total_customer_value: total_customer_value.sum()})

# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'total_order_num': lambda total_order_num: total_order_num,
                                     'total_customer_value': lambda total_customer_value: total_customer_value.sum()})


# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]
rfm.shape

rfm.head()


###################################################################################
# Görev 3: RF Skorunun Hesaplanması
###################################################################################
# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +rfm['frequency_score'].astype(str))

rfm.describe().T

###################################################################################
# Görev 4: RF Skorunun Segment Olarak Tanımlanması
###################################################################################
# Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# Adım 2: Yukardaki seg_map yardımı ile skorları segmentlere çeviriniz.
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)


###################################################################################
# Görev 5: Aksiyon Zamanı !
###################################################################################
# Adım1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# Adım2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.
# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
# iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
# yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.
# 1. adim sadece kadin kategorisinden alisveris yapanlar dataframe alinir

kadin_df=df[(df["interested_in_categories_12"].str.contains("KADIN",na=False))]
kadin_masterid_df=kadin_df["master_id"]
kadin_masterid_df.count()


# 2.Sadik musterilerin dataframe alinir.
rfm2=rfm[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")]
rfm3=rfm2.reset_index()

# 3.musteri_id kesisimde sadik musteriler sadece kadin kategorisinde fitlre yapilir.
common=rfm3.merge(kadin_masterid_df,on=["master_id"])
result=rfm3[rfm3.master_id.isin(common.master_id)]
result.count()

# 4.result csv dosyasina aktarilir.
result.to_csv("Adim2_a.csv")


# b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.
# 1. adim sadece kadin kategorisinden alisveris yapanlar dataframe alinir
erkek_cocuk_df=df[(df["interested_in_categories_12"].str.contains("ERKEK",na=False)) & (df["interested_in_categories_12"].str.contains("COCUK",na=False))]


erkek_cocuk_masterid_df=erkek_cocuk_df["master_id"]

rfm22=rfm[(rfm["segment"] == "cant_loose") | (rfm["segment"] == "new_customers")|(rfm["segment"] == "about_to_sleep")]
rfm33=rfm22.reset_index()

common2=rfm33.merge(erkek_cocuk_masterid_df,on=["master_id"])
result2=rfm33[rfm33.master_id.isin(common2.master_id)]
result2.count()
result2.to_csv("Adim2_b.csv")

