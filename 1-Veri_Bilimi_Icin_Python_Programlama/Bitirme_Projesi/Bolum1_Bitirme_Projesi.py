#########################################################################################################################
# BUSSINESS CASE:----------------------------------------------------------------------------------------
# A game company, by using some features of its customers, wants to create new level-based customer definitions[Persona]
# and new segments according to these new customer definitions.
# With these new segments, they also want to calculate and guess how much they can earn on average from these new customers.
# For example, they want to define, how much on average they can earn from 25years old, Male, iOS User, from Brazil.
#
# DATASET STORY:-----------------------------------------------------------------------------------------
# Persona.csv dataset shows the prices of products sold by an international game company and contains some demographic
# information
# of users who have purchased these products.
# The data set consists of records created in each sales transaction. This means that the table is not deduplicated.
# In other words, a user with specific demographic characteristics may have made more than one purchase.
#
# VARIABLES (Features):---------------------------------------------------------------------------------
#
# PRICE: Customer spend amount
# SOURCE: The type of device the customer is connecting to (IOS/Android)
# SEX: Gender of the customer
# COUNTRY: Country of the customer
# AGE: Age of the customer
#########################################################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
##########################################################################################################################
# Görev 1: Aşağıdaki Soruları Yanıtlayınız
# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df=pd.read_csv("Bolum1/persona.csv") #pycharm da dosya uzerine tıklayıp path from root alınır. ve df adında dataframe geldi
df.head() #ilk 5 gozlem gelir

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
no_of_uniuqe_source=df["SOURCE"].nunique()
frequency_of_source=df["SOURCE"].value_counts()
print("Number of unique SOURCE: {} ,\n Frekanslari:\n{}".format(no_of_uniuqe_source,frequency_of_source))
        # Number of unique SOURCE: 2 ,
        #  Frekanslari:
        # android    2974
        # ios        2026
        # Name: SOURCE, dtype: int64

# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()
        # Out[22]: 6

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts().sort_index(ascending=True)
        # Out[33]:
        # 9      200
        # 19     992
        # 29    1305
        # 39    1260
        # 49    1031
        # 59     212
        # Name: PRICE, dtype: int64

df["PRICE"].value_counts().sort_values(ascending=True)
        # Out[31]:
        # 9      200
        # 59     212
        # 19     992
        # 49    1031
        # 39    1260
        # 29    1305
        # Name: PRICE, dtype: int64

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts().sort_index(ascending=True)
df["COUNTRY"].value_counts().sort_values(ascending=False)
        # Out[37]:
        # usa    2065
        # bra    1496
        # deu     455
        # tur     451
        # fra     303
        # can     230
        # Name: COUNTRY, dtype: int64
df.groupby("COUNTRY")["PRICE"].count().sort_values(ascending=False)
        # Out[39]:
        # COUNTRY
        # usa    2065
        # bra    1496
        # deu     455
        # tur     451
        # fra     303
        # can     230
        # Name: PRICE, dtype: int64

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.head()
df.groupby("COUNTRY").agg({"PRICE": ["count", "sum"]})
        # Out[41]:
        #         PRICE
        #         count    sum
        # COUNTRY
        # bra      1496  51354
        # can       230   7730
        # deu       455  15485
        # fra       303  10177
        # tur       451  15689
        # usa      2065  70225

df.groupby("COUNTRY").agg({"PRICE": ["count", "sum"]}).sort_values(by=("PRICE", 'sum')) #ascending=True
        # Out[45]:
        #         PRICE
        #         count    sum
        # COUNTRY
        # can       230   7730
        # fra       303  10177
        # deu       455  15485
        # tur       451  15689
        # bra      1496  51354
        # usa      2065  70225

df.groupby("COUNTRY").agg({"PRICE": ["count", "sum"]}).sort_values(by=("PRICE", 'sum'),ascending=False)
        # Out[47]:
        #         PRICE
        #         count    sum
        # COUNTRY
        # usa      2065  70225
        # bra      1496  51354
        # tur       451  15689
        # deu       455  15485
        # fra       303  10177
        # can       230   7730

# Soru 7: SOURCE türlerine göre satış sayıları nedir?
df["SOURCE"].value_counts()
df.groupby("SOURCE")["PRICE"].count()
        # Out[49]:
        # SOURCE
        # android    2974
        # ios        2026
        # Name: PRICE, dtype: int64

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE": ["mean"]})
        # Out[50]:
        #              PRICE
        #               mean
        # COUNTRY
        # bra      34.327540
        # can      33.608696
        # deu      34.032967
        # fra      33.587459
        # tur      34.787140
        # usa      34.007264

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE").agg({"PRICE": ["mean"]})
# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(by=["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})
df.groupby(["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})

##########################################################################################################################
# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.groupby(["COUNTRY", 'SOURCE','SEX','AGE']).agg({"PRICE": "mean"})

##########################################################################################################################
# Görev 3: Çıktıyı PRICE’a göre sıralayınız.
agg_df = df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()

##########################################################################################################################
# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.
agg_df.index
agg_df.reset_index(inplace=True)
agg_df.head()
#Not:

# Examples for Görev 4:
#
# df['index1'] = df.index
# or, .reset_index:
#
# df = df.reset_index(level=0)
# so, if you have a multi-index frame with 3 levels of index, like:
#
# >>> df
#                        val
# tick       tag obs
# 2016-02-26 C   2    0.0139
# 2016-02-27 A   2    0.5577
# 2016-02-28 C   6    0.0303
# and you want to convert the 1st (tick) and 3rd (obs) levels in the index into columns, you would do:
#
# >>> df.reset_index(level=['tick', 'obs'])
#           tick  obs     val
# tag
# C   2016-02-26    2  0.0139
# A   2016-02-27    2  0.5577
# C   2016-02-28    6  0.0303
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
# df = pd.DataFrame([('bird', 389.0),
#                    ('bird', 24.0),
#                    ('mammal', 80.5),
#                    ('mammal', np.nan)],
#                   index=['falcon', 'parrot', 'lion', 'monkey'],
#                   columns=('class', 'max_speed'))
# df
#          class  max_speed
# falcon    bird      389.0
# parrot    bird       24.0
# lion    mammal       80.5
# monkey  mammal        NaN
# When we reset the index, the old index is added as a column, and a new sequential index is used:
#
# df.reset_index()
#     index   class  max_speed
# 0  falcon    bird      389.0
# 1  parrot    bird       24.0
# 2    lion  mammal       80.5
# 3  monkey  mammal        NaN
# We can use the drop parameter to avoid the old index being added as a column:
#
# df.reset_index(drop=True)
#     class  max_speed
# 0    bird      389.0
# 1    bird       24.0
# 2  mammal       80.5
# 3  mammal        NaN
# You can also use reset_index with MultiIndex.
#
# index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
#                                    ('bird', 'parrot'),
#                                    ('mammal', 'lion'),
#                                    ('mammal', 'monkey')],
#                                   names=['class', 'name'])
# columns = pd.MultiIndex.from_tuples([('speed', 'max'),
#                                      ('species', 'type')])
# df = pd.DataFrame([(389.0, 'fly'),
#                    ( 24.0, 'fly'),
#                    ( 80.5, 'run'),
#                    (np.nan, 'jump')],
#                   index=index,
#                   columns=columns)
# df
#                speed species
#                  max    type
# class  name
# bird   falcon  389.0     fly
#        parrot   24.0     fly
# mammal lion     80.5     run
#        monkey    NaN    jump
# If the index has multiple levels, we can reset a subset of them:
#
# df.reset_index(level='class')
#          class  speed species
#                   max    type
# name
# falcon    bird  389.0     fly
# parrot    bird   24.0     fly
# lion    mammal   80.5     run
# monkey  mammal    NaN    jump
# If we are not dropping the index, by default, it is placed in the top level. We can place it in another level:
#
# df.reset_index(level='class', col_level=1)
#                 speed species
#          class    max    type
# name
# falcon    bird  389.0     fly
# parrot    bird   24.0     fly
# lion    mammal   80.5     run
# monkey  mammal    NaN    jump
# When the index is inserted under another level, we can specify under which one with the parameter col_fill:
#
# df.reset_index(level='class', col_level=1, col_fill='species')
#               species  speed species
#                 class    max    type
# name
# falcon           bird  389.0     fly
# parrot           bird   24.0     fly
# lion           mammal   80.5     run
# monkey         mammal    NaN    jump
# If we specify a nonexistent level for col_fill, it is created:
#
# df.reset_index(level='class', col_level=1, col_fill='genus')
#                 genus  speed species
#                 class    max    type
# name
# falcon           bird  389.0     fly
# parrot           bird   24.0     fly
# lion           mammal   80.5     run
# monkey         mammal    NaN    jump

            # Out[68]:
            #   COUNTRY   SOURCE     SEX  AGE  PRICE
            # 0     bra  android    male   46   59.0
            # 1     usa  android    male   36   59.0
            # 2     fra  android  female   24   59.0

            # Onceki
            # Out[69]:
            #                             PRICE
            # COUNTRY SOURCE  SEX    AGE
            # bra     android male   46    59.0
            # usa     android male   36    59.0
            # fra     android female 24    59.0
            # usa     ios     male   32    54.0
            # deu     android female 36    49.0

# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
# ‘0_18', ‘19_23', '24_30', '31_40', '41_70'
def age_category(age):
    if age in range(0,19):
        return "0-18"
    elif age in range(18,24):
        return "19-23"
    elif age in range(24,31):
        return "24-30"
    elif age in range(31,41):
        return "31-40"
    else:
        return "41-70"
agg_df["AGE_CAT"]=agg_df.apply(lambda x:age_category(x["AGE"]),axis=1)

# Alternatif Cozum:
# # Age sayısal değişkenini kategorik değişkene çeviriniz.
# # Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# # Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'
#
# # AGE değişkeninin nerelerden bölüneceğini belirtelim:
# bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
#
# # Bölünen noktalara karşılık isimlendirmelerin ne olacağını ifade edelim:
# mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]
#
# # age'i bölelim:
# agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
# agg_df.head()

# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
agg_df["Customer_level_based"]=agg_df["COUNTRY"]+"_"+agg_df["SOURCE"]+"_"+agg_df["SEX"]+"_"+agg_df["AGE_CAT"]
agg_df["Customer_level_based"]=agg_df["Customer_level_based"].str.upper()
agg_df=agg_df.groupby("Customer_level_based").agg({"PRICE": "mean"})

# customers_level_based index'te yer almaktadır. Bunu değişkene çevirelim.
agg_df = agg_df.reset_index()
agg_df

# kontrol edelim. her bir persona'nın 1 tane olmasını bekleriz:
agg_df["Customer_level_based"].value_counts()
agg_df.head()

# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})

# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
# • 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user="TUR_ANDROID_FEMALE_31-40"
agg_df[agg_df["Customer_level_based"]==new_user]
# • 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user="FRA_IOS_FEMALE_31-40"
agg_df[agg_df["Customer_level_based"]==new_user]

