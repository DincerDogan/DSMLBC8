import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
df = sns.load_dataset("titanic")

df.head()
type(df)

# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
df["sex"].value_counts()
    # male      577
    # female    314
    # Name: sex, dtype: int64

# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.

num_of_unique_values = [(col,df[col].nunique() )for col in df.columns]

            # Out[95]:
            # [('survived', 2),
            #  ('pclass', 3),
            #  ('sex', 2),
            #  ('age', 88),
            #  ('sibsp', 7),
            #  ('parch', 7),
            #  ('fare', 248),
            #  ('embarked', 3),
            #  ('class', 3),
            #  ('who', 3),
            #  ('adult_male', 2),
            #  ('deck', 7),
            #  ('embark_town', 3),
            #  ('alive', 2),
            #  ('alone', 2)]

# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.

num_of_unique_values_of_pclass = df["pclass"].nunique()
print("Unique number of pclass variable: {} adet".format(num_of_unique_values_of_pclass))
    # Unique number of pclass variable: 3 adet

# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

[print("Unique number of {} variable: {} adet".format(col,df[col].nunique())) for col in df.columns if col in ["plcass", "parch"]]
        # Unique number of pclass variable: 3 adet
        # Unique number of parch variable: 7 adet

# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
str(df["embarked"].dtypes)
        # Out[103]: 'object'
df["embarked"]=df.embarked.astype('category')
df["embarked"]=df["embarked"].astype('category')
str(df["embarked"].dtypes)
        # Out[110]: 'category'

# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
df[df["embarked"]=="C"].head()

        #    survived  pclass     sex   age  sibsp  parch     fare embarked   class    who  adult_male deck embark_town alive  alone
        # 1          1       1  female  38.0      1      0  71.2833        C   First  woman       False    C   Cherbourg   yes  False
        # 9          1       2  female  14.0      1      0  30.0708        C  Second  child       False  NaN   Cherbourg   yes  False
        # 19         1       3  female   NaN      0      0   7.2250        C   Third  woman       False  NaN   Cherbourg   yes   True
        # 26         0       3    male   NaN      0      0   7.2250        C   Third    man        True  NaN   Cherbourg    no   True
        # 30         0       1    male  40.0      0      0  27.7208        C   First    man        True  NaN   Cherbourg    no   True

# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df[df["embarked"]!="S"].head()

        #  survived  pclass     sex   age  sibsp  parch     fare embarked   class    who  adult_male deck embark_town alive  alone
        # 1          1       1  female  38.0      1      0  71.2833        C   First  woman       False    C   Cherbourg   yes  False
        # 5          0       3    male   NaN      0      0   8.4583        Q   Third    man        True  NaN  Queenstown    no   True
        # 9          1       2  female  14.0      1      0  30.0708        C  Second  child       False  NaN   Cherbourg   yes  False
        # 16         0       3    male   2.0      4      1  29.1250        Q   Third  child       False  NaN  Queenstown    no  False
        # 19         1       3  female   NaN      0      0   7.2250        C   Third  woman       False  NaN   Cherbourg   yes   True

# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df[(df["age"] < 30) &  (df["sex"] =="female")].head()
        # survived  pclass     sex   age  sibsp  parch     fare embarked   class    who  adult_male deck  embark_town alive  alone
        # 2          1       3  female  26.0      0      0   7.9250        S   Third  woman       False  NaN  Southampton   yes   True
        # 8          1       3  female  27.0      0      2  11.1333        S   Third  woman       False  NaN  Southampton   yes  False
        # 9          1       2  female  14.0      1      0  30.0708        C  Second  child       False  NaN    Cherbourg   yes  False
        # 10         1       3  female   4.0      1      1  16.7000        S   Third  child       False    G  Southampton   yes  False
        # 14         0       3  female  14.0      0      0   7.8542        S   Third  child       False  NaN  Southampton    no   True

# Görev 10: Fare'i 500'den büyük veya yaşı 70’den büyük yolcuların bilgilerini gösteriniz.
df[(df["fare"] >500) | (df["age"] >70)].head()

        #  survived  pclass     sex   age  sibsp  parch      fare embarked  class    who  adult_male deck  embark_town alive  alone
        # 96          0       1    male  71.0      0      0   34.6542        C  First    man        True    A    Cherbourg    no   True
        # 116         0       3    male  70.5      0      0    7.7500        Q  Third    man        True  NaN   Queenstown    no   True
        # 258         1       1  female  35.0      0      0  512.3292        C  First  woman       False  NaN    Cherbourg   yes   True
        # 493         0       1    male  71.0      0      0   49.5042        C  First    man        True  NaN    Cherbourg    no   True
        # 630         1       1    male  80.0      0      0   30.0000        S  First    man        True    A  Southampton   yes   True


# df.loc[(df["age"] < 30) &  (df["sex"] =="female"), ["age","class"]].head()

# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
[(col,df[col].isnull().sum()) for col in df.columns]

        # Out[125]:
        # [('survived', 0),
        #  ('pclass', 0),
        #  ('sex', 0),
        #  ('age', 177),
        #  ('sibsp', 0),
        #  ('parch', 0),
        #  ('fare', 0),
        #  ('embarked', 2),
        #  ('class', 0),
        #  ('who', 0),
        #  ('adult_male', 0),
        #  ('deck', 688),
        #  ('embark_town', 2),
        #  ('alive', 0),
        #  ('alone', 0)]


# Görev 12: who değişkenini dataframe’den çıkarınız.
df2=df.drop("who", axis=1).head()

# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
df2=df
for col in df.columns:
    num = df2[col].mode()[0]
    df2[col].fillna(num, inplace=True)
df2.head()
[(col,df2[col].isnull().sum()) for col in df2.columns]  #check whether or not all null values are filled


# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df2=df
num = df2['age'].mean()
df2['age'].fillna(num, inplace=True)

# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby("survived").agg({"pclass":["sum","mean"],"sex":["count"]})

# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazın. Yazdığınız fonksiyonu
# kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
def add_age_flag(df):
    if df['age'] <30:
        return 1
    else:
        return 0
df['age_flag'] = df.apply(lambda df: add_age_flag(df), axis=1)
df.head()

# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("tips")
df.head()

# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerinin sum, min, max ve mean değerlerini bulunuz.
df.groupby("time").agg({"total_bill":["sum","min","max","mean"]})

        # Out[163]:
        #        total_bill
        #               sum   min    max       mean
        # time
        # Lunch     1167.47  7.51  43.11  17.168676
        # Dinner    3660.30  3.07  50.81  20.797159

# Görev 19: Day ve time’a göre total_bill değerlerinin sum, min, max ve mean değerlerini bulunuz.
df.groupby(["day","time"]).agg({"total_bill":["sum","min","max","mean"]})

        # Out[164]:
        #             total_bill
        #                    sum    min    max       mean
        # day  time
        # Thur Lunch     1077.55   7.51  43.11  17.664754
        #      Dinner      18.78  18.78  18.78  18.780000
        # Fri  Lunch       89.92   8.58  16.27  12.845714
        #      Dinner     235.96   5.75  40.17  19.663333
        # Sat  Lunch        0.00    NaN    NaN        NaN
        #      Dinner    1778.40   3.07  50.81  20.441379
        # Sun  Lunch        0.00    NaN    NaN        NaN
        #      Dinner    1627.16   7.25  48.17  21.410000

# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre
# sum, min, max ve mean değerlerini bulunuz.
df.head()
df3=df.groupby(["time","sex"]).agg({"total_bill":["sum","min","max","mean"],
                                "tip":["sum","min","max","mean"]})

# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
# df.loc[(df['total_bill']<3) | (df['total_bill']>10)].head()
# df[(df['total_bill']<3) | (df['total_bill']>10)].head()

df[(df['total_bill']<3) | (df['total_bill']>10)].loc[:,"total_bill"].mean()
        # Out[198]: 20.662378854625555

# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz.Her bir müşterinin ödediği totalbill ve
# tip in toplamını versin.
#version 1
def add(one, two):
  return one + two
df['total_bill_tip_sum'] = df.apply(lambda x :add(x["total_bill"], x["tip"]), axis=1)
df.head()
#version 2
df['total_bill_tip_sum']=df["total_bill"]+df["tip"]
df.head()

# Görev 23: Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulunuz. Bulduğunuz ortalamaların altında
# olanlara 0, üstünde ve eşit olanlara 1 verildiği yeni bir total_bill_flag değişkeni oluşturunuz.
# Kadınlar için Female olanlarının ortalamaları, erkekler için ise Male olanların ortalamaları dikkate alınacktır.
# Parametre olarak cinsiyet ve total_bill alan bir fonksiyon yazarak başlayınız. (If-else koşulları içerecek)
import seaborn as sns

df = sns.load_dataset("tips")
df.head()
df.head()

df5=df[df["sex"]=="Male"].mean()
df6=df[df["sex"]=="Female"].mean()
type(df5["total_bill"])

# df["total_bill_flag"] = df["sex"].apply(lambda x: 'True' if x == "Female" else 'False' )
df["total_bill_flag"] = df.apply(lambda x:( 0 if x["total_bill"] < df[df["sex"]=="Male"].mean()["total_bill"] else 1) if x["sex"] == "Male" else (0 if x["total_bill"]<df[df["sex"]=="Female"].mean()["total_bill"] else 1), axis=1)
df.head()

# Görev 24: total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyiniz.
df7=df.groupby(["sex","total_bill_flag"]).agg({"total_bill":["count"]})

            # total_bill
            #                             count
            # sex    total_bill_flag
            # Male   0                       95
            #        1                       62
            # Female 0                       54
            #        1                       33

# Görev 25: Veriyi total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
df['total_bill_tip_sum']=df["total_bill"]+df["tip"]
df.head()
df_new=df.sort_values('total_bill_tip_sum', ascending=False).head(30)
df_new

            #  total_bill    tip     sex  ... size total_bill_flag total_bill_tip_sum
            # 170       50.81  10.00    Male  ...    3               1              60.81
            # 212       48.33   9.00    Male  ...    4               1              57.33
            # 59        48.27   6.73    Male  ...    4               1              55.00
            # 156       48.17   5.00    Male  ...    6               1              53.17
            # 182       45.35   3.50    Male  ...    3               1              48.85
            # 197       43.11   5.00  Female  ...    4               1              48.11
            # 23        39.42   7.58    Male  ...    4               1              47.00
            # 102       44.30   2.50  Female  ...    3               1              46.80
            # 142       41.19   5.00    Male  ...    5               1              46.19
            # 95        40.17   4.73    Male  ...    4               1              44.90
            # 184       40.55   3.00    Male  ...    2               1              43.55
            # 112       38.07   4.00    Male  ...    3               1              42.07
            # 207       38.73   3.00    Male  ...    4               1              41.73
            # 56        38.01   3.00    Male  ...    4               1              41.01
            # 141       34.30   6.70    Male  ...    6               1              41.00
            # 238       35.83   4.67  Female  ...    3               1              40.50
            # 11        35.26   5.00  Female  ...    4               1              40.26
            # 52        34.81   5.20  Female  ...    4               1              40.01
            # 85        34.83   5.17  Female  ...    4               1              40.00
            # 47        32.40   6.00    Male  ...    4               1              38.40
            # 180       34.65   3.68    Male  ...    4               1              38.33
            # 179       34.63   3.55    Male  ...    2               1              38.18
            # 83        32.68   5.00    Male  ...    2               1              37.68
            # 39        31.27   5.00    Male  ...    3               1              36.27
            # 167       31.71   4.50    Male  ...    4               1              36.21
            # 175       32.90   3.11    Male  ...    2               1              36.01
            # 44        30.40   5.60    Male  ...    4               1              36.00
            # 173       31.85   3.18    Male  ...    2               1              35.03
            # 116       29.93   5.07    Male  ...    4               1              35.00
            # 155       29.85   5.14  Female  ...    5               1              34.99