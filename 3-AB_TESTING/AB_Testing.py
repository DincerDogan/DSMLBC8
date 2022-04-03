import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#
# Veri Seti Hikayesi
# Popüler bir web sitesi kısa bir süre önce maximum bidding adı verilen teklif vermetürüne alternatif
# olarak yeni bir teklif türü olan average bidding’i tanıttı.Müşterilerimizden birisi ise bu yeni özelliği
# test etmeye karar verdi ve averagebidding’in maximum bidding’den daha fazla dönüşüm getirip getirmediğini
# anlamakiçin bir A/B testi yapmak istiyor. Müşterimizin web site bilgilerini içeren bu veri
# setindekullanıcıların gördükleri ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra buradangelen
# kazanç bilgileri yer almaktadır. Kontrol ve Test grubu olmak üzere iki ayrı verisetimiz vardır.

# Impression: Reklam görüntülenme sayısı
# Click: Tıklama, Görüntülenen reklama tıklanma sayısı
# Purchase: Satın alım, Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Kazanç, Satın alınan ürünler sonrası elde edilen kazanç


#######################################################################################
# Görev 1: Veriyi Hazırlama ve Analiz Etme
#######################################################################################
# Adım 1: ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı
# değişkenlere atayınız.

df_c = pd.read_excel("0_DATASETS/ab_testing.xlsx", sheet_name="Control Group")
df_t = pd.read_excel("0_DATASETS/ab_testing.xlsx", sheet_name="Test Group")
df_control = df_c.copy()
df_test = df_t.copy()

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

df_control.head(10)
df_test.head(10)

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
print("---------------Control Group------------------------")
check_df(df_control)
print("----------------Test Group-----------------------")
check_df(df_test)

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test
# grubu verilerini birleştiriniz.
df_c_t=pd.concat([df_control, df_test], ignore_index=True)
df_c_t.head(4)


#######################################################################################
# Görev 2: A/B Testinin Hipotezinin Tanımlanması
#######################################################################################
# Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2  iki grup arasında istatistiksel olarak anlamlı bir farkyoktur
# H1 : M1!= M2  iki grup arasinda istatiksel fark vardir.

print("# H0 : M1 = M2  iki grup arasında istatistiksel olarak anlamlı bir farkyoktur")
print("# H1 : M1!= M2  iki grup arasinda istatiksel fark vardir. ")

# Adım 2: Kontrol ve test grubu için purchase (kazanç) ortalamalarını analiz ediniz.
print(r'-----------df_control["Purchase"].mean()------------------------')
df_control["Purchase"].mean()

print(r'-----------df_test["Purchase"].mean()------------------------')
df_test["Purchase"].mean()

# İki grubun da Purchase yani tıklanan reklamlar sonrası satın alınan ürün sayı sayısının ortalamasını
# bu sekilde gozlemlenir.

#######################################################################################
# Görev 3: Hipotez Testinin Gerçekleştirilmesi
#######################################################################################
# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# Bunlar Normallik Varsayımı ve Varyans Homojenliğidir. Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz.


# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ? Elde edilen p-value değerlerini yorumlayınız.


# Varyans Homojenliği :
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
# Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.

############################
# Normallik Varsayımı
############################
test_stat, pvalue = shapiro(df_control["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

        # Test Stat = 0.9773, p-value = 0.5891

test_stat, pvalue = shapiro(df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

        # Test Stat = 0.9589, p-value = 0.1541

        # H0: Normal dağılım varsayımı sağlanmaktadır.
        # P-value değeri < 0.05 ise H0 reddedilir, değilse kabul edilir ve dağılım varsayımısağlanmaktadır
        # denir. Normallik sağlanıyorsa bağımsız iki örneklem T testi yapılırkennormallik sağlanmaz ise
        # mannwithneyu testi yapılmaktadır.

        # Normallik varsayımı için gruplarımıza shapiro testini uyguladıktan sonra görüyoruz kiiki
        # grubumuz içinde p-value değeri 0.05 ten büyük çıkmıştır. Bu durumda H0'ıreddedemeyiz ve normal
        # dağılım varsayımı sağlanmaktadır deriz.


############################
# Varyans Homojenligi Varsayımı
############################

test_stat, pvalue = levene(df_control["Purchase"],df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

        # Test Stat = 2.6393, p-value = 0.1083
        # H0: Varyanslar homojendir.

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.

    # Varsayımlar sağlanıyorken bağımsız iki örneklem t testi(parametrik test) uygulanırken sağlanmaz ise
    # mannwhitneyu(non-parametrik test) uygulanmaktadır. Burada varsayımlar sağlandığı için parametrik test
    # olan bağımsız iki örneklem t testini uygulayacağız.

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın
# alma ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

        # Bağımsız iki örneklem T testi
        # H0: İki grup ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.
        # H1: İki grup ortalamaları arasında istatistiksel olarak anlamlı bir fark vardır.

test_stat, pvalue = ttest_ind(df_test["Purchase"],df_control["Purchase"],equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

        # Test Stat = 0.9416, p-value = 0.3493

        # Testlerimizde normal dağılım ve varyansların homojen olduğunu gördükten sonra Bağımsız iki
        # örneklem  t testine geçtik burada p-value değeri > 0.05 olduğuna göre yine h0 reddedemeyiz ve bu
        # iki grup arasında istatistiksel olarak anlamlı bir fark yoktur deriz.


#######################################################################################
# Görev 4: Sonuçların Analizi
#######################################################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.


# Testlerimizde normal dağılım ve varyansların homojen olduğunu gördükten sonra Bağımsız iki
# örneklem  t testine geçtik burada p-value değeri > 0.05 olduğuna göre yine h0 reddedemeyiz ve bu
# iki grup arasında istatistiksel olarak anlamlı bir fark yoktur deriz.

        ######################################################
        # AB Testing (Bağımsız İki Örneklem T Testi)
        ######################################################

        # 1. Hipotezleri Kur
        # 2. Varsayım Kontrolü
        #   - 1. Normallik Varsayımı
        #   - 2. Varyans Homojenliği
        # 3. Hipotezin Uygulanması
        #   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
        #   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
        # 4. p-value değerine göre sonuçları yorumla
        # Not:
        # - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
        # - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

    # Bu bilgiler dahilinde web sitesi müşterimize biraz daha zaman geçtikten sonra ve örneklem sayısı
    # arttıkça daha iyi bir örüntü elde edebileceğimizi ve bu modeli bir süre daha test etmemizin uygun
    # olacağını tavsiye  edebiliriz.

