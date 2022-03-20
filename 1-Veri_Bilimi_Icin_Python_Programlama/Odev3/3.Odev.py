# print("Merhaba ben Dincer Dogan!")
#
# import numpy
# print(numpy.__version__)
#
# import pandas
# print(pandas.__version__)

#3Gorev1: Verilen değerlerin veri yapılarını inceleyiniz.

x=8
type(x)
y=3.2
z=8j+16
a="Hello World"
b=True
c=23<21
l=[1,2,3,4]
d={"Name":"Jake",
   "Age":27,
   "Adress":"Downtown"}
t=("Machine Learning","Data Science")
s={"Python","Machine Learning","Data Science"}

#3Gorev2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz,
# kelime kelime ayırınız.
text="The is to turn data into information, information into insight."
text=text.upper().replace(".","").replace(",","").split()
print(text)

#3Gorev3:Verilen listeye aşağıdaki adımları uygulayınız.
# Adım 1: Verilen listenin eleman sayısına bakınız.
lst="DATASCIENCE"
lst=list(lst)
print(lst)
print("Adim sayısı:{}".format(len(lst)))
# Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
print("0. ve 10. indexteki elemanları:{}-{}".format(lst[0],lst[10]))
# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
print("DATA lıstesının olusturulması: {}".format(lst[:4]))
# Adım 4: Sekizinci indeksteki elemanı siliniz.
lst.pop(8)
print("8. indexteki elemanın silinmesi:{}".format(lst))
# Adım 5: Yeni bir eleman ekleyiniz.
lst.append("D")
print("Yeni eleman eklenmesi:{}".format(lst))
# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8,"N")
print("N elemanının eklenmesi:{}".format(lst))

#3Gorev4
dict={'Christian':["America",18],
      'Daisy':["England",12],
      'Antonio':["Spain",22],
      'Dante':["Italy",25]}
# Adım 1: Key değerlerine erişiniz.
print("1.adim key degerlerine erisiniz:{}".format(list(dict.keys())))
# Adım 2: Value'lara erişiniz.
print("2.adim value degerlerine erisiniz:{}".format(list(dict.values())))
# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict['Daisy'][1]=13
print("3.adim Daisy keyıne 13 atanması:{}".format(list(dict.values())))
# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict['Ahmet']=["Turkey",24]
print("4.adim Ahmet keyli eleman atanması keyıne 13 atanması:{}".format(dict))
# Adım 5: Antonio'yu dictionary'den siliniz.
dict.pop('Antonio')
print("5.adim Antonio keyli elemanın silinmesi:{}".format(dict))

#3Gorev5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları
# ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.
l=[2,13,18,93,22]

def func_list_even_odd(liste):
    odd_list = [number for number in liste if number % 2 != 0]
    even_list = [number for number in liste if number % 2 == 0]
    return odd_list, even_list

a,b=func_list_even_odd(l)
print("Cift sayılar:{}".format(a))
print("Tek sayılar:{}".format(b))

#3Gorev6:List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini
# büyük harfe çeviriniz ve başına NUM ekleyiniz.
import seaborn as sns
df=sns.load_dataset("car_crashes")

df.columns=["NUM_"+col.upper() if df[col].dtype !="O"else col.upper()   for col in df.columns]
print(df.columns)

#3Gorev7:List Comprehension yapısı kullanarak car_crashes verisinde isminde "no"
# barındırmayan değişkenlerin isimlerinin sonuna "FLAG" yazınız.
import seaborn as sns
df=sns.load_dataset("car_crashes")

df.columns=[col.upper()+"_FLAG" if "no" not in col else col.upper()   for col in df.columns]
print(df.columns)

#3Gorev8:List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan
# değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.
import seaborn as sns
df=sns.load_dataset("car_crashes")
df.columns
og_list=["abbrev","no_previous"]

num_cols=[col for col in df.columns if col not in og_list]
# print(num_cols)
df[num_cols].head(5)