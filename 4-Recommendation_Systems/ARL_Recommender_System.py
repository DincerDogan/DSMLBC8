# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

#######################################################################
# Görev 1: Veriyi Hazırlama
#######################################################################

### Adım 1: armut_data.csv dosyasını okutunuz.

df_ = pd.read_csv("0_DATASETS/armut_data.csv")

df = df_.copy()
df.head()

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri
# temsil edecek yeni bir değişken oluşturunuz. Elde edilmesi gereken çıktı:

df["Hizmet"] = df['ServiceId'].astype(str) +"_"+ df["CategoryId"].astype(str)

### Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır. Association Rule
# Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir müşterinin aylık aldığı
# hizmetlerdir. Örneğin; 25446 id'li müşteri 2017'in 8.ayında aldığı 4_5, 48_5, 6_7, 47_7 hizmetler bir sepeti; 2017'in 9.ayında aldığı 17_5, 14_7
# hizmetler başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren
# yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
# Elde edilmesi gereken çıktı:
df["New_Date"]=pd.DatetimeIndex(df['CreateDate']).year.astype(str)+"-"+pd.DatetimeIndex(df['CreateDate']).month.astype(str)
df["SepetID"] = df['UserId'].astype(str) +"_"+ df["New_Date"].astype(str)



#######################################################################
# Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
#######################################################################

### Adım 1: Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.
# Alternative1
df["value"]=1
df_gr_matrix1=pd.pivot_table(df, values="value", index=["SepetID"], columns="Hizmet", fill_value=0)

# Alternative2
df_gr_matrix2 =df.groupby(["SepetID","Hizmet"]).agg({"Hizmet": "count"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

df[df["Hizmet"] == '0_8'][["UserId"]].values[0:5].tolist()

### Adım 2: Birliktelik kurallarını oluşturunuz.
# Her bir itemin support değerlerini elde ettik.
support_items = apriori(df_gr_matrix2, min_support=0.01, use_colnames=True) # Min support alabilecek minimum support eşik değerini belirler.
support_items.sort_values("support", ascending=False).head(50)

# Confidence, lift değerlerini bulmak için yukarıda apriori ile bulduğumuz support değerlerini kullanarak association rule uyguladık.
rules = association_rules(support_items, metric='support', min_threshold=0.01)
rules.sort_values("support", ascending=False).head(50)

### Adım3: arl_recommender fonksiyonunu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
# Recommendation için script yazalım.


def arl_recommender(rules_df, hizmet, rec_count=1):
    print(rules_df, hizmet, rec_count)
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendations = []
    # print(sorted_rules)
    for k, product in sorted_rules["antecedents"].items():
        # print('k-proudct',k,"-",product)
        for j in list(product):
            # print('j == hizmet',j,hizmet,j == hizmet)
            if j[1] == hizmet:
                recommendations.append(list(sorted_rules.iloc[k]["consequents"]))

    recommendations = list(dict.fromkeys({item for item_list in recommendations for item in item_list})) # Tekrar eden itemleri tekilleştirme.
    return recommendations[:rec_count]

arl_recommender(rules, '15_1',3)