import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import warnings
warnings.simplefilter(action = "ignore")

#Display options
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)

######################################################################################################
# Görev 1 : Keşifçi Veri Analizi
######################################################################################################

# Adım 1: Genel resmi inceleyiniz.
df_ = pd.read_csv("0_DATASETS/diabetes.csv")
df=df_.copy()
df.info()
df.describe().T

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
# cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
## num_cols = [col for col in df.columns if df[col].dtypes != 'O']

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

cat_cols

        # Out[32]: ['Outcome'] Outcome numerik olmasina ragmen tekrar eden deger oldugu icin kategori
# grubunda ele alinir

num_cols

        # Out[33]:
        # ['Pregnancies',
        #  'Glucose',
        #  'BloodPressure',
        #  'SkinThickness',
        #  'Insulin',
        #  'BMI',
        #  'DiabetesPedigreeFunction',
        #  'Age']

# Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95): # Aykırı değişkenler
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low, up = outlier_thresholds(df,"Insulin")

def check_outlier(dataframe, col_name): # Aykırı değer var mı yok mu ? Check
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

# Outliers

def grab_outliers(dataframe, col_name, index=False): # Outliers'ları verir.
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

insulin_index = grab_outliers(df,"Insulin", True) # 13, 228


# Değerleri baskılama yöntemi kullanarak threshold değerlerine atayalım.

def replace_with_thresholds(dataframe, variable): # Aykırı değerleri eşik değere atayalım.
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    print(col, check_outlier(df,col))

# Adım 6: Eksik gözlem analizi yapınız.
# Eksik gözlem bulunmadığını görüyoruz fakat içinde 0 olmaması gereken değişkenler için değerlerin
# sıfır olduğunu gördük, bu değişkenler üzerinde işlem yapabilmek için önce 0 değerlerini nan
# yapıp ardından ortalama ile dolduralım.

df['Glucose']=df['Glucose'].replace(0,np.nan)
df['BMI']=df['BMI'].replace(0,np.nan)
df['SkinThickness']=df['SkinThickness'].replace(0,np.nan)
df['Insulin']=df['Insulin'].replace(0,np.nan)
df['BloodPressure']=df['BloodPressure'].replace(0,np.nan)

#Eksik degerler yerine ortalama degerler atanir
df['Glucose']=df['Glucose'].fillna(df['Glucose'].mean())
df['BMI']=df['BMI'].fillna(df['BMI'].mean())
df['SkinThickness']=df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin']=df['Insulin'].fillna(df['Insulin'].mean())
df['BloodPressure']=df['BloodPressure'].fillna(df['BloodPressure'].mean())

df.isnull().values.any()

# Adım 7: Korelasyon analizi yapınız.

def sns_heatmap(dataset,color):
    heatmap =sns.heatmap(dataset.corr(),vmin=-1,vmax=1, cmap=color)
    heatmap.set_title('Correlation Heatmap',fontdict={'fontsize':12},pad=12)
    plt.show()
    plt.pause(5)

sns_heatmap(df,color='Blues')

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (10, 10)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True)

drop_list = high_correlated_cols(df)

high_correlated_cols(df.drop(drop_list, axis=1), plot=True)


######################################################################################################
# Görev 2 : Feature Engineering
######################################################################################################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta
# ama Glikoz, Insulin vb.  değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini
# ilgili değerlerde NaN olarak atama yapıp sonrasında eksik  değerlere işlemleri uygulayabilirsiniz.
#0 sifir olmamasi gerken degerler eksik gozleme donusturulur
df.isnull().values.any()
df.isnull().sum()
df.isnull().sum().sort_values(ascending=False)
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
df.info()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns # # Verisetinde kaç eksiklik var, bunların yüzdeliği ne, hangi columnlar'da eksiklik var

missing_values_table(df)


df["Insulin"].describe(percentiles=[0.01,0.05, 0.1,0.2,0.3,0.4,0.6,0.7,.8,.9,.95,.99])

df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0) # Ortalama ile doldurma

missing_values_table(df) # None

df.info()
# Adım 2: Yeni değişkenler oluşturunuz.
#ilk olarak yas ve vucut kitle indeksini carparak yeni bir degisken olussun
# df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
# df['Glucose'] = pd.to_numeric(df['Glucose'], errors='coerce')

scaler=MinMaxScaler(feature_range=(1,5))
ages=scaler.fit_transform(df[["Age"]])
bmi=scaler.fit_transform(df[["BMI"]])
age_BMI=ages*bmi
df["age_BMI"]=pd.DataFrame(age_BMI)


#Diger degiskenler icin yas, dogum sayisi ve glikoz miktarlarini segmentlere ayiralim
df.loc[(df["Age"])<=35,"New Age"]="young"
df.loc[(df["Age"]>35) &(df["Age"]<=45),"New Age"]="middle"
df.loc[(df["Age"]>45) &(df["Age"]<=55),"New Age"]="mature"
df.loc[(df["Age"])>55,"New Age"]="old"

df.loc[(df["Pregnancies"])<=3,"New_Preg"]="lessequal_3"
df.loc[(df["Pregnancies"]>3) &(df["Pregnancies"]<=6),"New_Preg"]="lessequal_6"
df.loc[(df["Pregnancies"])>6,"New_Preg"]="morethan6"

df.loc[(df["Glucose"])<=140,"New_Glucose"]="normal"
df.loc[(df["Glucose"])>140,"New_Glucose"]="high"




cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

df.head()

binary_cols=[col for col in df.columns if df[col].dtype not in [int,float] and df[col].nunique()==2]

def label_encoder(dataframe,binary_col):
    label_encoder=LabelEncoder()
    dataframe[binary_col]=label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df,col)


# Adım 3: Encoding işlemlerini gerçekleştiriniz.
# İkiden daha fazla string sınıf içeren değişkenleri de onehot encoder kullanarak ve ilk dummy
# değişkenini silerek (dummy değişken tuzağına düşmemek için) dönüştürüyoruz. Oluşan yeni
# değişkenlerden sonra veri setini tekrar kategorik, numerik ve kardinal olarak tekrar
# ayırıyoruz.
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df,ohe_cols)
#olusan yeni degiskenlerle birlikte tekrar gruplar guncellenir
df.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
mms = MinMaxScaler()
df[num_cols] = mms.fit_transform(df[num_cols])
df.head()

#alternative2
# scaler=StandardScaler()
# df[num_cols]=scaler.fit_transform(df[num_cols])

# Adım 5: Model oluşturunuz.
# Model için hedef bağımlı değişkeni tanıttık ve veri setini eğitim ve test olarak ayırdık.
# Train ve test setini 70'e 30 olarak ayarladık.
df['Outcome'] = pd.to_numeric(df['Outcome'], errors='coerce')
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
# Modelde random forest classifier kullanarak tahminleme yaptık.
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
# Random forest skoru Out[23]: 0.7532467532467533

rf_model.feature_importances_

X_train.columns

# hangi değişkenin modelde tahminde daha çok etki ettiğini bulma
# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model,features,num=len(X),save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
        plt.pause(5)


plot_importance(rf_model, X_train)