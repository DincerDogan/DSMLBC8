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
df_ = pd.read_csv("0_DATASETS\Telco-Customer-Churn.csv")
df=df_.copy()
df.info()
df.describe().T
# df_.isnull().values.any()
# df.isin([0]).any().any()
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

# TotalCharges değişkenini gerekli duzenlemeler yapilir.

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip())

# df['TotalCharges'].replace([' '], '0.0', inplace=True)
# df["TotalCharges"] = df["TotalCharges"].astype(float)

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

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

df[num_cols].head()
df[num_cols].describe().T

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col,plot=False)
    # plt.pause(5)


# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin
# ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

df["_Churn"] = np.where(df["Churn"] == "Yes", 1, 0)
for col in cat_cols:
    print(df.groupby(col)["_Churn"].mean(), "\n")
df.drop("_Churn", inplace=True, axis=1)

df.groupby("Churn")[num_cols].mean()
df.head()

# def target_summary_with_num(dataframe, target, numerical_col):
#     print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
#
# for col in num_cols:
#     target_summary_with_num(df, "Churn", col)

# Adım 5: Aykırı gözlem analizi yapınız.

df[num_cols].describe([.25, .5, .75, .90, .95, .99]).T

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95): # Aykırı değişkenler
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name): # Aykırı değer var mı yok mu ? Check
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col)) # Aykırı gözlem bulunmamaktadır.

# Adım 6: Eksik gözlem analizi yapınız.
df.isnull().values.any()
df.isnull().sum()
df.notnull().sum()
df.shape
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# df["TotalCharges"].isnull().values.any()====> True
# df["tenure"][df[df["TotalCharges"].isnull()].index]
# df["Churn"][df[df["TotalCharges"].isnull()].index]
# df.isnull().values.sum()

# eksik deger durumu
#               n_miss  ratio
# TotalCharges      11  0.160


# Adım 7: Korelasyon analizi yapınız.
df.corr()
# cor = df.corr(method='pearson')
# cor
# sns.heatmap(cor)
# plt.show()

def sns_heatmap(dataset,color):
    heatmap =sns.heatmap(dataset.corr(),vmin=-1,vmax=1, cmap=color)
    heatmap.set_title('Correlation Heatmap',fontdict={'fontsize':12},pad=12)
    plt.show()
    plt.pause(5)

sns_heatmap(df,color='Blues')

######################################################################################################
# Görev 2 : Feature Engineering
######################################################################################################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.

# df["TotalCharges"][df["TotalCharges"]==0].to_list()
# df.index[df["tenure"]==0]
df["TotalCharges"].isnull().values.sum()    #11 adet null degeri var
df["TotalCharges"] = df["TotalCharges"].fillna(0) #yeni musteri oldugu dusunulup 0 ile charge edilir


for col in num_cols:
    print(col, check_outlier(df, col))

    # Numeric değişkenlere bakıldığı zaman herhangi bir aykırı değer gözlenmemektedir.
    # tenure False
    # MonthlyCharges False
    # TotalCharges False

# Numeric değişkenlere bakıldığı zaman herhangi bir aykırı değer gözlenmemektedir.

# Adım 2: Yeni değişkenler oluşturunuz.
df2=df.copy()
df2.loc[(df2["StreamingTV"] == "Yes") & (df2["StreamingMovies"] == "Yes"), ["Streaming_N"]] = "Yes"
df2.loc[~((df2["StreamingTV"] == "Yes") & (df2["StreamingMovies"] == "Yes")), ["Streaming_N"]] = "No"


df2.loc[(df2["OnlineSecurity"] == "Yes") & (df2["OnlineBackup"] == "Yes"), ["Online_N"]] = "Yes"
df2.loc[~((df["OnlineSecurity"] == "Yes") & (df2["OnlineBackup"] == "Yes")), ["Online_N"]] = "No"

# df["Number_AdditionalServices"] = (df[["OnlineSecurity","DeviceProtection", "StreamingMovies","TechSupport","StreamingTV",
#                                        "OnlineBackup"]] == "Yes").sum(axis=1)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
# cat_cols, num_cols, cat_but_car = grab_col_names(df2)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df2.columns if df2[col].dtype not in [int, float] and df2[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df2, col)

df2.head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df2.columns if 10 >= df2[col].nunique() > 2]
df2 = one_hot_encoder(df2, ohe_cols)
df2.head()

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
mms = MinMaxScaler()
df2[num_cols] = mms.fit_transform(df2[num_cols])

# Adım 5: Model oluşturunuz.
# df2['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')

y = df2["Churn"]
X = df2.drop(["customerID", "Churn"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Out[35]: 0.7841930903928065