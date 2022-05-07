
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Outliers bulma.
# Aykırı değerlere ulaşma.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer var mı yok mu check.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı değerleri verilen ayarlanan alt ve üst limitle baskılama.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



#Datanin okunmasi
df = pd.read_csv("0_DATASETS/hitters.csv")
df.head()


# Dataya genel bir bakis
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


# Degisken turlerini (Kategorik-numerik-kategorik-kardinal) verilen thresholda göre bulma.
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
        car_th: int, optinal
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
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

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

# Numerik ve kategorik değişkenlerin analizi

#Numerik degiskenler
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.01, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df,col)

#Categorik degiskenler
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df,col)



df.corr()


# Eksik degerlerin kontrolu
def missing_values_table(dataframe, na_name=False): # Verisetinde kaç eksiklik var, bunların yüzdeliği ne, hangi columnlar'da eksiklik var
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True) #Salary de esksik degerler var

# Verilen eksik bir değişkenin target değişken ile olan ortalaması ve sayısını verir.
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Salary", na_columns)

# Salary değişkeni bağımlı değişken, onu doldurmak modelde bi yanlılık yaratabilir.
# Bu yüzden bu null değerleri datasetten cıkarmak ve öyle modele sokmak mantıklıdır.



# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# eksik değer kontrolü
df.isnull().values.any()
df.dropna(inplace = True)
df.head()

df.isnull().values.any() #simdi na degerleri tamam.


# Feature Engineering----------------------------------------------------------------------------

df.columns = [col.upper() for col in df.columns]

# 86-87 sezonundaki yapmış olduğu oyunların kariyerlerindeki yeri.
df["ATBAT_DIV_CAREER"] = df["ATBAT"] / df["CATBAT"]
df["HITS_DIV_CAREER"] = df["HITS"] / df["CHITS"]
df["HMRUN_DIV_CAREER"] = df["HMRUN"] / df["CHMRUN"]
df["RUNS_DIV_CAREER"] = df["RUNS"] / df["CRUNS"]
df["RBI_DIV_CAREER"] = df["RBI"] / df["CRBI"]
df["WALKS_DIV_CAREER"] = df["WALKS"] / df["CWALKS"]

# 86-87 sezonunda yaptığı isabetli vuruşların tümüne oranı.
df["HITS_DIV_ATBAT"] = df["HITS"] / df["ATBAT"]
# 86-87 sezonunda yaptığı en değerli vuruşların tümüne oranı.
df["HMRUN_DIV_ATBAT"] = df["HMRUN"] / df["ATBAT"]
# 86-87 sezonunda yaptığı vuruşların ne kadarını takıma sayı olarak kazandırdığı.
df["RUNS_DIV_ATBAT"] = df["RUNS"] / df["ATBAT"]
# 86-87 sezonunda yaptığı vuruşların ne kadarı karşı oyuncuyu hataya sürükledi.
df["WALKS_DIV_ATBAT"] = df["WALKS"] / df["ATBAT"]


# Oyuncunun oynadığı yıl başına kariyerindeki toplam vuruş sayısı
df["CATBAT_DIV_YEARS"] = df["CATBAT"] / df["YEARS"]
# Oyuncunun oynadığı yıl başına kariyerindeki toplam isabet sayısı
df["CHITS_DIV_YEARS"] = df["CHITS"] / df["YEARS"]
# Oyuncunun oynadığı yıl başına kariyerindeki toplam en değerli vuruş sayısı
df["CHMRUN_DIV_YEARS"] = df["CHMRUN"] / df["YEARS"]
# Oyuncunun oynadığı yıl başına kariyerindeki toplam takıma kazandırdığı sayı
df["CRUNS_DIV_YEARS"] = df["CRUNS"] / df["YEARS"]
# Oyuncunun oynadığı yıl başına kariyerindeki toplam karşı takıma hata yaptırdığı sayı
df["CWALKS_DIV_YEARS"] = df["CWALKS"] / df["YEARS"]



#  Oyuncunun tüm sezonlar yaptığı isabetli vuruşların tümüne oranı.
df["CHITS_DIV_ATBAT"] = df["CHITS"] / df["CATBAT"]
#  Oyuncunun tüm sezonlar yaptığı en değerli vuruşların tümüne oranı.
df["CHMRUN_DIV_ATBAT"] = df["CHMRUN"] / df["CATBAT"]
#  Oyuncunun tüm sezonlar yaptığı vuruşların ne kadarını takıma sayı olarak kazandırdığı.
df["CRUNS_DIV_ATBAT"] = df["CRUNS"] / df["CATBAT"]
# Oyuncunun tüm sezonlar yaptığı vuruşların ne kadarı karşı oyuncuyu hataya sürükledi.
df["CWALKS_DIV_ATBAT"] = df["CWALKS"] / df["CATBAT"]

# Player junior, middle, senior durumu.
df.loc[(df["YEARS"] <= 3), "NEW_YEARS_LEVEL"] = "Junior_player"
df.loc[(df["YEARS"] > 3) & (df["YEARS"] <= 7), "NEW_YEARS_LEVEL"] = "Middle_player"
df.loc[(df["YEARS"] > 7), "NEW_YEARS_LEVEL"] = "Senior_player"

# E ve W (East ve Weast), playerları buna göre sınırlayabiliriz.

df.loc[(df["NEW_YEARS_LEVEL"] == "Junior_player") & (df["DIVISION"] == "E"), "NEW_DIV"] = "Junior_player_East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Junior_player") & (df["DIVISION"] == "W"), "NEW_DIV"] = "Junior_player_West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Middle_player") & (df["DIVISION"] == "E"), "NEW_DIV"] = "Middle_player_East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Middle_player") & (df["DIVISION"] == "W"), "NEW_DIV"] = "Middle_player_West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Senior_player") & (df["DIVISION"] == "E"), "NEW_DIV"] = "Senior_player_East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Senior_player") & (df["DIVISION"] == "W"), "NEW_DIV"] = "Senior_player_West"

# Oyuncuların 86-87 sezonunuda oynadıkları lig ile yeni sezonda oynadıkları lig karşılaştırılarak, lig yükselip düşme durumlarını öğrenebiliriz.
# A american ligi ve junior olarak, N ise National ligi ve senior olarak bilinir.
df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "A"), "NEW_SEASON_PROGRESS"] = "Same-A"
df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "N"), "NEW_SEASON_PROGRESS"] = "Same-N"
df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "N"), "NEW_SEASON_PROGRESS"] = "Higher"
df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "A"), "NEW_SEASON_PROGRESS"] = "Lower"

# 86-87 sezonunda oyunucun yapmış olduğu vuruş başına asist oranı.
df["ASSISTS_DIV_ATBAT"] = df["ASSISTS"] / df["ATBAT"]
# 86-87 sezonunda oyunucun yapmış olduğu vuruş başına hata oranı.
df["ERRORS_DIV_ATBAT"] = df["ERRORS"] / df["ATBAT"]

df.head()
df.shape # (263, 44)

# Eksiklik var mı kontrol edelim.
missing_values_table(df)

# missing_values_table(df)
#                   n_miss  ratio
# HMRUN_DIV_CAREER       3  1.140

df["HMRUN_DIV_CAREER"] = df["HMRUN_DIV_CAREER"].fillna(df["HMRUN_DIV_CAREER"].mean())


missing_values_table(df)

# Yeni oluşturulan değişkenlerde outlier var mı yok mu ?
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Observations: 263
# Variables: 44
# cat_cols: 6
# num_cols: 38
# cat_but_car: 0
# num_but_cat: 0

for col in num_cols:
    print(col, check_outlier(df, col))

# ASSISTS_DIV_ATBAT True
# ERRORS_DIV_ATBAT True

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# ASSISTS_DIV_ATBAT False
# ERRORS_DIV_ATBAT False


# Encoding

# Rare Encoding
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df,col)
# Gözlendiği gibi rare encoding yapmaya gerek kalmadı, çünkü yeni oluşturulan değişkenlerde
# küçük oranlara sahip değerler yok.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SALARY", cat_cols)




# One Hot Encoding Kategorik degiskenlerde
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df,cat_cols)
df

# Makineye input vermek için son işlemler yapılabilir.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if 'SALARY' not in col] # Salary değişkenini hedef değişken oldugu icin numerik değişkenlerden cıkaralım

# Son kez yeni oluşturdugumuz değişkenlerin rare olma durumlarına bakalım ve gereksiz olanları verisetinden cıkarabiliriz.
rare_analyser(df,'SALARY',cat_cols) #Ratiolar buyuk

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# Standard Scaler modele verilmek uzere Robust scallerde kullanabilirdik
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform((df[num_cols]))


# Modelimizi kuralım.

df = df.dropna(how='any',axis=0)
X = df.drop('SALARY', axis=1)
y = df[["SALARY"]]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12)

reg_model = LinearRegression()
reg_model.fit(X_train,y_train)

# bias değeri
reg_model.intercept_
# reg_model.intercept_
# Out[35]: array([329.02419201])

# weights
reg_model.coef_
# Out[34]:
# array([[-433.84780883,  582.22245943,   45.82961692,  -95.51956939,
#          -80.54522866,  203.61965503, -296.31670284, -462.1696953 ,
#          253.9670832 , -107.57131344,  266.12598557,  377.8444841 ,
#         -106.02905736,   66.10827307, -152.54244947,   74.00413925,
#          714.06475841, -567.52266624,  -12.03632192, -282.24755556,
#           66.42109495,  -80.48966487,  -62.84893495,    5.2058779 ,
#           10.97211147,  -76.8684271 , -133.16018994,  -14.01740759,
#           49.69660991,  144.93501127,   -3.80588175,   -6.84958889,
#           -5.11711552,  -49.44947533,   26.20212622,  155.45846871,
#          -61.83130088,  -38.04167968,   15.56560441,   52.07603746,
#          122.33847976,  277.19330867, -103.2484411 ,   83.29987606,
#           39.0386037 ,  197.41786686,   79.77544181,  -29.04325083,
#          -23.03278662,   -8.99842884]])

# Tahmin Başarısını değerlendirme

# Train RMSE (train verisi üzerinden tahminleme yapıldı.)
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred)) # 225.44625691679693

# Train R2
reg_model.score(X_train,y_train) # 0.7671656620625809

# Test RMSE (test verisi üzerinden tahminleme yapıldı.)
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred)) # 248.6717758266496

# Test R2
reg_model.score(X_test, y_test) # 0.5346780534177449

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error"))) # 277.93432030514543



