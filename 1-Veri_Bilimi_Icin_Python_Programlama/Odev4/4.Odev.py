import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
type(df)


# HW4-Part1
# Görev: cat_summary() fonksiyonuna 1 özellik ekleyiniz. Bu özellik argümanla biçimlendirilebilir olsun. Var olan
# özelliği de argümanla kontrol edilebilir hale getirebilirsiniz.
# Plot (degisken degerleri cizimi)ve Unique Deger Sayisi (print edildi Return Series of unique values in the object.
# Includes NA values.
def cat_summary(dataframe, col_name, plot=False, count=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
    if count:
        print(dataframe[col_name].nunique())


cat_summary(df, "sex", plot=True, count=True)


# HW4-Part2
# Görev: check_df(), cat_summary() fonksiyonlarına 4 bilgi (uygunsa) barındıran numpy tarzı docstring
# yazınız. (task, params, return, example)

#CHECK_DF FUNCTION
def check_df(dataframe, head=5):
    """
    This function prints given dataframe shape,types of variables, head,tail, Null values in variables,
    and calculates the quantile of the values in a given axis.
    :param dataframe: pandas.core.frame.DataFrame
    :param head: int
    :return: function will print  dataframe shape,types of variables, head,tail, Null values in variables,
    and quantile of the values in range [0, 0.05, 0.50, 0.95, 0.99, 1]
    """

    print("###################Shape####################")
    print(dataframe.shape)
    print("###################Types####################")
    print(dataframe.dtypes)
    print("###################Head#####################")
    print(dataframe.head(head))
    print("###################Tail#####################")
    print(dataframe.tail(head))
    print("###################NA#######################")
    print(dataframe.isnull().sum())
    print("################Quantiles###################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

#Not: quantile fonksiyonu bool tipindeki degiskenlerde calismaz. bunun icin astype ile donusum yapilir.
#Bool type degiskenleri quantile fonskiyonuya calistirmaya denenince TypeError alinir. Asagida cozum
#yollari var 1-astype 2-for dongusuyle kontrol

#Cozum:1 Bool degiskenleri astype ile integere cevrilir
for col in df.columns:
    if str(df[col].dtype) in ["bool"]:
        df[col]=df[col].astype(int)


# df["adult_male"]=df["adult_male"].astype(int)
# df["alone"]=df["alone"].astype(int)
df.head()
check_df(df,head=5)

#Cozum:1 Bool,object,category degiskenleri filtre edilerek check_df fonksiyonu calistirilir
for col in df.columns:
    if str(df[col].dtype) not in ["category", "object","bool"]:
        print(col,"-------------------------------")
        check_df(df[col], head=5)



#CAT_SUMMARY FUNCTION WITH TWO ADDTIONAL PARAMETERS
def cat_summary(dataframe, col_name, plot=False, count=False):
    """
    Cat_summary function gives information about values, values counts and class ratio for given variable
    Parameters
    ----------
    dataframe:pandas.core.frame.DataFrame
    col_name :string
    plot:bool
    count:int

    Returns:
        Function plot variables elements and print number of variable classes
    -------

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
    if count:
        print(dataframe[col_name].nunique())


cat_summary(df, "sex", plot=True, count=True)