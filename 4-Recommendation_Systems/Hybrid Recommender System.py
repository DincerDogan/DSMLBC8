import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


##################################################################################
###Görev 1: Veri Hazırlama
##################################################################################

### Adım 1: movie, rating veri setlerini okutunuz.
movie = pd.read_csv("0_DATASETS/movie.csv")
rating = pd.read_csv("0_DATASETS/rating.csv")
movie.info()
movie.shape
movie.columns

rating.info()
rating.shape
rating.columns


### Adım 2: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.

df_ = rating.merge(movie, how="left", on="movieId")
df_.head()
df=df_.copy()
# Adım3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri
# setinden çıkartınız.

comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.head()
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

# Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe
# için pivot table oluşturunuz.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

# Adım5: Yapılan tüm işlemleri fonksiyonlaştırınız.

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("0_DATASETS/movie.csv")
    rating = pd.read_csv("0_DATASETS/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 10000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df=create_user_movie_df()
user_movie_df.head()


##################################################################################
###Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
##################################################################################

# Adım 1: Rastgele bir kullanıcı id’si seçiniz.

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index == random_user]

# Adım3: Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()


##################################################################################
###Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
##################################################################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve
# movies_watched_df adında yeni bir dataframe oluşturunuz.

movies_watched_df=user_movie_df[movies_watched]
movies_watched_df[user_movie_df.index == random_user].head()

# Adım 2: Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini
# bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.

user_movie_count = movies_watched_df.T.notnull().sum()

# Adım3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı
# id’lerinden users_same_movies adında bir liste oluşturunuz.

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count[user_movie_count["movie_count"] > 60].sort_values("movie_count", ascending=False)

user_movie_count[user_movie_count["movie_count"] == 33].count()

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 100]["userId"]


##################################################################################
###Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
##################################################################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların
# id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],random_user_df[movies_watched]])

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()


# final_df[final_df.index=="121739"]
# corr_df[corr_df["user_id_1"]==121739]
# corr_df[corr_df["user_id_1"]==80064]

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df.head()

corr_df = corr_df.reset_index()

corr_df.head()

# Adım3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan)
# kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.4)][
    ["user_id_2", "corr"]].reset_index(drop=True)

# 0.65 den buyuk yok

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)


# Adım4: top_users dataframe’ine rating veri seti ile merge ediniz.

rating = pd.read_csv("0_DATASETS/rating.csv")
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


##################################################################################
### Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
##################################################################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan
# weighted_rating adında yeni bir değişken oluşturunuz.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama
# değerini içeren recommendation_df adında yeni bir dataframe oluşturunuz.

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

# Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve
# weighted rating’e göre sıralayınız.

recommendation_df[recommendation_df["weighted_rating"] > 2]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 2].sort_values("weighted_rating", ascending=False)

# 3.5 den buyuk yok 2 degeri alindi

# Adım4: movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.


movie = pd.read_csv("0_DATASETS/movie.csv")
movies_to_be_recommend.merge(movie[["movieId", "title"]]).head()

            # Out[95]:
            #    movieId  weighted_rating                   title
            # 0     2337         2.450393  Velvet Goldmine (1998)
            # 1     1755         2.450393    Shooting Fish (1997)
            # 2     2295         2.450393   Impostors, The (1998)
            # 3     3181         2.450393            Titus (1999)
            # 4     3720         2.450393         Sunshine (1999)

##################################################################################
##################################################################################
# Item Based Recommendation
##################################################################################
##################################################################################

##################################################################################
# Görev 1: Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız.
##################################################################################

# Adım 1: movie, rating veri setlerini okutunuz.
movie = pd.read_csv("0_DATASETS/movie.csv")
rating = pd.read_csv("0_DATASETS/rating.csv")

# Adım 2: Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] ==  5.0)].\
sort_values(by = "timestamp", ascending = False)["movieId"][0:6].values[0]

# Adım3: User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film
# id’sine göre filtreleyiniz.
movie_name = movie[movie["movieId"]== movie_id]["title"]
movie_name = user_movie_df[movie_name]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
movies_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False)

# Adım5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.

movies_from_item_based[1:6].index

            # Out[103]: Index(['10 Things I Hate About You (1999)',
            #                  '12 Angry Men (1957)',
            #                  '2001: A Space Odyssey (1968)',
            #                  '28 Days Later (2002)',
            #                  '300 (2007)'],
            #                 dtype='object', name='title')
