# VBO 8. Donem 
# Egitim Kurumu [VBO](https://www.veribilimiokulu.com/) 
## Hazirladigim calismalar
### 2-CRM-Analizi

CRM (Müşteri ilişkileri yönetimi) firmaların satışı, karlılıklarını artırmak için kullandıkları yöntemlerdir. Müşterinin ihtiyacını analiz etmek, müşteriyi anlamak, var olan müşteriyi korumak, sadık müşteriler elde etmek için yapılan tüm çalışmalara verilen isimdir.

### RFM Analizi
Müşteri analitiği alanında sık kullanılan bir yöntemdir. Firmalar tarafından müşterileri segmente ayırarak, her bir müşteriye özel firmanın satış ve pazarlama planları yapabilmesini sağlar. R-ecency, F-requency, M-onetary
Recency(Yenilik): Müşterinin son satın almasından bugüne kadar geçen süre.
Recency = Bugünün tarihi — Son satın alma tarihi.
Bugünün tarihi olarak belirttiğimiz kısım analizin yapıldığı tarih olarak kabul edilmektedir.
Frequency(Sıklık): Müşterinin toplam satın alma sayısı.
Monetary(Parasal Değer): Müşterinin yaptığı toplam harcama.
RFM analizi için öncelikle müşterilerinin R, F, M değerlerini belirlememiz gerekir. Ve bu değerlere göre müşterileri segmentlere ayırmamız gerekir. Daha sonra müşterinin bulunduğu segmente göre, belirli pazarlama satış yöntemleri belirlemek gerekmektedir.

### CLTV (Customer Lifetime Value)
Bir markanın ömür boyu müşteri için değeri anlamına gelmektedir. Müşteriler hakkında bilgi sahibi olmamızı sağlar. Bir müşterinin bir şirketle kurduğu ilişki-iletişim süresince bu şirkete kazandıracağı parasal değerdir. RFM analizinde müşterileri segmentlere ayırmıştık. Ancak bu segmentlerden sadece özel stratejiler, pazarlama stratejileri belirleyebiliyorduk. Geleceğe yönelik bir tahmin yapmıyorduk. Yani CLTV ile müşterilerimizi daha geniş bir perspektiften yani zaman projeksiyonuyla bize ne kadar katma değer sağlayabileceğini hesaplayabiliyoruz.
CLTV = (Customer Value / Churn Rate) * Profit Margin
Amacımız, bir veri setinde bütün kitlenin satın alma davranışını yakalayıp, bunu bireysel özellikler geldiğinde tahminde bulunabilmesini sağlamaktır.
Bu işlem için olasılıksal yöntemler kullanacağız.
CLTV = (BG/NBD Model) * (Gamma Gamma Submodel)
BG/NBD Model (Expected Number of Transaction), satın alma sayısını olasılıksal olarak ifade etmemizi sağlar.
Gamma Gamma Submodel (Conditional Expected Average Profit) ortalama kar miktarını olasılıksal olarak ifade etmemizi sağlar
