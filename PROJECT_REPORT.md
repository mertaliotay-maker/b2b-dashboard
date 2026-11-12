# B2B Sales Forecasting and Inventory Analysis

ISL467E HW1 kapsamında, B2B dağıtıcı işlem verisiyle geliştirilen satış tahmini ve stok politikası analizi.

## Veri

Kaynak veri `data/B2B_Transaction_Data_.xlsx` dosyasıdır. Bir satır fatura ürün kalemidir. Ortak `load_data()` fonksiyonu 2 Ocak-31 Aralık 2020 döneminde 657.855 temiz satır döndürür. Veri 43.575 ürün, 47.467 fatura, 1.878.778 adet satış ve 29.417.403,34 TL net ciro içerir. Net ciro, adet ile indirim sonrası birim fiyatın çarpımıdır.

## Tahmin yöntemi

Kısmi sınır haftaları eğitim dışında tutularak 51 tam Pazartesi-Pazar haftası oluşturulur. İç dönemde işlemsiz haftalar sıfır gözlenen satış kabul edilir. Kesintisiz veri kapsamı ve ürün mevcudu varsayımdır; stok yokluğu ile eksik veri aktarımı ayırt edilemez.

İlk 43 hafta geliştirme, son 8 hafta testtir. Geliştirme içinde 27 ve 35 haftalık eğitim pencerelerini izleyen iki 8 haftalık doğrulama uygulanır. Seçim ölçütü doğrulama RMSE'sidir. Son test model seçiminde ve tahmin aralığı kalibrasyonunda kullanılmaz.

Naive, son dört hafta ortalaması, sönümlenen Holt, ARIMA(1,1,1) ve gecikmeli Random Forest karşılaştırılır. RF, geçmiş 1/2/3/4/8/13 haftalık satışları, geçmiş hareketli istatistikleri ve zaman sırasını kullanır; çok adımlı tahminleri kendi çıktılarıyla ilerletir. Tek yıllık geçmişte yıllık mevsimsellik doğrulanamadığı için Holt mevsimsiz kullanılır.

| Hedef | Seçilen yöntem | Doğrulama RMSE | Test RMSE | Test MAPE |
|---|---|---:|---:|---:|
| Haftalık adet | Naive | 14.806 adet | 2.908 adet | %6,14 |
| Haftalık net ciro | Naive | 250.197 TL | 55.927 TL | %6,86 |

MAPE sıfır gerçekleşmeleri dışlar; geçerli gözlem sayısı ayrıca raporlanır. Temmuz sonundaki işlemsiz/çok düşük satışlı dönem doğrulama MAPE'sini yükselttiğinden RMSE ve WAPE birlikte değerlendirilir. Düşük test hatası yalnızca son sekiz haftaya ilişkindir.

3/6/12 aylık seçenekler yaklaşık 13/26/52 haftadır. Yaklaşık %95 tahmin bandı, seçilen modelin 16 doğrulama hatasının `abs(error)/sqrt(h)` değerlerinin %95 niceliğiyle hesaplanır ve ufkun kareköküyle genişletilir. Küçük ve bağımlı kalibrasyon örneklemi kapsama garantisi sağlamaz. Son testte gözlenen kapsama %100'dür. Sekiz haftalık değerlendirme, uzun ufuktaki doğruluğu doğrulamaz. İlk tahmin haftası 28 Aralık 2020-3 Ocak 2021 olduğundan kısmi son gerçek hafta ile örtüşür.

## Stok politikası

Varsayılan pencere 3 Ekim-31 Aralık 2020 arasındaki 90 gündür. Ürünler `StockCode` ile birleştirilir. ABC, net cirodaki birikimli paya göre yaklaşık %80/%15/%5 ayrımıdır; eşiği geçen ürün önceki sınıfta kalır. XYZ tam haftalardaki adet satışlarının varyasyon katsayısını kullanır: X için CV ≤0,5, Y için 0,5<CV≤1, Z için CV>1. Bu eşikler iş kuralıdır. Yeterli haftalık talep istatistiği bulunmayan ürünler sınıflandırılamamış olarak gösterilir.

Son 90 günde 32.581 ürünün 3.531'i A sınıfındadır ve cironun %80,003'ünü oluşturur. 2.097 AZ ürünü toplam cironun %31,40'ını sağlar; tedarikçi teyidi ve sık stok gözden geçirme için öncelikli gruptur.

Günlük ortalama ve örnek standart sapma sıfır satışlı günleri içerir. Güvenlik stoğu ve yeniden sipariş noktası:

```text
SS = z * sqrt(L * sigma_d^2 + mu_d^2 * sigma_L^2)
ROP = ceil(L * mu_d + SS)
```

`L` ortalama teslim süresi, `sigma_L` teslim süresi sapması, `mu_d` günlük ortalama talep, `sigma_d` günlük talep sapmasıdır. Varsayılanlar 14 takvim günü teslim süresi, sıfır teslim süresi sapması ve %95 çevrim hizmet düzeyidir. Talep ve teslim süresinin bağımsızlığı ile durağanlık varsayılır. Kesikli talep ve kısa geçmişler manuel incelemeye işaretlenir. Veri stok, açık sipariş veya teslim tarihleri içermediğinden gerçek sipariş miktarı ve gecikme performansı hesaplanmaz.

## Çalıştırma ve dosya yapısı

Python 3.12 ile proje kökünde:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Notebook için:

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyterlab
```

`sales_forecasting_inventory.ipynb` dosyası tüm hücreler sırayla çalıştırılarak yeniden üretilebilir. Notebook ve `pages/forecast_supply.py`, `analytics/forecasting.py` ve `analytics/inventory.py` fonksiyonlarını paylaşır. Uygulama temizlenmiş ve ortak filtrelerden geçmiş veriyi `render(df)` fonksiyonuna aktarır. Pahalı hesaplar `@st.cache_data` ile önbelleklenir. Notebook'un ürettiği CSV ve grafikler `results/` klasörüne yazılır. Bu üretilen dosyalar uygulamanın çalışması için gerekli değildir ve sürüm kontrolü dışında tutulur.

## Doğrulama

```bash
python -m unittest discover -s tests -v
python tests/smoke_dashboard.py
```

Analitik testler zaman sızıntısı, hata ölçüleri, sıfır talep dönemleri, ürün kimliği, kısa geçmişler ve stok formüllerini sınar. Arayüz testi dört sayfa, tahmin ufukları, ciro hedefi, stok varsayımları ve dar/boş filtreleri değerlendirir. Test edilen ortam Windows ve Python 3.12'dir. Notebook'un 12 kod hücresi, 9 analitik test ve 15 yerel arayüz senaryosu başarıyla çalıştırılmıştır. Yeni bölümün Streamlit Cloud doğrulaması yayımdan sonra ayrıca yapılmalıdır.

## Kaynaklar

- `B2B_Transaction_Data_.xlsx` ve `ISL467E_HW1_2026.pdf`.
- [Proje deposu](https://github.com/mertaliotay-maker/b2b-dashboard).
- [statsmodels ExponentialSmoothing](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html) ve [ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html).
- [scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/toolbox.html).
