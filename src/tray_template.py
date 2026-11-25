"""Tabldot tepsi şablonu: perspektif düzeltme, yön bulma ve bölme maskeleri.

Hattın 2. ve 3. adımı bu modüldedir:
  * Fotoğraftaki tepsinin 4 köşesinden homografi hesaplanır ve tepsi,
    şablonla aynı boyutta (1200x800) tam üstten görünüme çevrilir.
  * Tepsi fotoğrafta döndürülmüş olabilir. 4 olası yön denenir; kenar
    haritası şablona en çok benzeyen (en yüksek NCC) yön seçilir.
  * Elle tıklanan köşeler, şablonla hizalanacak şekilde otomatik ince ayarlanır.
  * Düzeltilmiş görüntüde her bölme, bolmeler.json'daki sabit maskeyle ayrılır.
"""
import json
from pathlib import Path

import cv2
import numpy as np

KOK = Path(__file__).resolve().parent.parent
SABLON_YOLU = KOK / "data" / "tepsi" / "sablon.png"
BOLME_YOLU = KOK / "data" / "tepsi" / "bolmeler.json"


# --------------------------------------------------------------------------
# Bölme maskeleri
# --------------------------------------------------------------------------
def bolmeleri_yukle(yol=BOLME_YOLU):
    """bolmeler.json dosyasını sözlük olarak döndürür."""
    return json.loads(Path(yol).read_text(encoding="utf-8"))


def yuvarlak_dikdortgen_maske(yukseklik, genislik, x1, y1, x2, y2, r):
    """Köşeleri r yarıçapıyla yuvarlatılmış, içi dolu dikdörtgen maske.

    Maske iki dikdörtgen (yatay ve dikey "artı" şekli) ile dört köşe
    dairesinin birleşimidir. Dönüş: 0/255 değerli uint8 dizi.
    """
    yy, xx = np.mgrid[0:yukseklik, 0:genislik]
    yatay = (xx >= x1) & (xx <= x2) & (yy >= y1 + r) & (yy <= y2 - r)
    dikey = (xx >= x1 + r) & (xx <= x2 - r) & (yy >= y1) & (yy <= y2)
    maske = yatay | dikey
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        maske |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return maske.astype(np.uint8) * 255


def bolme_maskeleri(bilgi=None, sadece_analiz=False):
    """Her bölme için {ad: maske} sözlüğü döndürür."""
    bilgi = bilgi or bolmeleri_yukle()
    genislik, yukseklik = bilgi["boyut"]
    maskeler = {}
    for ad, b in bilgi["bolmeler"].items():
        if sadece_analiz and not b["analiz"]:
            continue
        maskeler[ad] = yuvarlak_dikdortgen_maske(yukseklik, genislik, *b["kutu"], b["yaricap"])
    return maskeler


# --------------------------------------------------------------------------
# Yön bulma: kenar haritası + normalize çapraz korelasyon (NCC)
# --------------------------------------------------------------------------
def kenar_haritasi(gri):
    """Sobel gradyan büyüklüğü, sonra geniş Gauss bulanıklaştırma.

    Bulanıklaştırma, köşe tıklamalarındaki birkaç piksellik hatanın
    karşılaştırmayı bozmasını engeller. Renk yerine kenar kullanılır;
    çünkü artıklar renkleri değiştirir ama bölme sınırları aynı kalır.
    """
    gri = cv2.GaussianBlur(gri, (5, 5), 0).astype(np.float32)
    gx = cv2.Sobel(gri, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gri, cv2.CV_32F, 0, 1)
    return cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (15, 15), 0)


def ncc(a, b):
    """Normalize çapraz korelasyon: -1 (ters) ile 1 (aynı) arasında benzerlik.

    NCC = sum((a - ort_a) * (b - ort_b)) / sqrt(sum((a - ort_a)^2) * sum((b - ort_b)^2))
    Ortalama çıkarıldığı ve normalize edildiği için parlaklık farkından etkilenmez.
    """
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# --------------------------------------------------------------------------
# Perspektif düzeltme
# --------------------------------------------------------------------------
def koseleri_sirala(koseler):
    """4 köşeyi saat yönünde sıralar (kullanıcı hangi sırayla tıklarsa tıklasın).

    Görüntüde y aşağı doğru arttığı için, ağırlık merkezine göre açıya
    (atan2) göre artan sıralama saat yönünde bir dizilim verir.
    """
    p = np.asarray(koseler, dtype=np.float32)
    merkez = p.mean(axis=0)
    aci = np.arctan2(p[:, 1] - merkez[1], p[:, 0] - merkez[0])
    return p[np.argsort(aci)]


def koseleri_incelt(gri, kaynak, sablon_kenar, genislik, yukseklik,
                    olcek=0.5, adimlar=(32, 16, 8, 4, 2, 1)):
    """Elle seçilen köşeleri, şablonla hizalamayı en iyi yapacak şekilde ayarlar.

    Neden gerekli: tepsinin köşeleri yuvarlak olduğu için tıklanan nokta
    gerçek köşeden birkaç piksel içeride kalır ve bölmeler kayar.

    Yöntem (kaba-ince yerel arama / hill climbing):
      * 4 köşenin x ve y değerleri = 8 parametre.
      * Her parametre +adım ve -adım kaydırılır; düzeltilmiş görüntünün kenar
        haritasıyla şablonun kenar haritası arasındaki NCC artarsa kabul edilir.
      * İyileşme kalmayınca adım yarıya indirilir (32 -> 16 -> ... -> 1 piksel).
    Hız için karşılaştırma yarım çözünürlükte yapılır.
    Dönüş: (inceltilmis_koseler, son_ncc)
    """
    w, h = int(genislik * olcek), int(yukseklik * olcek)
    hedef = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    sablon_kucuk = cv2.resize(sablon_kenar, (w, h))

    def skor(koseler):
        M = cv2.getPerspectiveTransform(koseler, hedef)
        return ncc(kenar_haritasi(cv2.warpPerspective(gri, M, (w, h))), sablon_kucuk)

    en_iyi_koseler = np.float32(kaynak).copy()
    en_iyi_skor = skor(en_iyi_koseler)
    for adim in adimlar:
        iyilesti = True
        while iyilesti:
            iyilesti = False
            for i in range(4):          # köşe
                for j in range(2):      # x veya y
                    for d in (adim, -adim):
                        aday = en_iyi_koseler.copy()
                        aday[i, j] += d
                        s = skor(aday)
                        if s > en_iyi_skor:
                            en_iyi_koseler, en_iyi_skor, iyilesti = aday, s, True
    return en_iyi_koseler, en_iyi_skor


def tepsiyi_duzelt(goruntu, koseler, sablon=None, bilgi=None, incelt=True):
    """Tepsiyi şablon boyutunda üstten görünüme çevirir.

    1. Köşeler saat yönünde sıralanır.
    2. 4 olası yön denenir, kenar haritası şablona en çok benzeyen seçilir.
    3. (incelt=True ise) köşeler şablonla hizalanacak şekilde ince ayarlanır.

    Dönüş: (duzeltilmis_goruntu, homografi_matrisi, bilgi_sozlugu)
    bilgi_sozlugu: yön skorları, ince ayar öncesi/sonrası NCC ve son köşeler.
    """
    bilgi = bilgi or bolmeleri_yukle()
    genislik, yukseklik = bilgi["boyut"]
    if sablon is None:
        sablon = cv2.imread(str(SABLON_YOLU), cv2.IMREAD_GRAYSCALE)
    sablon_kenar = kenar_haritasi(sablon)
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)

    hedef = np.float32([[0, 0], [genislik, 0], [genislik, yukseklik], [0, yukseklik]])
    sirali = koseleri_sirala(koseler)

    # Yön seçimi: 4 köşeden hangisi şablonun sol-üst köşesi?
    yon_skorlari = []
    for k in range(4):
        M = cv2.getPerspectiveTransform(np.roll(sirali, -k, axis=0), hedef)
        duz_gri = cv2.warpPerspective(gri, M, (genislik, yukseklik))
        yon_skorlari.append(ncc(kenar_haritasi(duz_gri), sablon_kenar))
    en_iyi_yon = int(np.argmax(yon_skorlari))
    kaynak = np.roll(sirali, -en_iyi_yon, axis=0)

    sonuc = {"yon_skorlari": yon_skorlari, "ncc_once": yon_skorlari[en_iyi_yon]}
    if incelt:
        # Köşeler kötü tıklandıysa yön skorları birbirine çok yakın çıkabilir.
        # Bu yüzden her yön ayrı ayrı ince ayarlanır ve sonunda en iyisi seçilir.
        adaylar = [koseleri_incelt(gri, np.roll(sirali, -k, axis=0), sablon_kenar, genislik, yukseklik)
                   for k in range(4)]
        sonuc["inceltilmis_yon_skorlari"] = [s for _, s in adaylar]
        en_iyi_yon = int(np.argmax(sonuc["inceltilmis_yon_skorlari"]))
        kaynak, sonuc["ncc_sonra"] = adaylar[en_iyi_yon]
    sonuc["secilen_yon"] = en_iyi_yon
    sonuc["koseler"] = kaynak.round(1).tolist()

    M = cv2.getPerspectiveTransform(np.float32(kaynak), hedef)
    duz = cv2.warpPerspective(goruntu, M, (genislik, yukseklik))
    return duz, M, sonuc


# --------------------------------------------------------------------------
# Görselleştirme
# --------------------------------------------------------------------------
RENKLER = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 255)]


def bolmeleri_ciz(duz, bilgi=None, saydamlik=0.45):
    """Bölme maskelerini renkli ve etiketli olarak görüntünün üstüne çizer."""
    bilgi = bilgi or bolmeleri_yukle()
    cikti = duz.copy()
    for (ad, maske), renk in zip(bolme_maskeleri(bilgi).items(), RENKLER):
        alan = maske > 0
        cikti[alan] = ((1 - saydamlik) * cikti[alan] + saydamlik * np.array(renk)).astype(np.uint8)
        konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cikti, konturlar, -1, renk, 2)
        x1, y1 = bilgi["bolmeler"][ad]["kutu"][:2]
        cv2.putText(cikti, ad, (x1 + 15, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return cikti
