"""Boş tepsi fotoğrafından düzeltilmiş tepsi şablonunu (sablon.png) üretir.

Kullanım:
    python tools/sablon_olustur.py

Girdi : data/tepsi/bos_tepsi.jpg ve köşeleri (data/tepsi/bos_tepsi.koseler.json)
Çıktı : data/tepsi/sablon.png (1200x800, tam üstten görünüm)

Şablonda yön sabittir: dar uzun bölme (çatal-kaşık) solda, çorba bölmesi
üstte ortada. Bu yüzden köşeler burada saat yönünde, sol-üstten başlayarak
verilir ve yön araması yapılmaz.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tray_template import KOK, SABLON_YOLU, bolmeleri_yukle, bolmeleri_ciz  # noqa: E402


def main():
    goruntu = cv2.imread(str(KOK / "data" / "tepsi" / "bos_tepsi.jpg"))
    koseler = json.loads((KOK / "data" / "tepsi" / "bos_tepsi.koseler.json").read_text())["koseler"]
    genislik, yukseklik = bolmeleri_yukle()["boyut"]

    kaynak = np.float32(koseler)
    hedef = np.float32([[0, 0], [genislik, 0], [genislik, yukseklik], [0, yukseklik]])
    M = cv2.getPerspectiveTransform(kaynak, hedef)
    sablon = cv2.warpPerspective(goruntu, M, (genislik, yukseklik))

    cv2.imwrite(str(SABLON_YOLU), sablon)
    cv2.imwrite(str(KOK / "ciktilar" / "sablon_bolmeler.jpg"), bolmeleri_ciz(sablon))
    print("Şablon kaydedildi:", SABLON_YOLU)
    print("Kontrol görseli: ciktilar/sablon_bolmeler.jpg")


if __name__ == "__main__":
    main()
