"""Bir tepsi fotoğrafında 4 köşeyi fareyle işaretleyip JSON olarak kaydeder.

Kullanım:
    python tools/kose_sec.py data/ornekler/artik_001.jpg

- Tepsinin 4 köşesine sol tıklayın (sıra önemli değil).
- r: sıfırla, s: kaydet ve çık, Esc: kaydetmeden çık.
Sonuç, görüntünün yanına <ad>.koseler.json olarak yazılır.

Bu köşeler hem perspektif düzeltme için hem de ileride otomatik tepsi
tespitinin doğruluğunu ölçmek için "gerçek değer" olarak kullanılır.
"""
import json
import sys
from pathlib import Path

import cv2

EKRAN_MAKS = 900  # pencerenin en uzun kenarı (piksel)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    yol = Path(sys.argv[1])
    goruntu = cv2.imread(str(yol))
    if goruntu is None:
        sys.exit(f"Görüntü okunamadı: {yol}")

    olcek = min(1.0, EKRAN_MAKS / max(goruntu.shape[:2]))
    kucuk = cv2.resize(goruntu, None, fx=olcek, fy=olcek)
    noktalar = []

    def ciz():
        g = kucuk.copy()
        for i, (x, y) in enumerate(noktalar):
            cv2.circle(g, (int(x * olcek), int(y * olcek)), 6, (0, 0, 255), -1)
            cv2.putText(g, str(i + 1), (int(x * olcek) + 8, int(y * olcek) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("kose_sec", g)

    def tikla(olay, x, y, *_):
        if olay == cv2.EVENT_LBUTTONDOWN and len(noktalar) < 4:
            noktalar.append([round(x / olcek), round(y / olcek)])  # orijinal koordinat
            ciz()

    cv2.namedWindow("kose_sec")
    cv2.setMouseCallback("kose_sec", tikla)
    ciz()
    while True:
        tus = cv2.waitKey(20) & 0xFF
        if tus == ord("r"):
            noktalar.clear()
            ciz()
        elif tus == ord("s") and len(noktalar) == 4:
            hedef = yol.with_suffix(".koseler.json")
            hedef.write_text(json.dumps({"koseler": noktalar}), encoding="utf-8")
            print(f"Kaydedildi: {hedef} -> {noktalar}")
            break
        elif tus == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
