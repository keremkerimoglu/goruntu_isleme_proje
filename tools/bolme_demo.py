"""Bir tepsi fotoğrafını düzeltir, yönünü bulur ve bölmeleri üstüne çizer.

Kullanım:
    python tools/bolme_demo.py data/ornekler/artik_001.jpg

Köşeler <ad>.koseler.json dosyasından okunur (yoksa önce tools/kose_sec.py
ile işaretleyin). Çıktı ciktilar/ klasörüne yazılır.
"""
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tray_template import KOK, bolmeleri_ciz, tepsiyi_duzelt  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    yol = Path(sys.argv[1])
    goruntu = cv2.imread(str(yol))
    kose_dosyasi = yol.with_suffix(".koseler.json")
    if not kose_dosyasi.exists():
        sys.exit(f"Köşe dosyası yok: {kose_dosyasi}\nÖnce: python tools/kose_sec.py {yol}")
    koseler = json.loads(kose_dosyasi.read_text())["koseler"]

    duz, _, sonuc = tepsiyi_duzelt(goruntu, koseler)
    print("Yön skorları (NCC):", [round(s, 3) for s in sonuc["yon_skorlari"]])
    print("İnce ayar sonrası yön skorları:", [round(s, 3) for s in sonuc["inceltilmis_yon_skorlari"]])
    print(f"Seçilen yön: {sonuc['secilen_yon']}, NCC {sonuc['ncc_once']:.3f} -> {sonuc['ncc_sonra']:.3f}")
    print("Kullanılan köşeler:", sonuc["koseler"])

    cikti_klasoru = KOK / "ciktilar"
    cikti_klasoru.mkdir(exist_ok=True)
    cv2.imwrite(str(cikti_klasoru / f"{yol.stem}_duz.jpg"), duz)
    cv2.imwrite(str(cikti_klasoru / f"{yol.stem}_bolmeler.jpg"), bolmeleri_ciz(duz))
    print(f"Çıktılar: ciktilar/{yol.stem}_duz.jpg, ciktilar/{yol.stem}_bolmeler.jpg")


if __name__ == "__main__":
    main()
