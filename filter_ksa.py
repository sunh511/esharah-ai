from pathlib import Path
import pandas as pd

output_file = "ksa_network_data.csv"

LAT_MIN, LAT_MAX = 16.0, 32.5
LON_MIN, LON_MAX = 34.5, 56.0
chunk_size = 500_000

project_dir = Path(__file__).resolve().parent
input_path = project_dir / "data" / "420.csv"

if not input_path.exists():
  print(f"الملف غير موجود في المسار: {input_path}")
  raise SystemExit(1)

print(f"جاري قراءة وفلترة الملف: {input_path}")

opencellid_cols = [
    "radio",
    "mcc",
    "net",
    "area",
    "cell",
    "unit",
    "lon",
    "lat",
    "range",
    "samples",
    "changeable",
    "created",
    "updated",
    "averageSignal",
]

first_chunk = True
total_saved = 0

for chunk in pd.read_csv(
    input_path, chunksize=chunk_size, header=None, low_memory=False
):
  # تسمية الأعمدة حسب عدد الأعمدة في الملف
  if len(chunk.columns) == len(opencellid_cols):
    chunk.columns = opencellid_cols
  else:
    chunk.columns = [f"col_{i}" for i in range(len(chunk.columns))]

  # البحث عن عمودي الطول والعرض (إما بالاسم أو بالقيم الواقعة ضمن حدود السعودية)
  if "lat" in chunk.columns and "lon" in chunk.columns:
    lat_col, lon_col = "lat", "lon"
  else:
    lat_col = lon_col = None
    for col in chunk.columns:
      s = pd.to_numeric(chunk[col], errors="coerce")
      if lat_col is None and s.between(LAT_MIN, LAT_MAX).any():
        lat_col = col
      elif lon_col is None and s.between(LON_MIN, LON_MAX).any():
        lon_col = col

  if not lat_col or not lon_col:
    print("تعذر تحديد أعمدة الإحداثيات تلقائياً.")
    print(chunk.head(2))
    raise SystemExit(1)

  lat_vals = pd.to_numeric(chunk[lat_col], errors="coerce")
  lon_vals = pd.to_numeric(chunk[lon_col], errors="coerce")

  ksa_chunk = chunk[
      lat_vals.between(LAT_MIN, LAT_MAX) & lon_vals.between(LON_MIN, LON_MAX)
  ]

  if not ksa_chunk.empty:
    ksa_chunk.to_csv(
        output_file,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
    )
    first_chunk = False
    total_saved += len(ksa_chunk)
    print(f"تم حفظ {total_saved} صف للسعودية حتى الآن...")

print(
    f"انتهت العملية بنجاح. الملف الناتج: {output_file} | إجمالي الصفوف:"
    f" {total_saved}"
)