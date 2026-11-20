import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# تحميل مفتاح OpenCellID من ملف .env
load_dotenv()
API_TOKEN = os.getenv("OPENCELLID_API_KEY")

if not API_TOKEN:
    print("خطأ: مفتاح OpenCellID غير موجود")
    raise SystemExit

# إحداثيات تقريبية لمراكز المناطق الثلاث
AREAS = {
    "ظلم": (22.72, 42.18),
    "المويه": (22.43, 41.76),
    "الخرمة": (21.91, 42.03)
}

url = "https://opencellid.org/cell/getInArea"
all_cells = []

# تقسيم كل منطقة إلى مربعات صغيرة
for name, (lat, lon) in AREAS.items():
    print(f"\nجاري فحص منطقة {name}...")

    area_count = 0

    for i in range(-2, 3):
        for j in range(-2, 3):
            center_lat = lat + i * 0.012
            center_lon = lon + j * 0.012

            bbox = (
                f"{center_lat - 0.006},"
                f"{center_lon - 0.006},"
                f"{center_lat + 0.006},"
                f"{center_lon + 0.006}"
            )

            params = {
                "key": API_TOKEN,
                "BBOX": bbox,
                "mcc": 420,
                "format": "json",
                "limit": 1000
            }

            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    print("خطأ من OpenCellID:", data["error"])
                    continue

                cells = data.get("cells", [])

                for cell in cells:
                    cell["search_area"] = name

                all_cells.extend(cells)
                area_count += len(cells)

            except (requests.RequestException, ValueError) as e:
                print("تعذر فحص أحد المربعات:", e)

    print(f"عدد السجلات المستلمة من {name}: {area_count}")

# حفظ البيانات في ملف مستقل
if all_cells:
    df = pd.DataFrame(all_cells)

    # إزالة التكرارات باستخدام معرفات الخلايا
    identifiers = [
        "mcc", "mnc", "lac", "cellid", "radio"
    ]

    if all(col in df.columns for col in identifiers):
        df = df.drop_duplicates(subset=identifiers)

    Path("data").mkdir(exist_ok=True)

    output = "data/opencellid_three_areas_test.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print("\nتم حفظ البيانات بنجاح")
    print("الملف:", output)
    print("إجمالي السجلات الفريدة:", len(df))
else:
    print("\nلم يتم استلام سجلات")
    print("هذا لا يعني عدم وجود تغطية")