import getpass
import requests
import pandas as pd

# موقع تجربتنا في المويه
lat = 22.4298417
lon = 41.7603567

# منطقة صغيرة حول موقع التجربة
bbox = (
    f"{lat - 0.007},{lon - 0.007},"
    f"{lat + 0.007},{lon + 0.007}"
)

# إدخال المفتاح بدون إظهاره على الشاشة
api_key = getpass.getpass("Enter your API key: ")

url = "https://opencellid.org/cell/getInArea"

params = {
    "key": api_key,
    "BBOX": bbox,
    "mcc": 420,
    "format": "json",
    "limit": 50
}

try:
    response = requests.get(
        url, params=params, timeout=30
    )
    data = response.json()
except (requests.RequestException, ValueError):
    print("تعذر الاتصال أو قراءة استجابة الموقع")
    raise SystemExit

if response.status_code != 200 or "error" in data:
    print("فشل الطلب. رمز الحالة:", response.status_code)
    print("رمز الخطأ:", data.get("code", "غير معروف"))
    raise SystemExit

cells = data.get("cells", [])

print("عدد السجلات المستلمة:", len(cells))

if cells:
    df = pd.DataFrame(cells)
    df.to_csv(
        "data/opencellid_muwaih.csv",
        index=False
    )
    print("تم حفظ البيانات بنجاح")
    print(df.head())
else:
    print("لا توجد سجلات في المنطقة المحددة")
    print("هذا لا يعني عدم وجود تغطية")