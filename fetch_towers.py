import requests

# الرمز الخاص بك هنا داخل علامتي التنصيص
API_TOKEN = "pk.dedcfecd99990751b792f103d03ad94f"

# حدود الإحداثيات الجغرافية لمنطقة المشروع (المويه، الخرمة، ظلم)
lat_min, lon_min = 21.8, 41.7
lat_max, lon_max = 23.2, 42.6

# رابط طلب بيانات الأبراج للممنطقة بصيغة CSV
url = f"https://opencellid.org/cell/getCellsBox?key={API_TOKEN}&BBOX={lat_min},{lon_min},{lat_max},{lon_max}&format=csv"

print("جاري سحب بيانات الأبراج الحقيقية من خوادم OpenCellID...")

try:
    response = requests.get(url)
    if response.status_code == 200:
        # حفظ الملف مباشرة داخل مجلد data باسم real_towers.csv
        with open("data/real_towers.csv", "wb") as f:
            f.write(response.content)
        print("✅ تم بنجاح جلب وحفظ ملف الأبراج الحقيقي: data/real_towers.csv")
    else:
        print(f"⚠️ حدث خطأ في الاستجابة (رمز الخطأ: {response.status_code})")
        print(response.text)
except Exception as e:
    print(f"⚠️ حدث خطأ أثناء الاتصال: {e}")