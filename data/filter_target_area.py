import pandas as pd

print("جاري قراءة ملف بيانات Ookla الشامل...")

# قراءة الملف من داخل مجلد data مباشرة
df = pd.read_csv("data/ESHARAH_Ookla_Saudi_2024_Q4.csv")

lat_min, lat_max = 21.0, 23.5
lon_min, lon_max = 40.5, 43.5

filtered_df = df[
    (df["latitude"] >= lat_min)
    & (df["latitude"] <= lat_max)
    & (df["longitude"] >= lon_min)
    & (df["longitude"] <= lon_max)
].copy()

output_path = "data/target_area_metrics.csv"
filtered_df.to_csv(output_path, index=False)

print(f"✅ تم بنجاح استخراج {len(filtered_df)} نقطة وحفظها في {output_path}")