"""المساعد الذكي الدلالي لمشروع إشارة | ESHARAH AI.

ضع هذا الملف بجانب app.py باسم semantic_search.py.

لا يلزم تغيير الاستدعاء الموجود في app.py:
    render_qa_assistant(route_summaries=route_summaries)
"""

import math
import re

from html import escape as escape_html
from textwrap import dedent

import numpy as np
import streamlit as st


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


ROUTE_NAMES = [
    "المويه الجديدة إلى الخرمة",
    "الخرمة إلى ظلم",
    "ظلم إلى المويه الجديدة",
]


# =========================================
# الأسئلة والإجابات الحديثة
# =========================================

FAQ = [

    {
        "title": "كيف يعمل مشروع إشارة؟",

        "questions": [
            "كيف يعمل النظام وما هي المصادر؟",
            "اشرح طريقة عمل إشارة",
            "كيف يعمل مشروع إشارة؟",
        ],

        "answer": (
            "يعرض **إشارة | ESHARAH AI** طرق القيادة الفعلية "
            "بين المدن باستخدام **OSRM**، "
            "ويقسّم كل طريق إلى مقاطع طولها نحو **5 كم**.\n\n"

            "إذا توفر نموذج ضعف السرعة المدرّب، "
            "يجهّز التطبيق خصائص جغرافية لمركز كل مقطع "
            "باستخدام `prepare_features` "
            "في ملف `train_esharah_ai.py`.\n\n"

            "ثم يستخدم النموذج الموجود في "
            "`models/weak_speed_risk_model.joblib` "
            "لتقدير خطر أن تكون سرعة التنزيل "
            "**15 Mbps أو أقل**.\n\n"

            "بعد ذلك يلوّن مقاطع الطريق حسب مؤشر النموذج، "
            "ويعرض تقييمًا إجماليًا لكل مسار، "
            "وعدد المقاطع التي تجاوزت حد التنبيه.\n\n"

            "تُعرض أيضًا قياسات **Ookla** "
            "وسجلات خلايا الاتصال ومؤشرات الطريق المجمّعة.\n\n"

            "**مهم:** القياسات المرصودة تختلف عن توقعات النموذج. "
            "التنبؤ بضعف السرعة لا يعني تأكيد انقطاع الشبكة."
        ),

        "source": (
            "app.py — get_route وsplit_road_into_segments "
            "وprepare_features والتنبؤ"
        ),
    },

    {
        "title": "كيف يصنّف الذكاء الاصطناعي المسارات؟",

        "questions": [
            "وش المستويات الأربعة للتقييم؟",
            "كيف يتم تصنيف المسارات؟",
            "كيف يعطي النموذج تقييم الطريق؟",
            "كيف يحسب عدد المقاطع الضعيفة؟",
        ],

        "answer": (
            "**النموذج الأساسي الحالي:**\n\n"

            "يتنبأ بمؤشر خطر ضعف سرعة التنزيل "
            "لكل مقطع بطول يقارب **5 كم**.\n\n"

            "والمقصود بالضعف هنا سرعة تنزيل "
            "**15 Mbps أو أقل**.\n\n"

            "**ألوان خط الطريق:**\n\n"

            "🟢 أخضر: مؤشر أقل من **15%**.\n\n"

            "🟠 برتقالي: من **15%** إلى أقل من **30%**.\n\n"

            "🔴 أحمر: **30% فأكثر**.\n\n"

            "ويُحسب المقطع ضمن عداد التنبيه الأحمر "
            "إذا بلغ مؤشره **30%** أو أكثر.\n\n"

            "**التقييم الإجمالي للطريق:**\n\n"

            "يحسب التطبيق متوسط مؤشرات جميع المقاطع، "
            "ثم يعرض إحدى الفئات الأربع:\n\n"

            "- **ممتاز:** أقل من 15%.\n"
            "- **جيد جدًا:** من 15% إلى أقل من 20%.\n"
            "- **جيد:** من 20% إلى أقل من 30%.\n"
            "- **ضعيف:** 30% فأكثر.\n\n"

            "قد يحتوي طريق تقييمه الإجمالي جيدًا "
            "على مقاطع حمراء؛ "
            "لذلك يعرض التطبيق عددها بشكل منفصل.\n\n"

            "**ملاحظة:** "
            "المؤشرات تقديرية يحسبها النموذج لكل مقطع."
        ),

        "source": (
            "app.py — risk_color "
            "وroute_label_from_mean_risk وsegment_scores"
        ),
    },

    {
        "title": "ماذا تعني ألوان الخريطة؟",

        "questions": [
            "ماذا تعني ألوان الخريطة؟",
            "ما معنى اللون الأحمر والأخضر والبرتقالي؟",
            "وش تمثل رموز الاتصال على الخريطة؟",
            "وش الفرق بين الدوائر والخطوط؟",
        ],

        "answer": (
            "هناك فرق بين **نقاط القياس** "
            "و**خط الطريق التنبؤي**.\n\n"

            "**أولًا: دوائر قياسات سرعة الإنترنت**\n\n"

            "🟢 الأخضر: سرعة تنزيل **50 Mbps فأكثر**.\n\n"

            "🟠 البرتقالي: من **15 Mbps** "
            "إلى أقل من **50 Mbps**.\n\n"

            "🔴 الأحمر: سرعة أقل من **15 Mbps**.\n\n"

            "هذه الدوائر تعرض قياسات موجودة "
            "في ملفات البيانات.\n\n"

            "**ثانيًا: ألوان خطوط الطرق**\n\n"

            "🟢 الأخضر: مؤشر خطر ضعف السرعة أقل من 15%.\n\n"

            "🟠 البرتقالي: من 15% إلى أقل من 30%.\n\n"

            "🔴 الأحمر: مؤشر 30% فأكثر.\n\n"

            "هذه توقعات لكل مقطع من الطريق "
            "عندما يعمل النموذج المدرّب.\n\n"

            "**📡 رموز الاتصال:**\n\n"

            "توضح سجلات خلايا الشبكة المتاحة، "
            "وليست بالضرورة مواقع أبراج مستقلة مؤكدة.\n\n"

            "**اللون الأحمر لا يثبت حدوث انقطاع فعلي للاتصال.**"
        ),

        "source": (
            "app.py — قواعد ألوان Folium"
        ),
    },

    {
        "title": "كيف تُحسب المسافات ومؤشرات الطرق؟",

        "questions": [
            "كيف تُحسب المسافات والسرعات والخلية القريبة؟",
            "كيف تحسبون المتوسط المرجح IDW؟",
            "كيف يربط النظام البيانات بالطريق؟",
        ],

        "answer": (
            "يحصل التطبيق على **مسار القيادة الحقيقي ومسافته** "
            "من خدمة **OSRM**.\n\n"

            "ثم يقسم شكل الطريق إلى مقاطع "
            "طول كل منها نحو **5 كم**.\n\n"

            "**حساب متوسط السرعة:**\n\n"

            "يستخدم التطبيق المتوسط المرجح بعكس المسافة "
            "**IDW** لقياسات الإنترنت القريبة "
            "من الطريق ضمن مسافة **20 كم**.\n\n"

            "**حساب مؤشرات التغطية:**\n\n"

            "يبحث عن مؤشرات التغطية الجغرافية "
            "القريبة ضمن نطاق **15 كم**.\n\n"

            "**حساب سجلات الخلايا:**\n\n"

            "يعدّ سجلات الخلايا القريبة "
            "من نقاط الطريق ضمن نطاق **15 كم**.\n\n"

            "إذا لم تتوفر قياسات مناسبة، "
            "يمكن استخدام قيم احتياطية "
            "أو مراجع إقليمية حسب الكود.\n\n"

            "**مهم:** مسافات الربط هذه مخصصة للتحليل، "
            "ولا تعني أن كل برج يغطي "
            "نطاقًا ثابتًا بهذه المسافة.\n\n"

            "متوسط السرعة المعروض في بطاقة الطريق "
            "مؤشر وصفي، وهو منفصل عن توقع "
            "ضعف السرعة لكل مقطع من النموذج المدرّب."
        ),

        "source": (
            "app.py — get_route "
            "وweighted_route_metrics وحساب خلايا الطريق"
        ),
    },

    {
        "title": "كيف يتنبأ إشارة بمناطق ضعف الشبكة؟",

        "questions": [
            "كيف يتم رصد نقاط الضعف والتغطية؟",
            "كيف يتوقع الذكاء الاصطناعي ضعف الشبكة؟",
            "كيف يحدد النظام المناطق الضعيفة؟",
            "كيف يعمل التنبؤ بالتغطية؟",
            "كيف يتوقع المشروع انقطاع الاتصال؟",
        ],

        "answer": (
            "**في النسخة الحالية من مشروع إشارة** "
            "يوجد نموذج مدرّب للتنبؤ بخطر "
            "**ضعف سرعة تنزيل الإنترنت**.\n\n"

            "استُخدمت بيانات قياسات **Ookla** "
            "للربع الرابع من عام **2024** "
            "في تدريب نموذج ضعف السرعة.\n\n"

            "ويستخدم تجهيز خصائص النموذج "
            "إشارات جغرافية من الملفات المتاحة، "
            "مثل تقديرات تغطية **4G** "
            "وسجلات خلايا الاتصال.\n\n"

            "أما **M-Lab** فيستخدمه `app.py` "
            "كمرجع إقليمي لبعض حسابات العرض "
            "والقيم الاحتياطية، "
            "وليس كقياس مباشر لكل مقطع.\n\n"

            "**مراحل التنبؤ:**\n\n"

            "1. يحصل النظام على مسار القيادة "
            "باستخدام **OSRM**.\n\n"

            "2. يقسم الطريق إلى مقاطع "
            "بطول يقارب **5 كم**.\n\n"

            "3. يجهز خصائص مركز كل مقطع "
            "باستخدام `prepare_features`.\n\n"

            "4. يشغّل النموذج المدرّب لإنتاج "
            "مؤشر تقديري لخطر أن تكون سرعة التنزيل "
            "**15 Mbps أو أقل**.\n\n"

            "5. يلوّن المقاطع بالأخضر أو البرتقالي "
            "أو الأحمر حسب المؤشر.\n\n"

            "6. يحسب متوسط المؤشر لكل طريق "
            "وعدد المقاطع التي وصلت إلى حد التنبيه الأحمر.\n\n"

            "**ماذا يعني ذلك؟**\n\n"

            "يمكن للمشروع تقدير مناطق معرضة "
            "لضعف سرعة الإنترنت "
            "حتى عند عدم توفر قياس مباشر "
            "في كل نقطة من الطريق.\n\n"

            "**ماذا لا يعني؟**\n\n"

            "لا يرصد النظام انقطاعًا لحظيًا، "
            "ولا يؤكد انعدام التغطية الخلوية، "
            "ولا يعتبر غياب البيانات دليلًا على الانقطاع.\n\n"

            "تعتمد التوقعات على بيانات **Ookla** "
            "وتقديرات تغطية **4G** وسجلات الخلايا، "
            "وتُعرض كمؤشرات تساعد على اختيار الطريق."
        ),

        "source": (
            "app.py وtrain_esharah_ai.py"
        ),
    },

    {
        "title": "ما الأدوات البرمجية المستخدمة؟",

        "questions": [
            "ما هي التقنيات والأدوات المستخدمة في المشروع؟",
            "ما لغة البرمجة والمكتبات؟",
            "ما هو دور Sentence Transformers؟",
        ],

        "answer": (
            "المشروع مبني بلغة **Python**.\n\n"

            "**Streamlit:** لإنشاء واجهة الموقع.\n\n"

            "**Folium:** لعرض الخريطة والمسارات "
            "ونقاط قياس الإنترنت.\n\n"

            "**OSRM:** للحصول على مسارات القيادة الحقيقية.\n\n"

            "**Pandas وNumPy:** لتحليل البيانات "
            "وتجهيز المؤشرات.\n\n"

            "**Scikit-learn:** لتشغيل نماذج التصنيف.\n\n"

            "**Joblib:** لتحميل النموذج المدرّب المحفوظ.\n\n"

            "**Sentence Transformers:** "
            "لفهم التشابه الدلالي بين سؤال المستخدم "
            "والأسئلة المعروفة داخل المساعد.\n\n"

            "المساعد الدلالي يفسر المعلومات "
            "ونتائج الطرق المرسلة إليه من `app.py`، "
            "ولا يعيد تدريب نموذج توقع الشبكة بنفسه."
        ),

        "source": (
            "app.py وsemantic_search.py"
        ),
    },

    {
        "title": "ما المدن والطرق المشمولة حاليًا؟",

        "questions": [
            "ما هي المدن والطرق المشمولة في الدراسة؟",
            "ما المدن التي يحللها النظام الآن؟",
            "هل يغطي المشروع جميع طرق السعودية؟",
        ],

        "answer": (
            "تشمل واجهة الدراسة الحالية "
            "**المويه الجديدة والخرمة وظلم**.\n\n"

            "ويحلل التطبيق ثلاثة مسارات قيادة:\n\n"

            "- **المويه الجديدة إلى الخرمة**.\n"
            "- **الخرمة إلى ظلم**.\n"
            "- **ظلم إلى المويه الجديدة**.\n\n"

            "عند اختيار **جميع الطرق** "
            "تظهر بطاقات نتائج المسارات الثلاثة، "
            "ويمكن للمساعد مقارنتها.\n\n"

            "تظهر الخريطة في أعلى منطقة التحليل، "
            "وتقع إعدادات اختيار الطريق أسفلها.\n\n"

            "التوسع إلى طرق المملكة العربية السعودية "
            "كاملة هدف مستقبلي للمشروع، "
            "وليس تغطية متاحة في النسخة الحالية."
        ),

        "source": (
            "app.py — PLACES وROADS"
        ),
    },

    {
        "title": "ما مصادر بيانات مشروع إشارة؟",

        "questions": [
            "من وين تجيب المعلومات؟",
            "وش مصادر البيانات؟",
            "هل القياسات حقيقية أم توقعات؟",
            "وش ملفات E-1 إلى E-5؟",
        ],

        "answer": (
            "**أولًا: قياسات سرعات الإنترنت — Ookla**\n\n"

            "يستخدم المشروع ملفات قياسات مثل "
            "`target_area_metrics.csv` "
            "و`ESHARAH_Ookla_Saudi_2024_Q4.csv`.\n\n"

            "تتضمن البيانات مواقع القياس "
            "وسرعة الإنترنت وزمن الاستجابة "
            "بحسب الأعمدة الموجودة.\n\n"

            "**ثانيًا: بيانات التغطية الجغرافية — Zenodo**\n\n"

            "تستخدم الدراسة مجموعة بيانات بحثية "
            "لتقديرات تغطية شبكات **4G** "
            "على شبكة نقاط جغرافية.\n\n"

            "`E-2.csv`: مقارنة تقديرات التغطية "
            "بين **2024 و2026**، ويقابله "
            "`ESHARAH_4G_FOCUS_COMPARISON_2024_2026.csv`.\n\n"

            "`E-3.csv`: تقديرات تغطية **4G لعام 2024**، "
            "ويقابله `ESHARAH_4G_FOCUS_2024.csv`.\n\n"

            "`E-4.csv`: تقديرات تغطية **4G لعام 2025**، "
            "ويقابله `ESHARAH_4G_FOCUS_2025.csv`.\n\n"

            "`E-5.csv`: تقديرات تغطية **4G لعام 2026**، "
            "ويقابله `ESHARAH_4G_FOCUS_2026 2.csv`.\n\n"

            "هذه البيانات **تقديرات نموذجية** "
            "وليست قياسات ميدانية مباشرة للتغطية.\n\n"

            "**ثالثًا: Measurement Lab — M-Lab**\n\n"

            "`E-1.csv`: بيانات ومؤشرات أداء الإنترنت "
            "الإقليمية. ويقرأ التطبيق أيضًا الاسم البديل "
            "`ESHARAH_MLAB_Saudi_2026_09.csv` "
            "إذا كان الملف متاحًا.\n\n"

            "**رابعًا: سجلات خلايا الاتصال**\n\n"

            "يستخدم المشروع `ksa_network_data.csv` "
            "وملفات أخرى مثل `real_towers.csv` "
            "و`saudi_raw.csv` و`towers.csv`.\n\n"

            "هذه سجلات خلايا وليست بالضرورة "
            "أبراج اتصالات مستقلة مؤكدة.\n\n"

            "**خامسًا: OSRM**\n\n"

            "تستخدم الخدمة لجلب شكل الطريق الحقيقي "
            "ومسافة القيادة بين المدن.\n\n"

            "**مهم:** لا يعني وجود اسم الملف في الكود "
            "أن التطبيق تمكن من تحميله في كل تشغيل. "
            "كما يجب التمييز بين القياسات الفعلية "
            "والخرائط التقديرية ومخرجات الذكاء الاصطناعي."
        ),

        "source": (
            "app.py — تحميل البيانات وقسم المصادر"
        ),
    },

]


# =========================================
# تنسيق الكلمات العربية
# =========================================

def _normalize(value):

    text = str(value or "").lower()

    text = re.sub(
        r"[\u064b-\u065f\u0670\u0640]",
        "",
        text
    )

    return text.translate(
        str.maketrans({
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ى": "ي",
            "ة": "ه"
        })
    )


# =========================================
# التحقق من القيم الرقمية
# =========================================

def _safe_number(value):

    try:

        result = float(value)

        return (
            result
            if math.isfinite(result)
            else None
        )

    except (TypeError, ValueError):

        return None


# =========================================
# تجهيز الإجابة
# =========================================

def _reply(
    title,
    answer,
    source=None,
    status="success",
    similarity=None
):

    return {
        "title": title,
        "answer": answer,
        "source": source,
        "status": status,
        "similarity": similarity,
    }


# =========================================
# التعرف على اسم الطريق في السؤال
# =========================================

def _find_route_in_question(question):

    q = _normalize(question)

    if "المويه" in q and "الخرمه" in q:

        return ROUTE_NAMES[0]

    if "الخرمه" in q and "ظلم" in q:

        return ROUTE_NAMES[1]

    if "ظلم" in q and "المويه" in q:

        return ROUTE_NAMES[2]

    return None


# =========================================
# الإجابة عن تفاصيل الطريق الحالية
# =========================================

def _route_answer(route):

    """
    شرح نتائج الطريق المحسوبة والممررة من app.py.
    """

    parts = [
        f"**الطريق:** {route.get('route_name', 'غير معروف')}"
    ]

    distance = _safe_number(
        route.get("distance")
    )

    speed = _safe_number(
        route.get("speed")
    )

    latency = _safe_number(
        route.get("latency")
    )

    coverage = _safe_number(
        route.get("coverage")
    )

    cells = _safe_number(
        route.get("cells")
    )

    mean_risk = _safe_number(
        route.get("mean_risk")
    )

    weak_segments = _safe_number(
        route.get("weak_segments")
    )

    total_segments = _safe_number(
        route.get("total_segments")
    )

    if distance is not None:

        parts.append(
            f"**المسافة حسب OSRM:** {distance:.1f} كم"
        )

    if speed is not None:

        parts.append(
            "**متوسط سرعة التنزيل المحسوب للطريق:** "
            f"{speed:.1f} Mbps"
        )

    if latency is not None:

        parts.append(
            f"**زمن الاستجابة المحسوب:** {latency:.1f} ms"
        )

    if coverage is not None:

        parts.append(
            "**مؤشر التغطية الجغرافي التقديري:** "
            f"{coverage:.2f}"
        )

    if cells is not None:

        parts.append(
            f"**سجلات الخلايا القريبة:** {int(cells)}"
        )

    if route.get("eval"):

        parts.append(
            f"**التقييم المعروض للطريق:** {route['eval']}"
        )

    if mean_risk is not None:

        parts.append(
            "**متوسط مؤشر خطر ضعف السرعة عبر المقاطع:** "
            f"{mean_risk:.0%}"
        )

        if (
            weak_segments is not None
            and total_segments is not None
        ):

            parts.append(
                "**المقاطع التي بلغت حد التنبيه الأحمر "
                "(30% فأكثر):** "
                f"{int(weak_segments)} من {int(total_segments)}"
            )

        parts.append(
            "**التفسير:** هذه نتائج النموذج المدرّب "
            "لمقاطع تقارب 5 كم. "
            "الهدف هو توقع سرعة تنزيل 15 Mbps أو أقل. "
            "المؤشر ليس قياسًا ميدانيًا "
            "ولا إثباتًا لانقطاع الشبكة."
        )

    else:

        parts.append(
            "**وضع التشغيل:** "
            "تقييم تجريبي احتياطي للطريق ككل. "
            "توقع المقاطع من النموذج المدرّب "
            "غير متاح في العرض الحالي."
        )

    parts.append(
        "**ملاحظة:** متوسط السرعة وزمن الاستجابة "
        "مؤشرات تحليلية من القياسات القريبة "
        "أو المراجع الاحتياطية. "
        "وهي لا تمثل بالضرورة سرعة كل متر من الطريق. "
        "كما أن سجلات الخلايا ليست بالضرورة "
        "عدد أبراج فعلية مستقلة."
    )

    return "\n\n".join(parts)


# =========================================
# مقارنة الطرق
# =========================================

def _compare_routes(routes):

    if not routes:

        return _reply(
            "مقارنة الطرق",
            "لم تظهر نتائج مسارات محسوبة بعد. "
            "اختر **جميع الطرق** "
            "من الإعدادات الموجودة تحت الخريطة.",
            status="warning",
        )

    if len(routes) < 3:

        return _reply(
            "مقارنة الطرق",
            "اختر **جميع الطرق** "
            "من الإعدادات الموجودة تحت الخريطة "
            "حتى تظهر نتائج الطرق الثلاثة للمقارنة.",
            status="warning",
        )

    # نفس ترتيب المقارنة:
    # تقييم الطريق أولًا، ثم متوسط السرعة لكسر التعادل.

    evaluation_rank = {
        "ضعيف": 0,
        "جيد": 1,
        "جيد جداً": 2,
        "ممتاز": 3
    }

    ordered = sorted(
        routes,
        key=lambda r: (
            evaluation_rank.get(
                str(r.get("eval", "")),
                -1
            ),
            _safe_number(
                r.get("speed")
            ) or -1,
        ),
        reverse=True,
    )

    lines = []

    for route in ordered:

        speed = _safe_number(
            route.get("speed")
        )

        risk = _safe_number(
            route.get("mean_risk")
        )

        weak = _safe_number(
            route.get("weak_segments")
        )

        total = _safe_number(
            route.get("total_segments")
        )

        speed_text = (
            f"{speed:.1f} Mbps"
            if speed is not None
            else "غير متاحة"
        )

        line = (
            f"- **{route.get('route_name', 'مسار')}**: "
            f"التقييم **{route.get('eval', 'غير معروف')}**، "
            f"متوسط السرعة {speed_text}"
        )

        if risk is not None:

            line += (
                f"، متوسط مؤشر ضعف السرعة {risk:.0%}"
            )

            if weak is not None and total is not None:

                line += (
                    f"، المقاطع عند حد التنبيه "
                    f"{int(weak)} من {int(total)}"
                )

        lines.append(
            line
        )

    best = ordered[0]

    answer = (
        "**الأعلى وفق ترتيب التقييم المعروض:** "
        f"{best.get('route_name', 'مسار')} "
        f"({best.get('eval', 'غير معروف')}).\n\n"

        + "\n".join(lines)

        + "\n\n**طريقة المقارنة:** "
        "الترتيب يعتمد على فئة تقييم الطريق، "
        "ثم متوسط السرعة المحسوب لكسر التعادل. "

        "مؤشر المقاطع يوضح خطر ضعف سرعة التنزيل "
        "(15 Mbps أو أقل)، "
        "ولا يثبت انقطاع الشبكة. "

        "هذه المقارنة مبنية على تقييم النموذج لكل طريق."
    )

    return _reply(
        "مقارنة المسارات",
        answer,
    )


# =========================================
# تحميل نموذج البحث الدلالي
# =========================================

@st.cache_resource(show_spinner=False)
def _get_model():

    try:

        from sentence_transformers import (
            SentenceTransformer
        )

        return SentenceTransformer(
            MODEL_NAME
        )

    except Exception:

        return None


# =========================================
# تجهيز تمثيلات الأسئلة
# =========================================

@st.cache_data(show_spinner=False)
def _faq_vectors(questions):

    model = _get_model()

    if model is None:

        return None

    return model.encode(
        list(questions),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# =========================================
# تحليل السؤال واختيار الإجابة
# =========================================

def get_answer(
    question,
    route_summaries=None,
    threshold=0.42
):

    q = (question or "").strip()

    if not q:

        return _reply(
            "اكتب سؤالاً",
            "اختر سؤالاً جاهزاً أو اكتب سؤالك.",
            status="warning"
        )

    nq = _normalize(
        q
    )

    routes = (
        route_summaries or []
    )

    # =====================================
    # مقارنة جميع الطرق
    # =====================================

    if any(
        k in nq
        for k in (
            "قارن",
            "مقارنه",
            "الافضل",
            "افضل طريق",
            "ايهم افضل"
        )
    ):

        return _compare_routes(
            routes
        )

    # =====================================
    # السؤال عن طريق معين
    # =====================================

    route_name = _find_route_in_question(
        q
    )

    if route_name:

        current = next(
            (
                r
                for r in routes
                if _normalize(
                    r.get("route_name")
                ) == _normalize(
                    route_name
                )
            ),
            None,
        )

        if current is None:

            return _reply(
                "تفاصيل المسار",
                f"لم يُحسب طريق **{route_name}** "
                "في العرض الحالي. "
                "اختر هذا المسار أو **جميع الطرق** "
                "من الإعدادات تحت الخريطة.",
                status="warning",
            )

        return _reply(
            "تقييم الطريق وتفسيره",
            _route_answer(
                current
            ),
        )

    # =====================================
    # سؤال يجمع طريقة العمل والمصادر
    # =====================================

    if (
        "كيف يعمل" in nq
        and (
            "مصادر" in nq
            or "البيانات" in nq
        )
    ):

        return _reply(
            "كيف يعمل إشارة وما مصادره؟",
            FAQ[0]["answer"]
            + "\n\n"
            + FAQ[7]["answer"],
        )

    # =====================================
    # مطابقة الكلمات الواضحة
    # =====================================

    keyword_rules = [

        (
            (
                "الالوان",
                "لون",
                "الاحمر",
                "الاخضر",
                "برتقالي",
                "الخريطة",
                "رموز الاتصال",
                "الدوائر"
            ),
            2
        ),

        (
            (
                "المستويات",
                "التصنيف",
                "تصنيف",
                "الفئات",
                "ممتاز",
                "جيد جدا",
                "مقاطع ضعيفه"
            ),
            1
        ),

        (
            (
                "مسافه",
                "المسافات",
                "idw",
                "كيف تحسب",
                "قريبه"
            ),
            3
        ),

        (
            (
                "الضعف",
                "ضعف الشبكه",
                "انقطاع",
                "نقاط الضعف",
                "تنبؤ بالتغطيه",
                "توقع"
            ),
            4
        ),

        (
            (
                "تقنيات",
                "الادوات",
                "مكتبات",
                "برمجه",
                "sentence transformers"
            ),
            5
        ),

        (
            (
                "مدن",
                "المناطق المشموله",
                "نطاق الدراسه",
                "السعوديه كلها"
            ),
            6
        ),

        (
            (
                "المصادر",
                "مصادر",
                "من وين",
                "البيانات حقيقيه",
                "e-1",
                "e-2",
                "e-3",
                "e-4",
                "e-5",
                "zenodo",
                "mlab"
            ),
            7
        ),

        (
            (
                "كيف يعمل",
                "طريقه عمل",
                "اشرح النظام"
            ),
            0
        ),
    ]

    for keywords, faq_index in keyword_rules:

        if any(
            _normalize(word) in nq
            for word in keywords
        ):

            item = FAQ[
                faq_index
            ]

            return _reply(
                item["title"],
                item["answer"],
            )

    # =====================================
    # تشغيل البحث الدلالي
    # =====================================

    model = _get_model()

    if model is None:

        return _reply(
            "تعذر تشغيل البحث الدلالي",
            "الأسئلة الجاهزة ما زالت تعمل، "
            "لكن النموذج الدلالي غير متاح الآن. "
            "تأكد من تثبيت `sentence-transformers` "
            "والاتصال بالإنترنت عند أول تشغيل.",
            status="warning",
        )

    corpus = tuple(
        p
        for item in FAQ
        for p in item["questions"]
    )

    embeddings = _faq_vectors(
        corpus
    )

    if embeddings is None:

        return _reply(
            "تعذر البحث",
            "تعذر تجهيز فهرس الأسئلة.",
            status="warning"
        )

    query_vector = model.encode(
        q,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    scores = (
        np.asarray(embeddings)
        @ np.asarray(query_vector)
    )

    best_index = int(
        np.argmax(scores)
    )

    similarity = float(
        scores[best_index]
    )

    if similarity < threshold:

        return _reply(
            "لم أجد إجابة مناسبة للسؤال",
            "السؤال خارج المعلومات التي يستطيع "
            "المساعد شرحها حاليًا. "
            "يمكنك اختيار سؤال جاهز "
            "أو السؤال عن تحليل الطرق "
            "وتوقع ضعف سرعة الشبكة.",
            status="warning",
        )

    faq_index = next(
        i
        for i, item in enumerate(FAQ)
        if corpus[best_index] in item["questions"]
    )

    item = FAQ[
        faq_index
    ]

    return _reply(
        item["title"],
        item["answer"],
        similarity=similarity
    )


# =========================================
# التوافق مع الاستدعاءات القديمة
# =========================================

def search_answer(
    user_query,
    route_summaries=None,
    threshold=0.42
):

    result = get_answer(
        user_query,
        route_summaries,
        threshold
    )

    return (
        f"**{result['title']}**\n\n"
        f"{result['answer']}"
    )


# =========================================
# إصلاح اتجاه العربية مع الإنجليزية
# =========================================

def _format_mixed_direction(text):

    """
    الحفاظ على اتجاه الكلمات الإنجليزية
    وأسماء الملفات داخل النص العربي.
    """

    return re.sub(
        (
            r"[A-Za-z]"
            r"(?:[A-Za-z0-9+/-]|"
            r"[._-](?=[A-Za-z0-9]))*"
            r"(?: [A-Za-z]"
            r"(?:[A-Za-z0-9+/-]|"
            r"[._-](?=[A-Za-z0-9]))*)*"
        ),
        lambda match: (
            "\u2066"
            + match.group(0)
            + "\u2069"
        ),
        text,
    )


# =========================================
# واجهة مساعد إشارة الذكي
# =========================================

def render_qa_assistant(route_summaries=None):

    # =====================================
    # محاذاة المساعد من اليمين
    # =====================================

    st.markdown(
        dedent("""
        <style>

        div[data-testid="stPills"],
        div[data-testid="stPills"] label,
        div[data-testid="stPills"] button,
        div[data-testid="stPills"] [data-testid="stMarkdownContainer"],
        div[data-testid="stPills"] [role="group"],
        div[data-testid="stPills"] [role="radiogroup"],
        div[data-testid="stPills"] [data-testid="stButtonGroup"] {

            direction: rtl !important;
            text-align: right !important;
        }

        div[data-testid="stPills"] [role="group"],
        div[data-testid="stPills"] [role="radiogroup"],
        div[data-testid="stPills"] [data-testid="stButtonGroup"] {

            justify-content: flex-start !important;
            flex-direction: row !important;
            flex-wrap: wrap !important;
        }

        /* خانة السؤال */

        div[data-testid="stTextInput"],
        div[data-testid="stTextInput"] label,
        div[data-testid="stTextInput"] input,
        div[data-testid="stForm"] label {

            direction: rtl !important;
            text-align: right !important;
        }

        /* زر البحث */

        div[data-testid="stFormSubmitButton"] {

            direction: rtl !important;
            display: flex !important;
            justify-content: flex-start !important;
        }

        div[data-testid="stFormSubmitButton"] button {

            margin-left: auto !important;
            margin-right: 0 !important;
        }

        /* الإجابات والتنبيهات */

        div[data-testid="stAlert"],
        div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
        div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] li {

            direction: rtl !important;
            text-align: right !important;
            line-height: 2.0 !important;
        }

        div[data-testid="stAlert"]
        [data-testid="stMarkdownContainer"] code {

            direction: ltr !important;
            unicode-bidi: isolate !important;
        }

        /* النصوص التوضيحية */

        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] p {

            direction: rtl !important;
            text-align: right !important;
        }

        </style>
        """).strip(),
        unsafe_allow_html=True,
    )

    # =====================================
    # عنوان المساعد
    # =====================================

    st.markdown("---")

    st.markdown(
        '<h3 dir="rtl" style="text-align:right;'
        'margin-bottom:8px">'
        '🤖 مساعد إشارة الذكي | '
        '<span dir="ltr" style="unicode-bidi:isolate">'
        'ESHARAH AI'
        '</span>'
        '</h3>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div dir="rtl" style="text-align:right;'
        'opacity:.8;margin-bottom:8px">'
        'للأسئلة والتحليل الدلالي'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div dir="rtl" style="text-align:right;opacity:.8;'
        'font-size:.9rem;line-height:1.9;margin-bottom:16px">'
        'يفسر توقعات ضعف سرعة الإنترنت لكل مقطع من الطريق، '
        'ويعرض نتائج الطرق المحسوبة ومصادر البيانات.'
        '</div>',
        unsafe_allow_html=True,
    )

    # =====================================
    # الأسئلة الجاهزة
    # =====================================

    quick_questions = [
        "كيف يتوقع الذكاء الاصطناعي ضعف الشبكة؟",
        "قارن بين المسارات وأيها الأفضل حالياً؟",
        "تفاصيل مسار المويه الجديدة إلى الخرمة",
        "تفاصيل مسار الخرمة إلى ظلم",
        "تفاصيل مسار ظلم إلى المويه الجديدة",
        "وش المستويات الأربعة للتقييم؟",
        "ماذا تعني ألوان الخريطة؟",
        "كيف يعمل النظام وما هي المصادر؟",
        "كيف تُحسب المسافات والسرعات والخلية القريبة؟",
        "ما هي التقنيات والأدوات المستخدمة في المشروع؟",
        "ما هي المدن والطرق المشمولة في الدراسة؟",
    ]

    st.markdown(
        '<div dir="rtl" style="text-align:right;'
        'font-weight:600;margin-bottom:8px">'
        'اختر سؤالاً جاهزاً:'
        '</div>',
        unsafe_allow_html=True,
    )

    selected = st.pills(
        "اختر سؤالاً جاهزاً:",
        options=quick_questions,
        key="qa_quick_pills_v4",
        label_visibility="collapsed",
    )

    # =====================================
    # كتابة سؤال يدوي
    # =====================================

    with st.form("esharah_qa_form_v4"):

        typed = st.text_input(
            "أو اكتب سؤالك هنا:",
            placeholder=(
                "مثال: كيف يتنبأ النظام بضعف الشبكة؟"
            ),
        )

        submitted = st.form_submit_button(
            "بحث",
            type="primary"
        )

    # =====================================
    # حفظ السؤال الحالي
    # =====================================

    previous = st.session_state.get(
        "esharah_qa_previous_selection_v4"
    )

    if submitted and typed.strip():

        st.session_state[
            "esharah_qa_question_v4"
        ] = typed.strip()

    elif selected and selected != previous:

        st.session_state[
            "esharah_qa_question_v4"
        ] = selected

    st.session_state[
        "esharah_qa_previous_selection_v4"
    ] = selected

    current = st.session_state.get(
        "esharah_qa_question_v4"
    )

    if not current:

        return

    # =====================================
    # إظهار السؤال من اليمين
    # =====================================

    st.markdown(
        (
            '<div dir="rtl" '
            'style="text-align:right;margin-top:12px">'
            '<strong>السؤال:</strong> '
            f'{escape_html(current)}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    # =====================================
    # الحصول على الإجابة
    # =====================================

    result = get_answer(
        current,
        route_summaries=route_summaries
    )

    body = _format_mixed_direction(
        f"**{result['title']}**\n\n"
        f"{result['answer']}"
    )

    if result["status"] == "success":

        st.success(
            body,
            icon="✅"
        )

    else:

        st.warning(
            body,
            icon="⚠️"
        )

    # =====================================
    # نسبة التشابه إن كانت متاحة
    # =====================================

    if result["similarity"] is not None:

        st.markdown(
            (
                '<div dir="rtl" '
                'style="text-align:right;'
                'opacity:0.75;font-size:0.85rem;'
                'margin-top:8px">'
                'تشابه السؤال مع الأسئلة المعروفة: '
                f'{result["similarity"]:.0%} '
                '— ليس احتمال صحة التوقع '
                'أو دقة النموذج.'
                '</div>'
            ),
            unsafe_allow_html=True,
        )