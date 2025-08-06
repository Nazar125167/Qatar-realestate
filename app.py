import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="توقع الإيجار في قطر", layout="wide")

st.title("🏘️ توقع الإيجارات العقارية في قطر")
st.markdown("أدخل خصائص العقار أدناه للحصول على تقدير للإيجار الشهري عبر النموذجين")

try:
    # 🚨 تم تحميل النماذج بالأسماء الإنجليزية الصحيحة 🚨
    xgb = joblib.load("xgboost_model.pkl")
    lr = joblib.load("linear_model.pkl")
    features = joblib.load("features_X.pkl")
except:
    st.error("❌ فشل تحميل النماذج. تأكد من وجود الملفات في نفس مجلد التطبيق.")
    st.stop()

with st.form("form"):
    st.subheader("تفاصيل العقار")
    col1, col2 = st.columns(2)
    with col1:
        bed = st.slider("عدد غرف النوم", 1, 6, 3)
        bath = st.slider("عدد الحمامات", 1, 4, 2)
        area = st.number_input("المساحة بالمتر", min_value=40, max_value=600, value=150)
        year = st.number_input("سنة البناء", min_value=1995, max_value=2023, value=2010)
        parking = st.selectbox("عدد مواقف السيارات", [0, 1, 2])
    with col2:
        loc = st.selectbox("المنطقة", ['اللؤلؤة','الخليج الغربي','لوسيل','المناصير','الدفنة','الوكرة','الريان','أم صلال','الوعب','الغرافة'])
        ptype = st.selectbox("نوع العقار", ['شقة','فيلا','استوديو','بنتهاوس','غرفة وصالة','توين هاوس'])
        furnished = st.selectbox("الحالة", ['مفروشة بالكامل','غير مفروشة','شبه مفروشة'])
        view = st.selectbox("الإطلالة", ['إطلالة بحر','إطلالة مدينة','إطلالة حديقة','لا يوجد'])
        floor = st.number_input("الطابق", min_value=0, max_value=20, value=2)

    submitted = st.form_submit_button("تنفيذ التنبؤ")

    if submitted:
        user_input = pd.DataFrame([{
            'غرف_نوم': bed, 'حمامات': bath, 'المساحة_متر': area, 'سنة_البناء': year,
            'مواقف_سيارات': parking, 'الطابق': floor, 'المنطقة': loc,
            'نوع_العقار': ptype, 'مفروش': furnished, 'الإطلالة': view
        }])

        user_input_encoded = pd.get_dummies(user_input, columns=['المنطقة', 'نوع_العقار', 'مفروش', 'الإطلالة'], drop_first=True)
        user_input_encoded = user_input_encoded.reindex(columns=features, fill_value=0)
        
        if 'الطابق' in user_input_encoded.columns:
            user_input_encoded['الطابق'] = user_input_encoded['الطابق'].fillna(0)

        rent_xgb = xgb.predict(user_input_encoded)[0]
        rent_lr = lr.predict(user_input_encoded)[0]

        st.success(f"📌 تقدير الإيجار عبر XGBoost: {int(rent_xgb):,} ريال قطري")
        st.info(f"📌 تقدير الإيجار عبر الانحدار الخطي: {int(rent_lr):,} ريال قطري")
