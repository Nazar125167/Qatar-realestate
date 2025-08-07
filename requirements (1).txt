import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="توقع إيجار العقارات في قطر", layout="wide")

st.title("🏡 تطبيق توقع إيجار العقارات في قطر")
st.markdown("أدخل خصائص العقار أدناه للحصول على تقدير للإيجار الشهري")

# تحميل النموذج
try:
    model = joblib.load('xgboost_model.pkl')
    features = joblib.load('features.pkl')
except:
    st.error("❌ فشل تحميل النموذج. تأكد من وجود الملفات الصحيحة.")
    st.stop()

# قائمة الخصائص
locations = ['الدوحة', 'الخور', 'الوكرة', 'الريان', 'أم صلال']
property_types = ['شقة', 'فيلا', 'استوديو']
furnishing_types = ['مفروشة بالكامل', 'غير مفروشة', 'شبه مفروشة']

with st.form("prediction_form"):
    st.subheader("تفاصيل العقار")
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("الموقع", locations)
        bedrooms = st.number_input("عدد غرف النوم", min_value=1, max_value=10, value=2)
        parking = st.number_input("عدد مواقف السيارات", min_value=0, max_value=5, value=1)
    with col2:
        property_type = st.selectbox("نوع العقار", property_types)
        furnishing = st.selectbox("نوع التأثيث", furnishing_types)
        bathrooms = st.number_input("عدد الحمامات", min_value=1, max_value=10, value=2)

    submitted = st.form_submit_button("توقع الإيجار")

    if submitted:
        user_input = pd.DataFrame([{
            'الموقع': location,
            'نوع_العقار': property_type,
            'عدد_غرف_النوم': bedrooms,
            'عدد_الحمامات': bathrooms,
            'عدد_مواقف_السيارات': parking,
            'نوع_التأثيث': furnishing,
        }])
        
        user_input_encoded = pd.get_dummies(user_input, drop_first=True)
        user_input_encoded = user_input_encoded.reindex(columns=features, fill_value=0)

        predicted_price = model.predict(user_input_encoded)[0]
        
        st.success(f"💰 الإيجار الشهري المتوقع هو: {int(predicted_price):,} ريال قطري")
