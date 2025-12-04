import streamlit as st
import pandas as pd
import joblib

RAW_FEATURES = [
    "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Type_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Credit_History_Age",
    "Payment_of_Min_Amount", "Total_EMI_per_month",
    "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance",
]

VALID_CATS = {
    "Occupation": ["Accountant","Architect","Developer","Doctor","Engineer","HR_Manager","Lawyer","Manager","Media_Manager","Musician","Others","Scientist","Teacher","Writer"],
    "Type_of_Loan": ["auto loan","credit-builder loan","debt consolidation loan","home equity loan","mortgage loan","payday loan","personal loan","student loan","not specified","No Data"],
    "Credit_Mix": ["Bad","Good","Standard"],
    "Payment_of_Min_Amount": ["NM","No","Yes"],
    "Payment_Behaviour": ["High_spent_Large_value_payments","High_spent_Medium_value_payments","High_spent_Small_value_payments","Low_spent_Large_value_payments","Low_spent_Medium_value_payments","Low_spent_Small_value_payments"],
}

CLASS_MAP = {1: "poor", 2: "standard", 0: "good"}

# Các trường số thực
FLOAT_FIELDS = [
    "Annual_Income", "Monthly_Inhand_Salary", "Changed_Credit_Limit",
    "Outstanding_Debt", "Credit_Utilization_Ratio", 
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"
]

# Các trường số nguyên
INT_FIELDS = [
    "Age", "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Num_Credit_Inquiries", 
    "Credit_History_Age", "Interest_Rate"
]

# Định nghĩa lại class GetDummies giống hệt khi train
class GetDummies:
    def __init__(self, data_sep=',', col_name_sep='_'):
        self.data_sep     = data_sep
        self.col_name_sep = col_name_sep

    def fit(self, X, y=None): 
        object_cols = X.select_dtypes(include="object").columns
        self.dummy_cols = [col for col in object_cols if X[col].str.contains(self.data_sep, regex=True).any()]
        self.dummy_prefix = [
            ''.join(map(lambda x: x[0], col.split(self.col_name_sep))) if self.col_name_sep in col else col[:2]
            for col in self.dummy_cols
        ]
        dummy_X = X.copy()
        for col, pre in zip(self.dummy_cols, self.dummy_prefix):
            dummy_X = dummy_X.join(dummy_X[col].str.get_dummies(sep=self.data_sep).add_prefix(pre+self.col_name_sep))
        if self.dummy_cols:
            dummy_X.drop(columns=self.dummy_cols, inplace=True)
        self.columns = dummy_X.columns
        return self

    def transform(self, X, y=None):
        dummy_X = X.copy()
        for col, pre in zip(self.dummy_cols, self.dummy_prefix):
            dummy_X = dummy_X.join(dummy_X[col].str.get_dummies(sep=self.data_sep).add_prefix(pre+self.col_name_sep))
        if self.dummy_cols:
            dummy_X.drop(columns=self.dummy_cols, inplace=True)
        dummy_X = dummy_X.reindex(columns=self.columns, fill_value=0)
        return dummy_X

@st.cache_resource
def load_pipeline(path: str = 'pipeline.pkl'):
    pipeline = joblib.load(path)
    return pipeline

def user_input_features() -> pd.DataFrame:
    st.sidebar.markdown("## 📝 Nhập thông tin khách hàng")
    data = {}
    for feat in RAW_FEATURES:
        if feat in VALID_CATS:
            data[feat] = st.sidebar.selectbox(f"**{feat}**", VALID_CATS[feat])
        elif feat in FLOAT_FIELDS:
            data[feat] = st.sidebar.number_input(
                f"**{feat}**", min_value=0.0, max_value=1e7, value=0.0, step=0.1, format="%.2f"
            )
        elif feat in INT_FIELDS:
            data[feat] = st.sidebar.number_input(
                f"**{feat}**", min_value=0, max_value=1000, value=0, step=1
            )
        else:
            # fallback cho các trường khác nếu có
            data[feat] = st.sidebar.text_input(f"**{feat}**", "")
    return pd.DataFrame([data])

def main():
    st.markdown(
        "<h1 style='text-align:center; color:#4F8BF9;'>📊 Credit Scoring Web App</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align:center; color:gray;'>Nhập các thông tin khách hàng ở sidebar để xem phân lớp tín dụng và rủi ro tín dụng</div>",
        unsafe_allow_html=True
    )
    st.write("---")

    input_df = user_input_features()
    pipe = load_pipeline()

    col1, col2 = st.columns([6, 4])  # Cột thông tin khách hàng rộng hơn

    with col1:
        st.markdown("### 🧾 Thông tin khách hàng vừa nhập")
        st.dataframe(input_df.T, use_container_width=True, height=820)  # Tăng chiều cao để không bị cắt

    with col2:
        if st.sidebar.button("🔍 Dự đoán"):
            prediction = pipe.predict(input_df)
            proba = pipe.predict_proba(input_df)
            pred_label = CLASS_MAP.get(int(prediction[0]), str(prediction[0]))
            st.markdown("### 🎯 <span style='color:#4F8BF9'>Kết quả dự đoán</span>", unsafe_allow_html=True)
            if pred_label == "poor":
                st.markdown("⚠️ <span style='color:red; font-weight:bold;'>Cảnh báo:</span> Khách hàng này có <b>rủi ro cao</b> khi cho vay!", unsafe_allow_html=True)
            elif pred_label == "standard":
                st.markdown("⚠️ Khách hàng có <b>rủi ro trung bình</b>.", unsafe_allow_html=True)
            else:
                st.markdown("✅ Khách hàng có <b>rủi ro thấp</b>.", unsafe_allow_html=True)
            st.markdown("#### Xác suất từng lớp:")
            st.write({CLASS_MAP[i]: f"{p:.2%}" for i, p in enumerate(proba[0])})
        else:
            st.info("Vui lòng nhập thông tin và bấm nút **Dự đoán**.", icon="ℹ️")
if __name__ == "__main__":
    main()