import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="🌤 Weather Forecasting Dashboard",
                   layout="wide",
                   page_icon="🌦")

st.markdown("<h1 style='text-align:center;color:#0078D7;'>🌦 Weather Forecasting Data Analysis</h1>", unsafe_allow_html=True)
st.write("This interactive dashboard allows you to upload weather data, visualize it, and train a Linear Regression model to forecast temperature trends.")

uploaded_file = st.file_uploader("📂 Upload your weather dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset uploaded successfully!")
    st.sidebar.markdown("---")
    tab = st.sidebar.radio("📊 Select Section", ["Data Overview", "Visualization", "Model Training", "Results & Insights"])

    # -------------------- TAB 1: DATA OVERVIEW --------------------
    if tab == "Data Overview":
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.write("### Data Summary")
        st.write(df.describe())
        st.write("### Missing Values")
        st.write(df.isnull().sum())

    # -------------------- TAB 2: VISUALIZATION --------------------
    elif tab == "Visualization":
        st.subheader("📈 Exploratory Data Analysis")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not num_cols:
            st.warning("No numeric columns found in the dataset.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                feat = st.selectbox("Select feature to plot distribution", num_cols)
                fig = px.histogram(df, x=feat, nbins=30, color_discrete_sequence=["#00A6FB"], title=f"Distribution of {feat}")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.write("### Correlation Heatmap")
                fig2, ax = plt.subplots(figsize=(6,4))
                sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
                st.pyplot(fig2)

    # -------------------- TAB 3: MODEL TRAINING --------------------
    elif tab == "Model Training":
        st.subheader("🤖 Train Linear Regression Model")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        target = st.selectbox("Select Target (Y)", num_cols)
        features = st.multiselect("Select Features (X)", [c for c in num_cols if c != target])

        if st.button("Train Model"):
            if not features:
                st.warning("Please select at least one feature.")
            else:
                data = df.fillna(df.mean())
                X, y = data[features], data[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse, r2 = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

                st.session_state["model"] = {"model": model, "y_test": y_test, "y_pred": y_pred, "r2": r2, "mse": mse, "features": features}

                st.success("Model trained successfully.")
                st.metric("R² Score", f"{r2:.3f}")
                st.metric("MSE", f"{mse:.3f}")

    # -------------------- TAB 4: RESULTS & INSIGHTS --------------------
    elif tab == "Results & Insights":
        if "model" not in st.session_state:
            st.warning("Train the model first.")
        else:
            res = st.session_state["model"]
            st.subheader("Actual vs Predicted Temperature")
            fig3 = px.scatter(x=res["y_test"], y=res["y_pred"], labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=["#8856a7"])
            fig3.add_shape(type="line", x0=res["y_test"].min(), y0=res["y_test"].min(),
                           x1=res["y_test"].max(), y1=res["y_test"].max(),
                           line=dict(color="red", dash="dash"))
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Feature Coefficients")
            coefs = pd.DataFrame({"Feature": res["features"], "Coefficient": res["model"].coef_})
            st.dataframe(coefs, use_container_width=True)

            st.markdown("### 🧠 Insights")
            st.write(f"""
            - Model R² Score: **{res['r2']:.3f}**  
            - Mean Squared Error: **{res['mse']:.3f}**  
            - Features with high coefficients strongly influence temperature changes.  
            - A balanced coefficient pattern indicates model stability.  
            """)
else:
    st.info("👆 Please upload your dataset to begin.")
