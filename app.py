import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Forecasts for Gs", layout="wide", page_icon="💎")

# --- 2. PREMIUM DESIGN SYSTEM (CSS) ---
st.markdown("""
    <style>
        /* IMPORT ROBOTO CONDENSED */
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@300;400;700&display=swap');

        /* GLOBAL FONT */
        html, body, [class*="css"], h1, h2, h3, h4, h5, h6, span, p, label, button, input, .stMarkdown {
            font-family: 'Roboto Condensed', sans-serif !important;
        }

        /* HIGH CONTRAST COLOR SCHEME */
        .stApp { background-color: #1C3144; } /* Main Blue */

        /* Dark Navy Sidebar & Header */
        [data-testid="stSidebar"], header[data-testid="stHeader"] {
            background-color: #0f1621 !important;
            border-bottom: 1px solid #333;
        }
        [data-testid="stSidebar"] { border-right: 1px solid #0f1621; }

        /* GLASSMORPHISM CARDS */
        div.css-1r6slb0, div.stDataFrame, [data-testid="stMetric"], div[data-testid="stExpander"], div.block-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: white;
            padding: 20px;
        }
        .stDataFrame { background-color: #152232; }

        /* TYPOGRAPHY */
        h1 {
            font-weight: 700; text-transform: uppercase; letter-spacing: 2px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5); font-size: 3rem;
        }

        /* INPUT FIELDS */
        .stSelectbox > div > div, .stDateInput > div > div, .stNumberInput > div > div, .stSlider > div > div {
            background-color: rgba(0, 0, 0, 0.3) !important;
            border: 1px solid #4a6fa5 !important;
            color: white !important;
            border-radius: 4px;
        }

        /* BUTTONS - LUXURY GRADIENT */
        .stButton>button {
            background: linear-gradient(90deg, #c0392b 0%, #8e44ad 100%);
            color: white; border: none; border-radius: 4px; height: 50px;
            text-transform: uppercase; letter-spacing: 2px; font-weight: 700; width: 100%;
            transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(142, 68, 173, 0.5);
        }

        /* METRICS */
        [data-testid="stMetricLabel"] { color: #bdc3c7 !important; text-transform: uppercase; font-size: 0.8rem; }
        [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700; font-size: 2rem; }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1 { color: #bdc3c7 !important; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<h1 style="text-align: left; color: white;">Forecasts for Gs</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color: #aab7c4; font-size: 1.1rem; letter-spacing: 1px; margin-top: -15px;">AUTO-ML & CRYSTAL BALL ENGINE</p>',
    unsafe_allow_html=True)
st.write("")

# --- STATE ---
if 'model' not in st.session_state: st.session_state['model'] = None
if 'model_name' not in st.session_state: st.session_state['model_name'] = ""
if 'model_score' not in st.session_state: st.session_state['model_score'] = 0
if 'encoders' not in st.session_state: st.session_state['encoders'] = {}
if 'trained_cols' not in st.session_state: st.session_state['trained_cols'] = []
if 'time_cols_map' not in st.session_state: st.session_state['time_cols_map'] = {}
if 'cat_cols' not in st.session_state: st.session_state['cat_cols'] = []
if 'original_date_col' not in st.session_state: st.session_state['original_date_col'] = None


# --- AUTO-ML LOGIC (THE BATTLE) ---
def train_auto_model(df, target_col):
    # 1. Prepare Data
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col])

    encoders = {}
    processed_cols = []
    time_map = {}
    cat_cols_found = []
    found_date_col = None

    # Date Detection
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.datetime64) or 'date' in col.lower() or 'time' in col.lower():
            try:
                X[col] = pd.to_datetime(X[col], errors='coerce')
                if not X[col].isna().all():
                    found_date_col = col
                    break
            except:
                continue

    if found_date_col:
        X[found_date_col] = X[found_date_col].fillna(pd.Timestamp.today())
        col_y, col_m, col_d, col_dow = found_date_col + '_y', found_date_col + '_m', found_date_col + '_d', found_date_col + '_dw'
        X[col_y] = X[found_date_col].dt.year
        X[col_m] = X[found_date_col].dt.month
        X[col_d] = X[found_date_col].dt.day
        X[col_dow] = X[found_date_col].dt.dayofweek
        time_map = {'year': col_y, 'month': col_m, 'day': col_d, 'dow': col_dow}
        processed_cols.extend([col_y, col_m, col_d, col_dow])
        X = X.drop(columns=[found_date_col])

    # Encoding
    for col in X.columns:
        if col in processed_cols: continue
        if X[col].dtype == 'object' and X[col].nunique() > len(X) * 0.95: continue
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = X[col].astype(str).replace('nan', 'Unknown')
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
            processed_cols.append(col)
            cat_cols_found.append(col)
        else:
            processed_cols.append(col)

    X_final = X[processed_cols].fillna(0)

    # 2. AUTO-ML BATTLE
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    models_to_test = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,       # Mai stabil
            max_depth=15,           # Limită rezonabilă
            min_samples_split=5,    # Previne noduri prea specifice
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost AI": xgb.XGBRegressor(
            n_estimators=300,       # Suficienți arbori să învețe
            learning_rate=0.05,     # Învață fin, nu brusc
            max_depth=6,            # Adâncimea standard de aur
            subsample=0.8,          # Folosește 80% din date la fiecare pas (variație)
            colsample_bytree=0.8,   # Folosește 80% din coloane (variație)
            random_state=42,
            n_jobs=-1
        )
    }

    best_model = None
    best_name = ""
    best_score = -np.inf

    for name, model in models_to_test.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    best_model.fit(X_final, y)  # Retrain on full data

    return best_model, best_name, best_score, encoders, processed_cols, time_map, cat_cols_found, found_date_col


# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Data Source")
    uploaded_file = st.file_uploader("Upload CSV / Excel", type=['csv', 'xlsx'])

    df = pd.DataFrame()
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass

        st.success(f"Loaded {len(df)} rows")
        st.divider()
        st.header("2. Objective")
        target = st.selectbox("Predict Target", df.columns.tolist(), index=len(df.columns) - 1)

        st.write("")
        if st.button("RUN AUTO-ML BATTLE"):
            with st.spinner("⚡ Bots Fighting (Linear vs Forest vs XGBoost)..."):
                model, m_name, m_score, enc, cols, t_map, c_cols, date_orig = train_auto_model(df, target)
                st.session_state['model'] = model
                st.session_state['model_name'] = m_name
                st.session_state['model_score'] = m_score
                st.session_state['encoders'] = enc
                st.session_state['trained_cols'] = cols
                st.session_state['time_cols_map'] = t_map
                st.session_state['cat_cols'] = c_cols
                st.session_state['target_col'] = target
                st.session_state['original_date_col'] = date_orig
                st.success(f"Winner: {m_name}")

# --- DASHBOARD ---
if st.session_state['model']:

    # KPI SECTION
    st.markdown("### Executive Summary")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    avg_val = df[st.session_state['target_col']].mean() if pd.api.types.is_numeric_dtype(
        df[st.session_state['target_col']]) else 0
    date_col = st.session_state['original_date_col']
    date_info = "N/A"
    if date_col:
        min_d = df[date_col].min().strftime('%d %b %Y')
        max_d = df[date_col].max().strftime('%d %b %Y')
        date_info = f"{min_d} - {max_d}"

    kpi1.metric("Active Model", st.session_state['model_name'], "WINNER")
    kpi2.metric("Accuracy", f"{st.session_state['model_score']:.1%}")
    kpi3.metric("Avg Target", f"{avg_val:,.1f}")
    kpi4.metric("Timeline", date_info)

    st.write("")

    # MAIN COLUMNS
    col_input, col_result = st.columns([1, 2], gap="medium")

    # --- INPUT PANEL ---
    with col_input:
        st.subheader("Configuration")

        # 1. DATE & CRYSTAL BALL
        manual_date = st.date_input("Start Date", datetime.today())

        # Crystal Ball Slider
        st.write("")
        horizon = st.slider("🔮 Crystal Ball Horizon (Days)", min_value=1, max_value=30, value=1,
                            help="Predict trend for upcoming days")

        st.write("")

        # 2. FILTERS
        cat_cols = st.session_state['cat_cols']
        selected_values = {}
        df_filtered = df.copy()

        if cat_cols:
            hierarchy = st.multiselect("Filters", cat_cols, default=cat_cols)
            for col_name in hierarchy:
                vals = sorted(df_filtered[col_name].astype(str).unique())
                sel = st.selectbox(f"{col_name}", vals)
                selected_values[col_name] = sel
                df_filtered = df_filtered[df_filtered[col_name].astype(str) == sel]

        # 3. NUMERICS
        st.write("")
        input_nums = {}
        time_cols_vals = list(st.session_state['time_cols_map'].values())
        trained = st.session_state['trained_cols']
        num_cols_to_show = [c for c in trained if c not in time_cols_vals and c not in st.session_state['encoders']]

        if num_cols_to_show:
            st.markdown("**Numeric Params**")
            for col in num_cols_to_show:
                def_val = float(
                    df_filtered[col].mean()) if not df_filtered.empty and col in df_filtered.columns else 0.0
                input_nums[col] = st.number_input(f"{col}", value=def_val)

        st.write("")
        if horizon > 1:
            calc_btn = st.button("ACTIVATE CRYSTAL BALL")
        else:
            calc_btn = st.button("PREDICT SINGLE DAY")

    # --- RESULTS PANEL ---
    with col_result:
        if calc_btn:
            t_map = st.session_state['time_cols_map']

            # --- SINGLE DAY PREDICTION ---
            if horizon == 1:
                final_input = pd.DataFrame()
                # Time
                if t_map:
                    if 'year' in t_map: final_input[t_map['year']] = [manual_date.year]
                    if 'month' in t_map: final_input[t_map['month']] = [manual_date.month]
                    if 'day' in t_map: final_input[t_map['day']] = [manual_date.day]
                    if 'dow' in t_map: final_input[t_map['dow']] = [manual_date.weekday()]

                # Categories
                for col in st.session_state['trained_cols']:
                    if col in st.session_state['encoders']:
                        le = st.session_state['encoders'][col]
                        val_str = str(selected_values.get(col, "Unknown"))
                        try:
                            final_input[col] = le.transform([val_str])
                        except:
                            final_input[col] = 0

                # Numerics
                for col, val in input_nums.items(): final_input[col] = [val]

                final_input = final_input[st.session_state['trained_cols']]
                res = st.session_state['model'].predict(final_input)[0]

                # Save single result
                st.session_state['last_result'] = res
                st.session_state['last_date'] = manual_date
                st.session_state['crystal_data'] = None  # Clear crystal ball

            # --- CRYSTAL BALL (MULTI-DAY LOOP) ---
            else:
                future_preds = []
                current_d = manual_date

                for i in range(horizon):
                    # Construct row for current_d
                    row_data = {}

                    # Time update
                    if t_map:
                        if 'year' in t_map: row_data[t_map['year']] = [current_d.year]
                        if 'month' in t_map: row_data[t_map['month']] = [current_d.month]
                        if 'day' in t_map: row_data[t_map['day']] = [current_d.day]
                        if 'dow' in t_map: row_data[t_map['dow']] = [current_d.weekday()]

                    # Static Data (Categories + Numerics)
                    for col in st.session_state['trained_cols']:
                        if col in st.session_state['encoders']:
                            le = st.session_state['encoders'][col]
                            val_str = str(selected_values.get(col, "Unknown"))
                            try:
                                row_data[col] = [le.transform([val_str])[0]]
                            except:
                                row_data[col] = [0]
                        elif col in input_nums:
                            row_data[col] = [input_nums[col]]

                    # Build DF
                    row_df = pd.DataFrame(row_data)
                    # Add missing columns (numerics that are not time)
                    for col in st.session_state['trained_cols']:
                        if col not in row_df.columns:
                            row_df[col] = 0

                    row_df = row_df[st.session_state['trained_cols']]
                    pred_val = st.session_state['model'].predict(row_df)[0]

                    future_preds.append({'Date': current_d, 'Predicted Value': pred_val})

                    # Next day
                    current_d += timedelta(days=1)

                st.session_state['crystal_data'] = pd.DataFrame(future_preds)
                st.session_state['last_result'] = None  # Hide single result

        # --- DISPLAY LOGIC ---

        # SCENARIO A: CRYSTAL BALL CHART
        if st.session_state.get('crystal_data') is not None:
            st.markdown(f"""
            <div style="background: rgba(0, 210, 211, 0.1); border: 1px solid #00d2d3; padding: 20px; border-radius: 8px; text-align: center;">
                <h4 style="color: #00d2d3; margin:0; letter-spacing: 2px;">CRYSTAL BALL ACTIVATED</h4>
                <p style="color: white;">Projection for next <b>{len(st.session_state['crystal_data'])} days</b></p>
            </div>
            """, unsafe_allow_html=True)

            cb_df = st.session_state['crystal_data']
            fig_cb = px.line(cb_df, x='Date', y='Predicted Value', title="FUTURE TREND PROJECTION",
                             template="plotly_dark", markers=True)
            fig_cb.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto Condensed", color="white")
            )
            fig_cb.update_traces(line_color='#00d2d3', line_width=4, marker_size=10)
            st.plotly_chart(fig_cb, use_container_width=True)

            # Show sum
            total_sum = cb_df['Predicted Value'].sum()
            st.info(f"💰 Total Predicted Volume for Period: {total_sum:,.2f}")

        # SCENARIO B: SINGLE RESULT
        elif st.session_state.get('last_result') is not None:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.2); padding: 30px; border-radius: 8px; text-align: center; backdrop-filter: blur(10px);">
                <h4 style="color: #aab7c4; margin:0; letter-spacing: 3px;">PROJECTED OUTCOME</h4>
                <h1 style="color: #fff; font-size: 5rem; margin: 10px 0; text-shadow: 0 0 20px rgba(255,255,255,0.3);">{st.session_state['last_result']:,.0f}</h1>
                <p style="color: #fff; margin:0; letter-spacing: 1px;">DATE: <b>{st.session_state['last_date'].strftime('%d %B %Y')}</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            tab1, tab2 = st.tabs(["HISTORICAL", "DRIVERS"])
            with tab1:
                date_col = st.session_state['original_date_col']
                if date_col and not df_filtered.empty:
                    try:
                        hist_data = df_filtered.groupby(date_col)[st.session_state['target_col']].mean().reset_index()
                        fig = px.area(hist_data, x=date_col, y=st.session_state['target_col'], template="plotly_dark")
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font=dict(family="Roboto Condensed", color="white"))
                        fig.update_traces(line_color='#00d2d3', fillcolor='rgba(0, 210, 211, 0.2)')
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.warning("Insufficient Data")
            with tab2:
                if hasattr(st.session_state['model'], 'feature_importances_'):
                    imp = st.session_state['model'].feature_importances_
                    cols = st.session_state['trained_cols']
                    feat_df = pd.DataFrame({'Factor': cols, 'Impact': imp}).sort_values(by='Impact', ascending=True)
                    fig_bar = px.bar(feat_df, x='Impact', y='Factor', orientation='h', template="plotly_dark")
                    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font=dict(family="Roboto Condensed", color="white"))
                    fig_bar.update_traces(marker_color='#5f27cd')
                    st.plotly_chart(fig_bar, use_container_width=True)

        # SCENARIO C: IDLE
        else:
            st.info("System Standby. Awaiting Input.")
            try:
                fig_dist = px.histogram(df, x=st.session_state['target_col'], nbins=30, template="plotly_dark",
                                        title="Global Distribution")
                fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font=dict(family="Roboto Condensed", color="white"))
                st.plotly_chart(fig_dist, use_container_width=True)
            except:
                pass

else:
    # LANDING PAGE
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: white;">
        <h2 style="letter-spacing: 5px; opacity: 0.7;">SYSTEM OFFLINE</h2>
        <h1 style="font-size: 3rem; margin-top: 10px;">PLEASE UPLOAD DATASET</h1>
        <div style="margin-top: 50px; display: flex; justify-content: center; gap: 20px;">
             <div style="border: 1px solid #444; padding: 20px; width: 150px; border-radius: 8px;">1. UPLOAD</div>
             <div style="border: 1px solid #444; padding: 20px; width: 150px; border-radius: 8px;">2. AUTO-ML</div>
             <div style="border: 1px solid #444; padding: 20px; width: 150px; border-radius: 8px;">3. CRYSTAL BALL</div>
        </div>
    </div>
    """, unsafe_allow_html=True)