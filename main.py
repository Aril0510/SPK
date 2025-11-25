import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SPK SAW-TOPSIS", layout="wide")

# =====================================================
# SESSION STATE
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

def go_to_input():
    st.session_state.page = "input"


# =====================================================
# 1️⃣ DASHBOARD
# =====================================================
if st.session_state.page == "dashboard":

    st.markdown("""
    <h1 style='text-align:center; color:#2E86C1;'>📊 Sistem Pendukung Keputusan — SAW → TOPSIS</h1>
    <p style='text-align:center; font-size:18px;'>
    Aplikasi ini memproses perhitungan SAW dan TOPSIS secara otomatis dan dinamis.
    Anda dapat menentukan kriteria, tipe, dan bobot sesuai kebutuhan.
    </p>
    """, unsafe_allow_html=True)

    st.write("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("⚙ **Metode SAW**\n- Normalisasi Benefit/Cost\n- Pembobotan r × w")
    with col2:
        st.warning("📌 **Metode TOPSIS**\n- A+, A−\n- Nilai Preferensi")
    with col3:
        st.success("📁 **Input Manual**\n- Pilih kriteria\n- Tentukan bobot\n- Pilih Benefit/Cost")

    st.write("---")
    st.button("🚀 MULAI PERHITUNGAN", on_click=go_to_input)


# =====================================================
# 2️⃣ HALAMAN INPUT
# =====================================================
elif st.session_state.page == "input":

    st.markdown("## 📁 Upload Dataset & Masukkan Kriteria")

   # =====================================================
# TOMBOL DOWNLOAD DATASET CONTOH (DARI GITHUB)
# =====================================================
st.info("📥 Contoh Dataset dapat diunduh di bawah ini:")

GITHUB_DATASET_URL = "https://raw.githubusercontent.com/Aril0510/SPK/main/Dataset_SPK_ekstrakurikuler.xlsx"

st.download_button(
    label="📄 Download Contoh Dataset",
    data=requests.get(GITHUB_DATASET_URL).content,
    file_name="Dataset_SPK_ekstrakurikuler.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


    # =====================================================

    uploaded = st.file_uploader("Upload Dataset (xlsx/csv)", type=["xlsx", "csv"])

    if uploaded:

        # baca dataset
        if uploaded.name.endswith("xlsx"):
            df_raw = pd.read_excel(uploaded)
        else:
            df_raw = pd.read_csv(uploaded)

        st.subheader("📄 Dataset Original")
        st.dataframe(df_raw.head())

        # =====================================================
        # VALIDASI: hanya kolom numerik boleh jadi kriteria
        # =====================================================
        numeric_columns = df_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if len(numeric_columns) == 0:
            st.error("❌ Dataset tidak memiliki kolom numerik.")
            st.stop()

        st.success(f"✔ {len(numeric_columns)} kolom numerik ditemukan.")

        st.subheader("🔧 Pilih Kolom Kriteria")

        # jumlah kriteria
        num_criteria = st.number_input(
            "Berapa jumlah kriteria?",
            min_value=1,
            max_value=len(numeric_columns),
            value=1,
            step=1
        )

        criteria_list = []
        criteria_type = []
        weights_list = []

        st.write("### ➤ Masukkan Kriteria")

        for i in range(num_criteria):
            st.markdown(f"#### Kriteria {i+1}")
            c1, c2, c3 = st.columns(3)

            with c1:
                col = st.selectbox("Pilih kolom (Hanya Numerik)", numeric_columns, key=f"crit_{i}")
            with c2:
                t = st.selectbox("Tipe", ["Benefit", "Cost"], key=f"type_{i}")
            with c3:
                w = st.number_input("Bobot", 0.0, 1.0, 0.1, 0.01, key=f"w_{i}")

            criteria_list.append(col)
            criteria_type.append(t)
            weights_list.append(w)

        weights = np.array(weights_list)

        # =====================================================
        # VALIDASI BOBOT
        # =====================================================
        total_weight = weights.sum()
        if round(total_weight, 4) != 1.0:
            st.error(f"❌ Total bobot = {total_weight:.2f} (harus = 1.00)")
            st.stop()

        df = df_raw.copy()

        # pastikan numerik
        for col in criteria_list:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


        # =====================================================
        # SAW FUNCTION
        # =====================================================
        def compute_saw(df):
            X = df[criteria_list].copy()
            norm = pd.DataFrame(index=X.index, columns=criteria_list)

            for i, col in enumerate(criteria_list):
                col_data = X[col].astype(float)
                max_val = col_data.max()
                min_val = col_data.min()

                if criteria_type[i] == "Benefit":
                    norm[col] = col_data / max_val if max_val != 0 else 0
                else:
                    col_safe = col_data.replace(0, np.nan)
                    norm[col] = (min_val / col_safe).fillna(0)

            weighted = norm.multiply(weights, axis=1)
            scores = weighted.sum(axis=1)
            return norm, weighted, scores


        # =====================================================
        # TOPSIS FUNCTION
        # =====================================================
        def compute_topsis(weighted):
            ideal_pos = weighted.max(axis=0)
            ideal_neg = weighted.min(axis=0)

            D_plus = np.sqrt(((weighted - ideal_pos) ** 2).sum(axis=1))
            D_minus = np.sqrt(((weighted - ideal_neg) ** 2).sum(axis=1))

            pref = D_minus / (D_plus + D_minus)
            return pref, D_plus, D_minus, ideal_pos, ideal_neg


        # =====================================================
        # PROSES SAW–TOPSIS
        # =====================================================
        if st.button("🔍 Proses SAW → TOPSIS"):

            norm_saw, weighted_saw, saw_scores = compute_saw(df)

            df_scores = df.copy()
            df_scores["SAW_Score"] = saw_scores

            df_saw_rank = df_scores.sort_values("SAW_Score", ascending=False).reset_index(drop=True)
            df_saw_rank.insert(0, "Rank_SAW", range(1, len(df_saw_rank) + 1))

            preferensi, D_plus, D_minus, ideal_pos, ideal_neg = compute_topsis(weighted_saw)

            df_scores["TOPSIS_Score"] = preferensi
            df_topsis_rank = df_scores.sort_values("TOPSIS_Score", ascending=False).reset_index(drop=True)
            df_topsis_rank.insert(0, "Rank_TOPSIS", range(1, len(df_topsis_rank) + 1))

            st.subheader("📌 Hasil TOPSIS")
            st.dataframe(df_topsis_rank)

