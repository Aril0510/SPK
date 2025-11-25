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

    uploaded = st.file_uploader("Upload Dataset (xlsx/csv)", type=["xlsx", "csv"])

    if uploaded:

        # baca dataset
        if uploaded.name.endswith("xlsx"):
            df_raw = pd.read_excel(uploaded)
        else:
            df_raw = pd.read_csv(uploaded)

        st.subheader("📄 Dataset Original")
        st.dataframe(df_raw.head())

        st.subheader("🔧 Pilih Kolom Kriteria")

        all_columns = df_raw.columns.tolist()

        # jumlah kriteria
        num_criteria = st.number_input(
            "Berapa jumlah kriteria?",
            min_value=1, max_value=20, value=5, step=1
        )

        criteria_list = []
        criteria_type = []
        weights_list = []

        st.write("### ➤ Masukkan Kriteria")

        for i in range(num_criteria):
            st.markdown(f"#### Kriteria {i+1}")
            c1, c2, c3 = st.columns(3)

            with c1:
                col = st.selectbox("Pilih kolom", all_columns, key=f"crit_{i}")
            with c2:
                t = st.selectbox("Tipe", ["Benefit", "Cost"], key=f"type_{i}")
            with c3:
                w = st.number_input("Bobot", 0.0, 1.0, 0.1, 0.01, key=f"w_{i}")

            criteria_list.append(col)
            criteria_type.append(t)
            weights_list.append(w)

        weights = np.array(weights_list)

        # =====================================================
        # VALIDASI TOTAL BOBOT HARUS 1.00
        # =====================================================
        total_weight = weights.sum()
        if round(total_weight, 4) != 1.0:   # toleransi 4 decimal
            st.error(f"❌ Total bobot saat ini = {total_weight:.2f}. Total bobot harus = 1.00 (100%).")
            st.warning("⚠️ Silakan sesuaikan bobot sampai totalnya tepat 1.00.")
            st.stop()
        # =====================================================

        # pastikan numerik
        for col in criteria_list:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0)

        df = df_raw.copy()

        # =====================================================
        # SAW FUNCTION (FINAL FIXED)
        # =====================================================
        def compute_saw(df):
            X = df[criteria_list].copy()
            norm = pd.DataFrame(index=X.index, columns=criteria_list)

            for i, col in enumerate(criteria_list):
                col_data = X[col].astype(float)
                max_val = col_data.max()
                min_val = col_data.min()

                # BENEFIT
                if criteria_type[i] == "Benefit":
                    if max_val == 0:
                        norm[col] = 0
                    else:
                        norm[col] = col_data / max_val

                # COST
                else:
                    col_safe = col_data.replace(0, np.nan)
                    norm[col] = min_val / col_safe
                    norm[col] = norm[col].fillna(0)

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
        # TOMBOL PROSES
        # =====================================================
        if st.button("🔍 Proses SAW → TOPSIS"):

            # SAW Proses
            norm_saw, weighted_saw, saw_scores = compute_saw(df)

            df_scores = df.copy()
            df_scores["SAW_Score"] = saw_scores

            df_saw_rank = df_scores.sort_values("SAW_Score", ascending=False).reset_index(drop=True)
            df_saw_rank.insert(0, "Rank_SAW", range(1, len(df_saw_rank) + 1))


            # TOPSIS Proses
            preferensi, D_plus, D_minus, ideal_pos, ideal_neg = compute_topsis(weighted_saw)

            df_scores["TOPSIS_Score"] = preferensi
            df_topsis_rank = df_scores.sort_values("TOPSIS_Score", ascending=False).reset_index(drop=True)
            df_topsis_rank.insert(0, "Rank_TOPSIS", range(1, len(df_topsis_rank) + 1))

            st.subheader("📌 Hasil TOPSIS")
            st.dataframe(df_topsis_rank)

            # grafik
            st.subheader("📈 Grafik Perbandingan SAW vs TOPSIS")

            if "student_id" in df_scores.columns:
                chart = pd.DataFrame({
                    "Nama": df_scores["student_id"],
                    "SAW": df_scores["SAW_Score"],
                    "TOPSIS": df_scores["TOPSIS_Score"]
                })
                st.bar_chart(chart.set_index("Nama"))
            else:
                st.info("Kolom 'student_id' tidak ditemukan untuk grafik.")
