import streamlit as st
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Generic Item Detection & Mapping",
    page_icon="💊",
    layout="wide"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'final_results' not in st.session_state:
    st.session_state.final_results = None
if 'mapping_done' not in st.session_state:
    st.session_state.mapping_done = False
if 'mapped_results' not in st.session_state:
    st.session_state.mapped_results = None

# Fungsi normalize_name
def normalize_name(name):
    normalization_dict = {
        'TAB': 'TABLET', 'AMP': 'AMPUL', 'SYR': 'SIRUP', 'INJ': 'INJEKSI', 'SUSP': 'SUSPENSI',
        'CR': 'CREAM', 'HJ': 'HEXPHARM', 'HEXP': 'HEXPHARM', 'KF': 'KIMIA FARMA', 'NOVA': 'NOVAPHARIN',
        'FM': 'FIRST MEDIFARMA', 'DX': 'OGB DEXA', 'DXM': 'OGB DEXA', 'NVL': 'NOVEL', 'IFI': 'IMFARMIND',
        'SAMP': 'SAMPHARINDO', 'HEXAPHARM': 'HEXPHARM', 'HEXPARM': 'HEXPHARM', 'KIMIAFARMA': 'KIMIA FARMA',
        'ERL': 'ERELA', 'MRS': 'MERSI', 'TRM': 'TRIMAN', 'BTL': 'BOTOL', 'ETA': 'ERRITA', 'BERNO': 'BERNOFARM',
        'MBF': 'MAHAKAM', 'MEF': 'MEGA ESA FARMA', 'FLS': 'BOTOL', 'ERITA': 'ERRITA'
    }
    
    if pd.isna(name):
        return name
    
    for short, full in normalization_dict.items():
        name = re.sub(rf'\b{short}\b', full, name, flags=re.IGNORECASE)
    
    return name

# Fungsi get_item_before_dosage
def get_item_before_dosage(item):
    return re.split(r' (TABLET|KAPLET|SIRUP|KAPSUL)', item, flags=re.IGNORECASE)[0]

# Fungsi clean_text
def clean_text(text):
    if pd.isna(text):
        return text
    
    text = text.replace('-', '').replace(',', '.')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Fungsi check_principal_contains
def check_principal_contains(item, principal_list):
    item_lower = str(item).lower()
    for principal in principal_list:
        if str(principal).lower() in item_lower:
            return True
    return False

# Header
st.title("💊 Generic Item Detection & Mapping System")
st.markdown("---")

# Sidebar untuk upload file
with st.sidebar:
    st.header("📂 Upload File")
    
    dataset_file = st.file_uploader(
        "Upload Dataset (Excel)",
        type=['xlsx', 'xls'],
        help="Upload file dataset yang berisi kolom 'item_id' dan 'full_name'"
    )
    
    st.markdown("---")
    # process_button = st.button("🚀 Process Data", type="primary", width="stretch")
    process_button = st.button("🚀 Process Data", type="primary", width="stretch")
    
    # Reset button
    if st.session_state.processed:
        if st.button("🔄 Reset", width="stretch"):
            st.session_state.processed = False
            st.session_state.final_results = None
            st.session_state.mapping_done = False
            st.session_state.mapped_results = None
            st.rerun()

# Main content
if dataset_file:
    
    if process_button and not st.session_state.processed:
        with st.spinner("Processing data..."):
            try:
                # Load data
                dataset = pd.read_excel(dataset_file)
                
                # Load master files dari sistem (hardcoded path)
                try:
                    master_generic = pd.read_excel('Product_Generic_for_Generic_Identified_Update.xlsx')
                    master_principal = pd.read_excel('Principal for Mapping.xlsx')
                    st.success("✅ Master files loaded from system")
                except FileNotFoundError as e:
                    st.error(f"❌ Master file tidak ditemukan: {str(e)}")
                    st.info("📌 Pastikan file berikut ada di direktori yang sama dengan aplikasi:")
                    st.code("""
- Product_Generic_for_Generic_Identified_Update.xlsx
- Principal for Mapping.xlsx
                    """)
                    st.stop()
                
                # Step 1: Normalisasi nama
                st.subheader("📊 Step 1: Normalisasi Nama")
                normalized_df = dataset.copy()
                normalized_df['full_name_normalize'] = normalized_df['full_name'].apply(normalize_name)
                st.success(f"✅ Normalisasi selesai: {len(normalized_df)} rows")
                with st.expander("Preview Normalized Data"):
                    st.dataframe(normalized_df.sample(min(5, len(normalized_df))), width="stretch")
                
                # Step 2: Proses Master Generic
                st.subheader("🔍 Step 2: Ekstrak Item")
                master_generic['item_before_dosage'] = master_generic['results'].apply(get_item_before_dosage)
                master_generic = master_generic[['item_before_dosage']]
                generic_list = master_generic['item_before_dosage'].str.upper().tolist()
                st.success(f"✅ Generic list created: {len(generic_list)} items")
                with st.expander("Preview Generic List"):
                    st.dataframe(master_generic.head(10), width="stretch")
                
                # Step 3: Filter Generic
                st.subheader("🎯 Step 3: Filter Generic Items")
                generic_df = normalized_df.copy()
                generic_df = generic_df[generic_df['full_name_normalize'].str.upper().str.contains('|'.join(generic_list), case=False, na=False)]
                generic_df = generic_df[['item_id', 'full_name_normalize']].reset_index(drop=True)
                generic_df['generic_flag'] = 'Generic'
                st.success(f"✅ Generic items found: {len(generic_df)} items")
                
                # Step 4: Merge dengan flag
                st.subheader("🔄 Step 4: Merge dengan Generic Flag")
                data_with_flag = normalized_df.copy()
                data_with_flag = data_with_flag.merge(generic_df[['item_id','generic_flag']], on='item_id', how='left')
                
                # Step 5: Clean text
                st.subheader("🧹 Step 5: Clean Text")
                data_with_flag['full_name_cleaned'] = data_with_flag['full_name_normalize'].apply(clean_text)
                
                # Step 6: Remove duplicates
                st.subheader("🗑️ Step 6: Remove Duplicates")
                data_oke = data_with_flag.drop_duplicates(subset='item_id', keep='first').reset_index(drop=True)
                st.success(f"✅ After removing duplicates: {len(data_oke)} rows")
                
                # Step 7: Check Principal
                st.subheader("🏢 Step 7: Check Principal")
                final_results = data_oke.copy()
                final_results = final_results.drop(columns='full_name_normalize')
                
                final_results['has_principal'] = final_results['full_name_cleaned'].apply(
                    lambda x: check_principal_contains(x, master_principal['principal'].tolist())
                )
                
                # Simpan ke session state
                st.session_state.final_results = final_results
                st.session_state.processed = True
                
                st.success("✅ Processing completed!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Display results (baik setelah processing atau dari session state)
    if st.session_state.processed and st.session_state.final_results is not None:
        final_results = st.session_state.final_results
        
        # Display results
        st.markdown("---")
        st.header("📋 Generic Detection Results")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Items", len(final_results))
        with col2:
            generic_count = final_results['generic_flag'].notna().sum()
            st.metric("Generic Items", generic_count)
        with col3:
            count_generic_no_principal = final_results[
                (final_results['generic_flag'] == 'Generic') &
                (final_results['has_principal'] == False)
            ].shape[0]
            st.metric("Generic w/o Principal", count_generic_no_principal)

        # Display data
        # sample_100 = final_results.sample(n=100, random_state=42)
        # st.dataframe(sample_100, width="stretch", height=400)
        st.dataframe(final_results, width="stretch", height=400)
        
        # Download buttons untuk Generic Detection
        st.subheader("📥 Download Generic Detection Results")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # Download Excel
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                final_results.to_excel(writer, index=False, sheet_name='Results')
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Results (Excel)",
                data=buffer,
                file_name="generic_detection_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )
        
        with col_dl2:
            # Download CSV
            csv = final_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="generic_detection_results.csv",
                mime="text/csv",
                width="stretch"
            )
        
        # ========== MAPPING SECTION ==========
        st.markdown("---")
        st.header("🔗 Step 8: Mapping to Master Data")
        st.info("💡 Gunakan BioBERT untuk mapping item ke Master Data dengan cosine similarity")
        
        mapping_button = st.button("🚀 Start Mapping", type="primary", width="content")

        if mapping_button and not st.session_state.mapping_done:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load Master Data (10%)
                status_text.text("🔄 Loading Master Data...")
                progress_bar.progress(5)
                acuan = pd.read_excel('Master Data P1-P6 with Added.xlsx')
                acuan = acuan.drop_duplicates(keep='first')
                st.success(f"✅ Master Data loaded: {len(acuan)} items")
                
                # Step 2: Prepare target data (20%)
                status_text.text("📋 Preparing target data...")
                progress_bar.progress(10)
                data = final_results.copy()
                target = data[['full_name_cleaned']]
                target = target.rename(columns={'full_name_cleaned': 'id_name_assist'})
                target["id_name_assist"] = target["id_name_assist"].fillna("").astype(str)
                
                # Step 3: Load BioBERT model (30%)
                status_text.text("📦 Loading BioBERT model...")
                progress_bar.progress(15)
                model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
                st.success("✅ BioBERT model loaded")
                
                # Step 4: Encode master data (50%)
                embeddings_file = 'embeddings_master_data.npz'
                
                if os.path.exists(embeddings_file):
                    status_text.text("📂 Loading saved master embeddings...")
                    progress_bar.progress(20)
                    
                    # cara lama
                    # emb_acuan = np.load(embeddings_file)

                    # cara baru
                    emb_acuan = np.load(embeddings_file)
                    emb_acuan = emb_acuan['emb_acuan']

                    progress_bar.progress(30)
                    st.success(f"✅ Master embeddings loaded from file: {emb_acuan.shape}")
                    st.info("💡 Menggunakan embeddings yang sudah tersimpan (lebih cepat!)")
                else:
                    status_text.text("🧬 Encoding master data... (First time)")
                    progress_bar.progress(40)
                    emb_acuan = model.encode(
                        acuan["Item"].tolist(), 
                        convert_to_tensor=False, 
                        normalize_embeddings=True, 
                        show_progress_bar=False
                    )
                    # Save embeddings for future use
                    np.save(embeddings_file, emb_acuan)
                    progress_bar.progress(50)
                    st.success(f"✅ Master data encoded and saved: {emb_acuan.shape}")
                    st.info(f"💾 Embeddings saved to '{embeddings_file}'!")
                
                # Step 5: Encode target data (70%) - INI HARUS DI LUAR if/else
                status_text.text("🧬 Encoding target data...")
                progress_bar.progress(60)
                emb_target = model.encode(
                    target["id_name_assist"].tolist(), 
                    convert_to_tensor=False, 
                    normalize_embeddings=True, 
                    show_progress_bar=False
                )
                
                # konversi ke float16 agar sama dengan yang di acuan
                emb_target = emb_target.astype(np.float16)
                
                progress_bar.progress(70)
                st.success(f"✅ Target data encoded: {emb_target.shape}")
                
                # Step 6: Compute similarity matrix (85%)
                status_text.text("🔍 Computing similarity matrix...")
                progress_bar.progress(75)
                similarity_matrix = cosine_similarity(emb_target, emb_acuan)
                progress_bar.progress(85)
                st.success("✅ Similarity matrix computed")
                
                # Step 7: Find top 3 matches (90%)
                status_text.text("🎯 Finding top 3 matches...")
                progress_bar.progress(90)
                top_3_indices = np.argsort(-similarity_matrix, axis=1)[:, :3]
                top_3_scores = np.sort(-similarity_matrix, axis=1)[:, :3] * -1
                
                # Step 8: Prepare results (95%)
                status_text.text("📊 Preparing results...")
                progress_bar.progress(95)
                results = []
                for i in range(len(target)):
                    results.append({
                        "id_name_assist": target.iloc[i]["id_name_assist"],
                        "Nama_Terbaik_1": acuan.iloc[top_3_indices[i][0]]["Item"],
                        "Skor_Similarity_1": float(top_3_scores[i][0]),
                        "Nama_Terbaik_2": acuan.iloc[top_3_indices[i][1]]["Item"],
                        "Skor_Similarity_2": float(top_3_scores[i][1]),
                        "Nama_Terbaik_3": acuan.iloc[top_3_indices[i][2]]["Item"],
                        "Skor_Similarity_3": float(top_3_scores[i][2])
                    })
                
                df_hasil = pd.DataFrame(results)
                
                # Step 9: Finalize (100%)
                status_text.text("✨ Finalizing...")
                progress_bar.progress(98)
                simpan = pd.concat([data.reset_index(drop=True), df_hasil.reset_index(drop=True)], axis=1)
                
                # Save to session state
                st.session_state.mapped_results = simpan
                st.session_state.mapping_done = True
                
                progress_bar.progress(100)
                status_text.text("✅ Mapping completed!")
                st.success("✅ Mapping completed successfully!")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Mapping Error: {str(e)}")
                st.exception(e)
        
        # Display mapping results
        if st.session_state.mapping_done and st.session_state.mapped_results is not None:
            mapped_results = st.session_state.mapped_results
            
            st.markdown("---")
            st.header("📊 Mapping Results")
            
            # Metrics untuk mapping
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total Mapped", len(mapped_results))
            with col_m2:
                high_similarity = (mapped_results['Skor_Similarity_1'] >= 0.9).sum()
                st.metric("High Similarity (≥0.9)", high_similarity)
            with col_m3:
                avg_similarity = mapped_results['Skor_Similarity_1'].mean()
                st.metric("Avg Similarity", f"{avg_similarity:.3f}")
            
            # Filter by similarity
            st.subheader("🔎 Filter by Similarity Score")
            min_similarity = st.slider(
                "Minimum Similarity Score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05
            )
            
            filtered_mapped = mapped_results[mapped_results['Skor_Similarity_1'] >= min_similarity]
            st.info(f"Showing {len(filtered_mapped)} items with similarity ≥ {min_similarity}")
            
            # Display mapped data
            st.dataframe(filtered_mapped, width="stretch", height=400)
            # sample_100 = filtered_mapped.sample(n=100, random_state=42)
            # st.dataframe(sample_100, width="stretch", height=400)
            
            # Download buttons untuk Mapping Results
            st.subheader("📥 Download Mapping Results")
            col_dl3, col_dl4 = st.columns(2)
            
            with col_dl3:
                # Download Excel
                buffer_map = BytesIO()
                with pd.ExcelWriter(buffer_map, engine='openpyxl') as writer:
                    filtered_mapped.to_excel(writer, index=False, sheet_name='Mapping Results')
                buffer_map.seek(0)
                
                st.download_button(
                    label="📥 Download Mapping (Excel)",
                    data=buffer_map,
                    file_name="mapping_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch"
                )
            
            with col_dl4:
                # Download CSV
                csv_map = filtered_mapped.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Mapping (CSV)",
                    data=csv_map,
                    file_name="mapping_results.csv",
                    mime="text/csv",
                    width="stretch"
                )

else:
    st.info("👈 Silakan upload file Dataset di sidebar untuk memulai.")
    
    # Informasi tambahan
    with st.expander("ℹ️ Informasi Aplikasi"):
        st.markdown("""
        ### Cara Menggunakan:
        1. Upload **Dataset** (harus memiliki kolom `item_id` dan `full_name`)
        2. Klik tombol **Process Data**
        3. Setelah selesai, klik **Start Mapping** untuk mapping ke Master Data
        
        ### Master Files (Sudah di Sistem):
        - `Product_Generic_for_Generic_Identified_Update.xlsx` (kolom: `results`)
        - `Principal for Mapping.xlsx` (kolom: `principal`)
        - `Master Data P1-P6 with Added.xlsx` (kolom: `Item`)
        
        ### Proses yang Dilakukan:
        **Step 1-7: Generic Detection**
        - Normalisasi nama item
        - Ekstraksi nama generic
        - Identifikasi item generic
        - Cleaning text
        - Pengecekan principal
        - Generate flag `generic_flag` dan `has_principal`
        
        **Step 8: Mapping**
        - BioBERT encoding
        - Cosine similarity computation (sklearn)
        - Top 3 matches dengan skor similarity
        
        ### Output:
        - Tabel hasil Generic Detection dengan download
        - Tabel hasil Mapping dengan filter similarity dan download
        """)