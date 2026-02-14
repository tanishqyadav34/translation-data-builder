import streamlit as st
import pandas as pd
from utils import align_texts_naive, load_nltk_resources
import base64

# Page Config
st.set_page_config(
    page_title="Translation Data Builder",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Aesthetics
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Global Styles */
        .stApp {
            background-color: #0e1117;
            color: #e0e0e0;
            font-family: 'Inter', sans-serif;
        }
        
        /* Headings */
        h1, h2, h3, h4 {
            color: #ffffff;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        h1 {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #2ecc71 0%, #27ae60 100%);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(46, 204, 113, 0.5);
        }
        .stButton>button:active {
            transform: translateY(0);
        }

        /* Text Areas */
        .stTextArea textarea {
            background-color: #1a1c24;
            color: #f1f3f5;
            border: 1px solid #2d3436;
            border-radius: 10px;
            transition: border-color 0.3s ease;
        }
        .stTextArea textarea:focus {
            border-color: #4facfe;
            box-shadow: 0 0 0 2px rgba(79, 172, 254, 0.2);
        }
        
        /* Data Editor */
        div[data-testid="stDataEditor"] {
            border-radius: 12px;
            border: 1px solid #2d3436;
            overflow: hidden;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #13151b;
            border-right: 1px solid #2d3436;
        }
        
        /* Metrics */
        div[data-testid="metric-container"] {
            background-color: #1a1c24;
            border: 1px solid #2d3436;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #1a1c24;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Application Title
st.title("🌍 Translation Data Builder")
st.markdown("### Build aligned parallel datasets using AI-powered sentence segmentation.")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    src_lang = st.selectbox("Source Language", ["english"], help="NLTK segmentation language")
    tgt_lang = st.selectbox("Target Language", ["german", "spanish", "french", "italian"], index=0, help="NLTK segmentation language")
    
    st.info("""
    **How to use:**
    1. Paste raw text for Source and Target languages.
    2. Click 'Align & Generate Pairs'.
    3. Manually correct misaligned pairs in the table.
    4. Download your clean dataset.
    """)
    
    st.divider()
    st.caption("Powered by NLTK & Streamlit")

# Load resources
with st.spinner("Loading NLTK resources..."):
    load_nltk_resources()

# Main Input Area
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Source Text ({src_lang.capitalize()})")
    text_source = st.text_area("Source Text", placeholder=f"Paste {src_lang.capitalize()} text here...", height=300, label_visibility="collapsed")
    uploaded_source = st.file_uploader("📂 Upload Source (.txt)", type=["txt"], key="src_upload")
    if uploaded_source:
        text_source = uploaded_source.read().decode("utf-8")

with col2:
    st.subheader(f"Target Text ({tgt_lang.capitalize()})")
    text_target = st.text_area("Target Text", placeholder=f"Paste {tgt_lang.capitalize()} text here...", height=300, label_visibility="collapsed")
    uploaded_target = st.file_uploader("📂 Upload Target (.txt)", type=["txt"], key="tgt_upload")
    if uploaded_target:
        text_target = uploaded_target.read().decode("utf-8")

# Processing Logic
if st.button("🚀 Align & Generate Pairs", use_container_width=True):
    if not text_source or not text_target:
        st.warning("⚠️ Please provide both source and target text to proceed.")
    else:
        with st.spinner("Processing text... (This may take a moment for large texts)"):
            try:
                # Call helper function
                df = align_texts_naive(text_source, text_target, src_lang, tgt_lang)
                
                if df.empty:
                    st.error("Failed to generate pairs. Make sure models are installed.")
                else:
                    # Success
                    st.success("Alignment Complete!")
                    
                    # Metrics
                    total = len(df)
                    filled = len(df[(df["Source"] != "") & (df["Target"] != "")])
                    avg_ratio = df["Char Ratio"].mean() if not df.empty else 0
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Rows", total)
                    m2.metric("Valid Pairs", filled, delta=f"{filled/total:.0%}")
                    m3.metric("Avg Length Ratio", f"{avg_ratio:.2f}")

                    st.markdown("---")
                    st.subheader("📝 Review & Edit Dataset")
                    
                    # Filtering Options
                    with st.expander("Filter Options", expanded=False):
                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            min_r, max_r = st.slider("Filter by Character Ratio", 0.0, 5.0, (0.1, 4.0), step=0.1, help="Keep pairs within this length ratio range")
                        with col_f2:
                            remove_empty = st.checkbox("Remove Empty Rows", value=True, help="Remove rows where either Source or Target is empty")
                        
                        # Apply Filters
                        if remove_empty:
                             df = df[(df["Source"] != "") & (df["Target"] != "")]
                        
                        df = df[(df["Char Ratio"] >= min_r) & (df["Char Ratio"] <= max_r)]
                        st.caption(f"Showing {len(df)} rows after filtering.")

                    # Editable Dataframe
                    edited_df = st.data_editor(
                        df,
                        num_rows="dynamic",
                        use_container_width=True,
                        height=500,
                        hide_index=True,
                        column_config={
                            "ID": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                            "Source": st.column_config.TextColumn("Source Segment", width="medium"),
                            "Target": st.column_config.TextColumn("Target Segment", width="medium"),
                            "Char Ratio": st.column_config.ProgressColumn(
                                "Ratio",
                                help="Character length ratio (Source/Target)",
                                format="%.2f",
                                min_value=0,
                                max_value=2,
                                width="small"
                            ),
                        }
                    )
                    
                    st.markdown("---")
                    st.subheader("💾 Export Data")
                    
                    # Export options
                    col_dl1, col_dl2 = st.columns(2)
                    
                    # CSV
                    csv = edited_df.to_csv(index=False).encode('utf-8')
                    with col_dl1:
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name='aligned_pairs.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    
                    # JSON
                    json_data = edited_df.to_json(orient="records", indent=2)
                    with col_dl2:
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name='aligned_pairs.json',
                            mime='application/json',
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
