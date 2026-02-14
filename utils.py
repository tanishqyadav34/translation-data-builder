import pandas as pd
import streamlit as st
import nltk

@st.cache_resource
def load_nltk_resources():
    """Download necessary NLTK data."""
    resources = ["punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)
    return True

@st.cache_data(show_spinner=False)
def align_texts_naive(source_text, target_text, src_lang="english", tgt_lang="german"):
    # Ensure resources are loaded
    load_nltk_resources()
    
    # helper for safe tokenization
    def safe_sent_tokenize(text, language):
        try:
            return nltk.sent_tokenize(text, language=language)
        except LookupError:
            # excessive fallback: try main language, then default
            try: 
                nltk.download('punkt')
                nltk.download('punkt_tab')
                return nltk.sent_tokenize(text, language=language) 
            except:
                return nltk.sent_tokenize(text)

    # Segment Source
    segs_src = safe_sent_tokenize(source_text, src_lang)

    # Segment Target
    segs_tgt = safe_sent_tokenize(target_text, tgt_lang)

    max_len = max(len(segs_src), len(segs_tgt))
    
    # Pad to match maximum length
    segs_src += [""] * (max_len - len(segs_src))
    segs_tgt += [""] * (max_len - len(segs_tgt))
    
    data = []
    
    for i in range(max_len):
        s_text = segs_src[i]
        t_text = segs_tgt[i]
        
        ratio = 0.0
        if t_text and s_text:
           ratio = len(s_text) / len(t_text) if len(t_text) > 0 else 0
           
        data.append({
             "ID": i + 1,
             "Source": s_text,
             "Target": t_text,
             "Char Ratio": round(ratio, 2)
        })
        
    return pd.DataFrame(data)
