# Translation Data Builder

A powerful tool to build parallel and aligned datasets for machine translation models using **NLTK** for sentence segmentation and **Streamlit** for an intuitive, editable interface.

## Quick Start

1. **Install Dependencies**
   Run the following command to install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   (On Windows, you can also run `run_app.bat`)

2. **Run the Application**
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. **Usage**
   - **Step 1:** Select the source and target languages (e.g., English, German).
   - **Step 2:** Paste raw text or upload `.txt` files for both source and target languages.
   - **Step 3:** Click **Align & Generate Pairs**. The tool will automatically segment sentences and align them 1-to-1.
   - **Step 4:** Review the table. You can directly edit the content or delete incorrect rows.
   - **Step 5:** Export the clean dataset to CSV or JSON format.

## Features

- **Text Alignment:** Automatically aligns sentences using NLTK's robust segmentation.
- **Interactive Editing:** Easily correct misaligned or empty pairs directly in the browser.
- **Filtering:** Filter by character length ratio to quickly identify and remove bad alignments.
- **Export Formats:** Supports CSV and JSON for compatibility with most ML pipelines.
