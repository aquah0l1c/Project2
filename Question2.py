import streamlit as st
import fitz  # PyMuPDF
import docx
import re
from langchain_ollama import ChatOllama

# Initialize LLM (open-source)
llm = ChatOllama(
    model="llama3.2:3B",
    temperature=0
)

st.title("Abbreviation Index Generator")

# ---------- File text extraction ----------

def extract_pdf_text(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_docx_text(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([p.text for p in doc.paragraphs])

# ---------- Regex-based abbreviation finder ----------

def extract_abbreviation_candidates(text: str) -> dict:
    """
    Find patterns like 'full term (ABBR)' in the text.
    Return a dict {ABBR: full_term} with one entry per abbreviation.
    """
    # Reasonable heuristic: "some words" followed by ALL-CAPS abbreviation
    pattern = r"\b([A-Za-z][A-Za-z\s\-]+?)\s*\(\s*([A-Z]{2,10})\s*\)"

    matches = re.findall(pattern, text)
    candidates = {}

    for full_term, abbr in matches:
        abbr = abbr.strip()
        full_term = " ".join(full_term.split())  # normalize spaces

        # keep the first mapping we see for each abbreviation
        if abbr not in candidates:
            candidates[abbr] = full_term

    return candidates

# ---------- Clean LLM output & enforce no hallucinations ----------

def parse_llm_index(raw_output: str, allowed_abbrs: set) -> dict:
    """
    Parse lines like 'ABBR: full term' from the LLM output.
    Only keep entries whose ABBR is in allowed_abbrs (no hallucinated ones).
    """
    final_index = {}

    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    for line in lines:
        if ":" not in line:
            continue
        abbr, term = line.split(":", 1)
        abbr = abbr.strip()
        term = term.strip()
        if not abbr or not term:
            continue
        if abbr in allowed_abbrs and abbr not in final_index:
            final_index[abbr] = term

    return final_index

# Optional extra instructions from user
question = st.text_input(
    "Optional: extra instructions (e.g., 'only include abbreviations used more than once')",
    value="Generate an abbreviation index."
)

# Allow multiple files for Article1.pdf, Article2.pdf, Article3.pdf
files = st.file_uploader(
    "Upload article(s) (e.g., Article1.pdf, Article2.pdf, Article3.pdf)",
    type=["txt", "pdf", "docx", "html"],
    accept_multiple_files=True
)

if st.button("Submit"):

    if not files:
        st.warning("Please upload at least one article.")
    else:
        for uploaded_file in files:
            st.markdown("---")
            st.subheader(f"Abbreviation Index for **{uploaded_file.name}**")

            # ----- Extract raw text -----
            if uploaded_file.type == "application/pdf":
                file_text = extract_pdf_text(uploaded_file)
            elif uploaded_file.type in ["text/plain", "text/html"]:
                uploaded_file.seek(0)
                file_text = uploaded_file.read().decode(errors="ignore")
            elif uploaded_file.type == (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                file_text = extract_docx_text(uploaded_file)
            else:
                st.write("Unsupported file type:", uploaded_file.type)
                continue

            if not file_text or file_text.strip() == "":
                st.write("No readable text found in this file.")
                continue

            # ----- Step 1: regex to get REAL abbreviations -----
            candidates = extract_abbreviation_candidates(file_text)

            if not candidates:
                st.write("No abbreviations in 'full term (ABBR)' format were found.")
                continue

            allowed_abbrs = set(candidates.keys())

            # Build a simple list for the LLM to clean up
            # e.g. "WDC: weighted degree centrality"
            candidate_block = "\n".join(
                f"{abbr}: {term}" for abbr, term in candidates.items()
            )

            # ----- Step 2: LLM cleans & finalizes the index -----
            prompt = f"""
You are a helpful assistant for processing scientific articles.

You are given a list of abbreviation candidates extracted from an article.
Each line is in the format:

ABBR: full term as extracted

Your job:
- Clean and finalize this abbreviation index.
- For each ABBR, produce a clear and concise full term.
- Do NOT invent any new abbreviations that are not in the list.
- If two candidates use the same ABBR, choose the best full term.
- Keep one line per abbreviation.

Output rules (IMPORTANT):
- Format MUST be exactly:
  ABBR: full term
  ABBR2: full term 2
- One abbreviation per line.
- Sort abbreviations alphabetically by ABBR.
- No explanation before or after the list.

Here are the abbreviation candidates:

{candidate_block}

Additional user instructions (optional from the user):
{question.strip()}
"""

            ai_msg = llm.invoke(prompt)
            raw_output = ai_msg.content

            # ----- Step 3: filter LLM output to prevent hallucinations -----
            final_index = parse_llm_index(raw_output, allowed_abbrs)

            if not final_index:
                st.write(
                    "The LLM did not return any valid abbreviations from the candidate list."
                )
            else:
                # Sort and print in the required format
                lines = [
                    f"{abbr}: {term}"
                    for abbr, term in sorted(final_index.items(), key=lambda kv: kv[0])
                ]
                st.text("\n".join(lines))
