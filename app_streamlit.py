import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# ── Page Config ──
st.set_page_config(
    page_title="CyberGen AI",
    page_icon="🔐",
    layout="wide"
)

# ── Session State ──
if "cve_input" not in st.session_state:
    st.session_state.cve_input = ""
if "custom_input" not in st.session_state:
    st.session_state.custom_input = ""

# ── Load Model ──
@st.cache_resource
def load_model():
    MODEL_PATH = "priyankaguptaact/cybergen-gpt2-medium"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ── Generate Function ──
def generate(prompt, max_new_tokens=200, temperature=0.7):
    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean hallucinations
    for marker in ["WARNING:", "------------", "Traceback", "CPU:", "PID:", "~~~~~", "====", "____"]:
        if marker in generated:
            generated = generated[:generated.index(marker)].strip()

    return generated

# ── UI ──
st.title("🔐 CyberGen AI")
st.caption("Domain Specific Text Generation Using Fine-Tuned GPT-2 Medium")

# Stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "GPT-2 Medium")
col2.metric("Parameters", "345M")
col3.metric("ROUGE-1", "0.5632")
col4.metric("BLEU-1", "0.3886")

st.divider()

# Tabs
tab1, tab2 = st.tabs(["🔍 CVE Lookup", "✍️ Custom Prompt"])

# ── CVE Tab ──
with tab1:
    st.subheader("CVE Description Generator")
    col_in, col_out = st.columns(2)

    with col_in:
        # Quick Example Buttons
        st.write("**Quick Examples:**")
        ex1, ex2, ex3, ex4 = st.columns(4)
        if ex1.button("CVE-2024-53113", key="ex1"):
            st.session_state.cve_input = "CVE-2024-53113"
        if ex2.button("CVE-2023-44487", key="ex2"):
            st.session_state.cve_input = "CVE-2023-44487"
        if ex3.button("CVE-2024-38221", key="ex3"):
            st.session_state.cve_input = "CVE-2024-38221"
        if ex4.button("CVE-2022-30190", key="ex4"):
            st.session_state.cve_input = "CVE-2022-30190"

        # Input
        cve_id = st.text_input(
            "Enter CVE ID",
            value=st.session_state.cve_input,
            placeholder="e.g. CVE-2024-1234",
            key="cve_field"
        )

        generate_cve = st.button("Generate Description", type="primary", key="cve_btn", use_container_width=True)

    with col_out:
        st.write("**Generated Output:**")
        if generate_cve:
            if not cve_id:
                st.error("Please enter a CVE ID")
            else:
                if not cve_id.upper().startswith("CVE-"):
                    cve_id = f"CVE-{cve_id}"
                with st.spinner("Analyzing vulnerability..."):
                    prompt = f"Vulnerability: {cve_id.upper()} Description:"
                    result = generate(prompt)
                st.success("Generated successfully!")
                st.text_area("Result", result, height=250, key="cve_result")
        else:
            st.info("Enter a CVE ID and click Generate")

# ── Custom Tab ──
with tab2:
    st.subheader("Custom Prompt Generator")
    col_in2, col_out2 = st.columns(2)

    with col_in2:
        # Quick Prompt Buttons
        st.write("**Quick Prompts:**")
        qp1, qp2 = st.columns(2)
        if qp1.button("Buffer Overflow", key="qp1", use_container_width=True):
            st.session_state.custom_input = "A buffer overflow vulnerability in"
        if qp2.button("XSS Attack", key="qp2", use_container_width=True):
            st.session_state.custom_input = "Cross-site scripting vulnerability in"
        qp3, qp4 = st.columns(2)
        if qp3.button("RCE", key="qp3", use_container_width=True):
            st.session_state.custom_input = "Remote code execution vulnerability in"
        if qp4.button("SQL Injection", key="qp4", use_container_width=True):
            st.session_state.custom_input = "SQL injection vulnerability in"

        # Input
        prompt_input = st.text_area(
            "Enter Prompt",
            value=st.session_state.custom_input,
            placeholder="e.g. A SQL injection vulnerability in...",
            height=120,
            key="custom_field"
        )

        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.1,
            max_value=1.5,
            value=0.7,
            step=0.1
        )

        max_tokens = st.slider(
            "Max Output Tokens",
            min_value=50,
            max_value=400,
            value=200,
            step=50
        )

        generate_custom = st.button("Generate Text", type="primary", key="custom_btn", use_container_width=True)

    with col_out2:
        st.write("**Generated Output:**")
        if generate_custom:
            if not prompt_input:
                st.error("Please enter a prompt")
            else:
                with st.spinner("Generating text..."):
                    result = generate(prompt_input, max_new_tokens=max_tokens, temperature=temperature)
                st.success("Generated successfully!")
                st.text_area("Result", result, height=250, key="custom_result")
        else:
            st.info("Enter a prompt and click Generate")

st.divider()
st.caption("Research Project 2025 — Domain Specific Text Generation Using Fine-Tuned Transformer Models")
