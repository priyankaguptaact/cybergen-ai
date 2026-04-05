import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

st.set_page_config(
    page_title="CyberGen AI",
    page_icon="🔐",
    layout="wide"
)

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

def generate(prompt, max_new_tokens=200, temperature=0.7):
    encoding = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=128
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True, top_k=50, top_p=0.95,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    for marker in ["WARNING:", "------------", "Traceback", "CPU:", "~~~~~"]:
        if marker in generated:
            generated = generated[:generated.index(marker)].strip()
    return generated

# ── UI ──
st.title("🔐 CyberGen AI")
st.caption("Domain Specific Text Generation Using Fine-Tuned GPT-2 Medium")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "GPT-2 Medium")
col2.metric("Parameters", "345M")
col3.metric("ROUGE-1", "0.5632")
col4.metric("BLEU-1", "0.3886")

st.divider()

tab1, tab2 = st.tabs(["🔍 CVE Lookup", "✍️ Custom Prompt"])

# ── CVE Tab ──
with tab1:
    st.subheader("CVE Description Generator")
    col_in, col_out = st.columns(2)

    with col_in:
        # Dropdown instead of buttons
        cve_example = st.selectbox(
            "Select Quick Example (or type your own below)",
            ["-- Select --", "CVE-2024-53113", "CVE-2023-44487",
             "CVE-2024-38221", "CVE-2022-30190"]
        )

        # Auto fill input from dropdown
        default_cve = "" if cve_example == "-- Select --" else cve_example
        cve_id = st.text_input("Enter CVE ID", value=default_cve,
                                placeholder="e.g. CVE-2024-1234")

        generate_cve = st.button("Generate Description",
                                  type="primary", use_container_width=True)

    with col_out:
        st.write("**Generated Output:**")
        if generate_cve:
            if not cve_id:
                st.error("Please enter a CVE ID")
            else:
                if not cve_id.upper().startswith("CVE-"):
                    cve_id = f"CVE-{cve_id}"
                with st.spinner("Analyzing vulnerability..."):
                    result = generate(f"Vulnerability: {cve_id.upper()} Description:")
                st.success("Done!")
                st.text_area("Result", result, height=250)
        else:
            st.info("Enter a CVE ID and click Generate")

# ── Custom Tab ──
with tab2:
    st.subheader("Custom Prompt Generator")
    col_in2, col_out2 = st.columns(2)

    with col_in2:
        # Dropdown instead of buttons
        quick_prompts = {
            "-- Select --": "",
            "Buffer Overflow": "A buffer overflow vulnerability in",
            "XSS Attack": "Cross-site scripting vulnerability in",
            "RCE": "Remote code execution vulnerability in",
            "SQL Injection": "SQL injection vulnerability in",
            "Privilege Escalation": "Privilege escalation vulnerability in"
        }

        selected = st.selectbox("Select Quick Prompt (or type your own below)",
                                 list(quick_prompts.keys()))

        default_prompt = quick_prompts[selected]
        prompt_input = st.text_area("Enter Prompt", value=default_prompt,
                                     placeholder="e.g. A SQL injection vulnerability in...",
                                     height=120)

        temperature = st.slider("Temperature (Creativity)",
                                 min_value=0.1, max_value=1.5,
                                 value=0.7, step=0.1)

        max_tokens = st.slider("Max Output Tokens",
                                min_value=50, max_value=400,
                                value=200, step=50)

        generate_custom = st.button("Generate Text",
                                     type="primary", use_container_width=True)

    with col_out2:
        st.write("**Generated Output:**")
        if generate_custom:
            if not prompt_input:
                st.error("Please enter a prompt")
            else:
                with st.spinner("Generating text..."):
                    result = generate(prompt_input,
                                      max_new_tokens=max_tokens,
                                      temperature=temperature)
                st.success("Done!")
                st.text_area("Result", result, height=250)
        else:
            st.info("Enter a prompt and click Generate")

st.divider()
st.caption("Research Project 2025 — Domain Specific Text Generation Using Fine-Tuned Transformer Models")
