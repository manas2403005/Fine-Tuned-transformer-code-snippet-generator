import streamlit as st
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import os

# 🎯 Title and Subheader
st.title("Code Generation Using Fine-Tuned CodeT5")
st.subheader("Exploratory project - ECE, IIT BHU")

# ✅ Load model
model_path = "./codet5_ensemblemm_model-20250421T123706Z-001/codet5_ensemblemm_model"
try:
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # Use same tokenizer as model
    model.eval()
except Exception as e:
    st.error(f"❌ Error loading model/tokenizer: {e}")
    st.stop()

# 🚀 Inference function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_code(query, max_length=256, num_beams=5):
    input_text = "generate code: " + query.lower()
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            num_return_sequences=1
        )

    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output.strip()


# 🎯 Example prompts
example_queries = [
    "create a 2x2 numpy matrix and subtract another matrix",
    "sort a list using bubble sort",
    " program to remove a key from a dictionary",
    "program to get the length of an array. ",
    "reverse a string in python"
]

# 📥 User input area
query_option = st.selectbox("Choose an example or write your own query:", [""] + example_queries)
custom_query = st.text_input("Enter your own query:", value=query_option)

# ▶️ Generate button
if st.button("Generate Code"):
    if custom_query.strip():
        with st.spinner("Generating code..."):
            generated_code = generate_code(custom_query)

        # 🧹 Clean and format output
        formatted_code = generated_code.strip().replace('\r\n', '\n').replace('\r', '\n')

        # ✅ Display the formatted code
        st.subheader("💡 Generated Code:")
        st.code(formatted_code, language="python")
    else:
        st.warning("Please enter a query to generate code.")
