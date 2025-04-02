import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from huggingface_hub import login  # Import login function

# Authenticate with Hugging Face
login(token=st.secrets["HF_TOKEN"])  # Load token from Streamlit Secrets

# Load the fine-tuned model
model = T5ForConditionalGeneration.from_pretrained("manas2403005/t5-fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("t5-small")  # Adjust if needed

# Define a function for inference
def generate_code(query):
    input_text = "generate code: " + query.lower()
    input_ids = tokenizer.encode(input_text, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app UI
st.title('Code Generation Using Fine-Tuned T5 Model')
query = st.text_input("Enter your query to generate code:")

if st.button("Generate Code"):
    if query:
        code = generate_code(query)
        st.subheader("Generated Code:")
        st.code(code, language="python")
    else:
        st.warning("Please enter a query to generate code.")

# Push the model to Hugging Face Hub **only if logged in**
if st.button("Push Model to Hub"):
    with st.spinner("Pushing model to Hugging Face Hub..."):
        model.push_to_hub("manas2403005/t5-fine-tuned", exist_ok=True)
        tokenizer.push_to_hub("manas2403005/t5-fine-tuned")
    st.success("Model successfully pushed to Hugging Face Hub!")
