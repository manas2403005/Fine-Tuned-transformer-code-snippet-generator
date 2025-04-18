import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from huggingface_hub import notebook_login

# Load the fine-tuned model
model_path = "t5_finetuned150man_model"  # Replace this with the path or Hugging Face model identifier if needed

# Ensure safe loading of the model with 'safe_serialization=True'
try:
    model = T5ForConditionalGeneration.from_pretrained(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")  # Change to your model's variant


# Define a function for inference
def generate_code(query):
    query = query.lower()
    input_text = "generate code: " + query
    input_ids = tokenizer.encode(input_text,  max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app
st.title('Code Generation Using Fine-Tuned T5 Model')
st.subheader('Exploratory project , ECE , IIT BHU')
# Input field for user query
query = st.text_input("Enter your query to generate code:")

# Button to trigger code generation
if st.button("Generate Code"):
    if query:
        code = generate_code(query)
        st.subheader("Generated Code:")
        st.code(code, language="python")
    else:
        st.warning("Please enter a query to generate code.")

