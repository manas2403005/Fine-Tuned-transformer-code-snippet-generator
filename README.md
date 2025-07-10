# Code Snippet Generator using Fine-tuned CodeT5

This project implements a code generation system that translates natural language queries into Python code snippets using a fine-tuned CodeT5 model. It incorporates ensemble learning, k-fold cross-validation, and a Streamlit web interface for practical use.

## Overview

We fine-tuned the CodeT5-base transformer model on a curated dataset of problem-solution pairs. The model takes a natural language query as input and outputs relevant, syntactically correct Python code.

**Key Features:**
- Fine-tuning of CodeT5 on 3,300 Python problem-solution pairs
- K-Fold Cross-Validation (k=3) for robust evaluation
- Ensemble learning through majority voting
- Temperature-based sampling for diverse code generation
- Deployed using Streamlit for real-time interaction

## Dataset

The dataset consists of:
- Natural language queries describing programming tasks
- Corresponding Python code snippets

It covers areas such as:
- NumPy and Pandas operations
- String processing
- Data structure manipulation
- Common algorithms

## Training Details

- Model: CodeT5-base (220M parameters)
- Token limit: 256 tokens
- Learning rate: 5e-5
- Batch size: 4
- Epochs: 5
- Weight decay: 0.01

We used Hugging Face's Trainer API for training and implemented temperature-based sampling with top-k (50) and top-p (0.9) filtering for inference.

## Evaluation

| Metric       | Score  |
|--------------|--------|
| ROUGE-L      | 39.14  |
| BLEU         | 25.66  |
| Exact Match  | 0.09   |

These scores reflect the model’s ability to capture semantic similarity and functional correctness, despite variations in implementation.

## Sample Output

**Query:** Find the smallest element in the array  
**Generated Code:**
```python
arr = []
size = int(input("Enter_the_size_of_the_array:"))
print("Enter_the_elements_of_the_array:")
for i in range(0, size):
    num = int(input())
    arr.append(num)

print("Smallest_element_in_array_is:", min(arr)) 

Streamlit App
The web interface includes:

Input field for custom queries

Dropdown of example problems

Syntax-highlighted code output

Error handling for generation failures

Limitations
Low exact match score due to multiple valid solutions

Code quality may vary for complex or ambiguous queries

Performance degrades on domain-specific tasks underrepresented in training data

Future Work
Expand dataset with more diverse tasks

Add runtime validation for generated code

Support for multiple programming languages

Apply parameter-efficient fine-tuning techniques (e.g., LoRA)

Incorporate human feedback to refine model behavior

Citation
This work was developed as part of an exploratory project (Jan–Apr 2025) at the Department of Electronics Engineering, IIT (BHU), Varanasi.
