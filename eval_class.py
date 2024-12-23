import os
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# Initialize lemmatizer for singular/plural normalization
lemmatizer = WordNetLemmatizer()

# Define paths
gold_standard_path = "Domain Models/Gold standard - Classes.xlsx"
models_root_folder = r"E:\Adi\UO\A Fall\NLP\test\project\outputs"

# Load gold-standard data
gold_df = pd.read_excel(gold_standard_path)
gold_classes = {
    column.lower(): set(
        lemmatizer.lemmatize(word.lower()) for word in gold_df[column].dropna()
    )
    for column in gold_df.columns  # Columns are project names
}

# Function to evaluate classes for all models
def evaluate_classes(models_root_folder, gold_classes):
    results = []

    for model in ["gemini", "gpt4o", "llama", "mistral", "gpt4omini", "gpt4ofewshotcot", "gpt4ozeroshot"]:
        model_folder = os.path.join(models_root_folder, model, "class")
        print(f"\nChecking folder: {model_folder}")

        if not os.path.exists(model_folder):
            print(f"Folder does not exist: {model_folder}")
            continue

        for filename in os.listdir(model_folder):
            if filename.endswith(".txt"):
                project_name = filename.split(".")[0].lower()

                # Load LLM output
                file_path = os.path.join(model_folder, filename)
                print(f"Processing file: {file_path}")
                with open(file_path, 'r') as f:
                    predicted_classes = set(
                        lemmatizer.lemmatize(line.strip().lower())
                        for line in f if line.strip()
                    )

                # Compare with gold standard
                gold_set = gold_classes.get(project_name, set())

                true_positive = len(predicted_classes & gold_set)
                precision = true_positive / len(predicted_classes) if predicted_classes else 0
                recall = true_positive / len(gold_set) if gold_set else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f0_5 = (1 + 0.5**2) * (precision * recall) / (0.5**2 * precision + recall) if (precision + recall) > 0 else 0
                f2 = (1 + 2**2) * (precision * recall) / (2**2 * precision + recall) if (precision + recall) > 0 else 0

                # Print debug info
                print(f"\nProject: {project_name}")
                print(f"Gold Standard: {gold_set}")
                print(f"Predicted Classes: {predicted_classes}")
                print(f"True Positives: {true_positive}")
                print(f"F0.5-Score: {f0_5:.2f}, F1-Score: {f1:.2f}, F2-Score: {f2:.2f}\n")

                results.append({
                    "Model": model,
                    "Project": project_name,
                    "F0.5-Score": f0_5,
                    "F1-Score": f1,
                    "F2-Score": f2
                })

    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    print("\nClass Evaluation Results for Class Identification:")
    print(results_df)
    print("\nAverage Metrics by Model:")
    print(results_df.groupby("Model").mean(numeric_only=True))
    results_df.to_csv("class_evaluation_results.csv", index=False)

# Run evaluation
evaluate_classes(models_root_folder, gold_classes)
