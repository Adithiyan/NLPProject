import os
import pandas as pd
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer for singular/plural normalization
lemmatizer = WordNetLemmatizer()

# Define paths
gold_standard_path = "Domain Models/Gold standard - Associations.xlsx"
models_root_folder = r"E:\Adi\UO\A Fall\NLP\test\project\outputs"  # Adjust to match your setup

# Load gold-standard data
gold_df = pd.read_excel(gold_standard_path)

# Process gold standard associations
gold_associations = {}
for column_group in range(0, len(gold_df.columns), 2):  # Iterate over pairs of columns (x, y)
    project_name = gold_df.columns[column_group].strip().lower()  # Project name
    x_col = gold_df.columns[column_group]
    y_col = gold_df.columns[column_group + 1]
    
    gold_associations[project_name] = set(
        f"({lemmatizer.lemmatize(row[x_col].strip().lower())}, {lemmatizer.lemmatize(row[y_col].strip().lower())})"
        for _, row in gold_df.iterrows()
        if pd.notnull(row[x_col]) and pd.notnull(row[y_col])
    )

# Function to evaluate associations for all models
def evaluate_associations(models_root_folder, gold_associations):
    results = []

    for model in ["gemini", "gpt4o", "llama", "mistral", "gpt4omini", "gpt4ofewshotcot", "gpt4ozeroshot"]:
        model_folder = os.path.join(models_root_folder, model, "associationgold")
        print(f"\nChecking folder: {model_folder}")  # Debugging path
        
        if not os.path.exists(model_folder):
            print(f"Folder does not exist: {model_folder}")
            continue  # Skip missing folders

        for filename in os.listdir(model_folder):
            if filename.endswith(".txt"):
                project_name = filename.split(".")[0].lower()

                # Load LLM output
                file_path = os.path.join(model_folder, filename)
                print(f"Processing file: {file_path}")  # Debugging file path
                with open(file_path, 'r') as f:
                    predicted_associations = set()
                    for line in f:
                        line = line.strip().lower()
                        if not line:
                            continue
                        # Remove extra parentheses if they exist
                        line = line.strip('()')
                        parts = [lemmatizer.lemmatize(part.strip()) for part in line.split(',')]
                        if len(parts) == 2:
                            predicted_associations.add(f"({parts[0]}, {parts[1]})")
                        else:
                            print(f"Skipping malformed line: {line}")

                # Compare with gold standard
                gold_set = gold_associations.get(project_name, set())
                true_positive = len(predicted_associations & gold_set)
                precision = true_positive / len(predicted_associations) if predicted_associations else 0
                recall = true_positive / len(gold_set) if gold_set else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f0_5 = (1 + 0.5**2) * (precision * recall) / ((0.5**2) * precision + recall) if (precision + recall) > 0 else 0
                f2 = (1 + 2**2) * (precision * recall) / ((2**2) * precision + recall) if (precision + recall) > 0 else 0

                # Debugging individual results
                print(f"\nProject: {project_name}")
                print(f"Gold Standard Associations: {gold_set}")
                print(f"Predicted Associations: {predicted_associations}")
                print(f"True Positives: {true_positive}")
                print(f"F1-Score: {f1:.2f}, F0.5-Score: {f0_5:.2f}, F2-Score: {f2:.2f}")

                results.append({
                    "Model": model,
                    "Project": project_name,
                    "F0.5-Score": f0_5,
                    "F1-Score": f1,
                    "F2-Score": f2
                })

    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    print("\nAssociation Evaluation Results:")
    print(results_df)
    print("\nAverage Metrics by Model for Associations:")
    print(results_df.groupby("Model").mean(numeric_only=True))

    # Debugging final output format
    print("\nFinal Evaluation DataFrame:")
    print(results_df.to_string(index=False))

    results_df.to_csv("association_evaluation_results.csv", index=False)

# Run evaluation
evaluate_associations(models_root_folder, gold_associations)
