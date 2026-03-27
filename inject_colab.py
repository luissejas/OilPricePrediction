import json

file_path = "interaction_GoogleColab/OilPricePrediction_Interaction.ipynb"

# Read notebook
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define the new training cell
new_cell = {
  "cell_type": "code",
  "source": [
    "# --- 5. COLAB GPU ACCELERATED TRAINING ---\n",
    "# Ensure you have selected 'T4 GPU' in Runtime -> Change runtime type before running\n",
    "print(\"Initializing Cloud Hardware...\")\n",
    "!uv run python src/models/train.py\n"
  ],
  "metadata": {
    "id": "Training_Cell_GPU"
  },
  "execution_count": None,
  "outputs": []
}

# Insert it immediately after the environment sync (cell 0)
nb["cells"].insert(1, new_cell)

# Write it back without escaping unicode
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
    # add trailing newline to match standard .ipynb formatting
    f.write("\n")

print("Safely injected the Training Cell into the Jupyter Notebook!")
