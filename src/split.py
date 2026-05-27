import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your labels file
LABELS_FILE = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\humanized_linear_dataset\labels.csv"

# Output files
TRAIN_FILE = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\humanized_linear_dataset\labels_train.csv"
VAL_FILE = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\humanized_linear_dataset\labels_val.csv"

# Load labels.csv
df = pd.read_csv(LABELS_FILE)

# Split into train/val (80% train, 20% val)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save
train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)

print(f"✅ Split completed: {len(train_df)} train samples, {len(val_df)} validation samples")
print(f"Train file saved to {TRAIN_FILE}")
print(f"Val file saved to {VAL_FILE}")
