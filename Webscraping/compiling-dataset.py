import pandas as pd
from datasets import Dataset

def combine_data(squad_df, scorecard_df, match_info_df, dream_team_df):
    combined_data = []
    for match_id in squad_df['match_id'].unique():
        squad_info = squad_df[squad_df['match_id'] == match_id]['player_names'].values[0]
        scorecard_info = scorecard_df[scorecard_df['match_id'] == match_id]['scorecard_info'].values[0]
        match_info = match_info_df[match_info_df['match_id'] == match_id]['match_info'].values[0]
        dream_team = dream_team_df[dream_team_df['match_id'] == match_id]['dream_team'].values[0]
        combined_data.append({
            'input': f"Generate the combined playing-11 from the following information: {squad_info} {scorecard_info} {match_info}",
            'output': dream_team
        })
    return pd.DataFrame(combined_data)

# Load CSV files
squad_df = pd.read_csv("IPLsquad.csv")
scorecard_df = pd.read_csv("IPLscorecard.csv")
match_info_df = pd.read_csv("IPLmatch_info.csv")
dream_team_df = pd.read_csv("IPLdream_team.csv")

# Combine data
combined_dataset = combine_data(squad_df, scorecard_df, match_info_df, dream_team_df)

# Shuffle the dataset
combined_dataset = combined_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Split dataset into train and eval
train_size = int(0.8 * len(combined_dataset))
train_dataset = combined_dataset[:train_size]
eval_dataset = combined_dataset[train_size:]

# Convert to Hugging Face Dataset format
combined_huggingface_dataset = Dataset.from_pandas({"train": train_dataset, "eval": eval_dataset})

# Save the combined dataset
combined_huggingface_dataset.save_to_disk("combined_playing_11_dataset")