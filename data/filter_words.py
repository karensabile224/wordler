import pandas as pd

df = pd.read_csv("data/eng_word_freq.csv")

# keep only words that have exactly 5 letters
five_letter_words = df[df["word"].str.len() == 5]

# save the filtered CSV into the data directory
output_path = "data/filtered_words.csv"
five_letter_words.to_csv(output_path, index=False)

print(f"Filtered 5-letter words saved to {output_path}")


# assuming you're running this script from the wordler directory
# run `python3 data/filter_words.py` from terminal
