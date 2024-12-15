import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

SNAPSHOT = "arxiv-snapshot"
SAMPLE_SNAPSHOT = "small-sample-json"

MINI_MODEL_NAME = "prajjwal1/bert-mini"
MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LENGTH = 200

_stopwords_set = None


def get_stopwords():
    global _stopwords_set
    if not _stopwords_set:
        import nltk
        nltk.download("stopwords")

        from nltk.corpus import stopwords
        _stopwords_set = set(stopwords.words("english"))
    return _stopwords_set


def preprocess_abstract(abstract: str) -> str:
    text = re.sub(r"[^\w\s]", "", abstract.strip().lower())
    text = re.sub(r"\d+", "", text)
    # tokens = word_tokenize(text.lower())
    return " ".join([word for word in text.split() if word not in get_stopwords()])


def get_unique_categories(categories_column):
    all_categories = set()
    for categories in categories_column:
        all_categories.update(categories.strip().split())
    return sorted(all_categories)


def plot_analysis(df=None, id2cat=None, df_title=None):
    if df is None or df.empty:
        datasets_dir = Path(__file__).resolve().parents[2] / "data" / "datasets"
        # resource_json_path = datasets_dir / f"{SAMPLE_SNAPSHOT}.json"
        # resource_json_path = datasets_dir / f"arxiv-snapshot-unseen-sample_87%-json.json"
        resource_json_path = datasets_dir / f"arxiv-snapshot-training-json.json"
        df = pd.read_json(resource_json_path)
        print(f"---> File {resource_json_path} loaded successfully.")

    # Recreate the "categories" column if "label" column exists
    if "label" in df.columns and id2cat is not None:
        df["categories"] = df["label"].apply(
            lambda labels: " ".join([id2cat[idx] for idx, value in enumerate(labels) if value == 1])
        )

    # 1. Occurrences of each category (fine-grained adjustment)
    category_counts = df['categories'].str.split(expand=True).stack().value_counts().sort_index()
    plt.figure(figsize=(10, 6))

    counts = category_counts.values
    threshold = np.percentile(counts, 66)  # Bottom 66% threshold
    def adjust_height_category(value):
        if value <= threshold:
            return value * 10  # Fine-grained adjustment for bottom values
        else:
            return threshold * 10 + (value - threshold) / 6  # Coarse-grained adjustment for top values

    adjusted_heights_category = [adjust_height_category(value) for value in counts]
    plt.bar(category_counts.index, adjusted_heights_category)
    plt.xticks(rotation=55, ha='right')
    plt.title(f"Occurrences of Each Category{' in ' + df_title if df_title else ''}")
    plt.xlabel("Category")
    plt.ylabel("Count (Adjusted)")

    # Adjust y-axis ticks to reflect the adjusted heights
    y_ticks_adjusted_category = np.linspace(0, max(adjusted_heights_category), 10, dtype=int)
    y_ticks_original_category = [round(
        threshold * (tick / max(adjusted_heights_category)) if tick <= threshold * 10 else threshold + (
                    tick - threshold * 10) * 6, 1) for tick in y_ticks_adjusted_category]

    plt.ylim(0, max(adjusted_heights_category))
    plt.gca().set_yticks(y_ticks_adjusted_category)
    plt.gca().set_yticklabels([f"{int(tick)}" for tick in y_ticks_original_category])

    plt.tight_layout()
    plt.show()

    # 2. Occurrences of each grouped category
    def group_categories(category):
        return f"{category.split('.')[0]}.*"

    df["grouped_categories"] = df["categories"].apply(
        lambda x: " ".join([group_categories(cat) for cat in x.split()]))
    grouped_category_counts = df["grouped_categories"].str.split(expand=True).stack().value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    counts = grouped_category_counts.values
    threshold = np.percentile(counts, 50)  # Bottom 66% threshold
    def adjust_height(value):
        if value <= threshold:
            return value * 100  # Fine-grained adjustment for bottom values
        else:
            return threshold * 100 + (value - threshold) / 6  # Coarse-grained adjustment for top values

    adjusted_heights = [adjust_height(value) for value in counts]

    plt.bar(grouped_category_counts.index, adjusted_heights)
    plt.xticks(rotation=55, ha='right')
    plt.title(f"Number of Articles in Grouped Categories{' in ' + df_title if df_title else ''}")
    plt.xlabel("Category Group")
    plt.ylabel("Count")
    y_ticks_adjusted = np.linspace(0, max(adjusted_heights), 10, dtype=int)
    y_ticks_original = [round(threshold * (tick / max(adjusted_heights))
                        if tick <= threshold * 100
                        else threshold + (tick - threshold * 100) * 6, 1)
                        for tick in y_ticks_adjusted]
    plt.ylim(0, max(adjusted_heights))
    plt.gca().set_yticks(y_ticks_adjusted)
    plt.gca().set_yticklabels([f"{int(tick)}" for tick in y_ticks_original])

    plt.tight_layout()
    plt.show()

    if df_title:
        return

    # 3. Distribution of abstract word counts with a horizontal bar chart
    abstract_word_counts = df['abstract'].apply(lambda x: len(preprocess_abstract(x).split()))
    last = abstract_word_counts.max()
    word_count_bins = np.histogram(abstract_word_counts, bins=list(range(20, last // 3, 20)) + [last])
    bin_edges = word_count_bins[1].astype(int)
    bin_values = word_count_bins[0]
    filtered = np.where(bin_values > 0)[0]

    plt.figure(figsize=(8, 6))
    plt.barh(y=filtered, width=bin_values[filtered], height=0.8, alpha=0.75)
    plt.yticks(
        ticks=filtered,
        labels=[f"{bin_edges[i]}+" if i == len(bin_edges) - 2 else f"{bin_edges[i]}" for i in filtered]
    )
    plt.title('Distribution of Abstract Word Counts (after pre-processing)')
    plt.ylabel('Average Word Count of Abstract')
    plt.xlabel('Number of Articles')
    plt.tight_layout()
    plt.show()

    # 4. Average Abstract Word Count per Category
    def length_per_category(df):
        category_word_counts = defaultdict(list)
        for _, row in df.iterrows():
            abs_word_count = len(preprocess_abstract(row["abstract"]).split())
            for category in row["categories"].split():
                category_word_counts[category].append(abs_word_count)
        return {cat: np.mean(lengths) for cat, lengths in category_word_counts.items()}

    category_word_counts = length_per_category(df)
    sorted_categories = dict(sorted(category_word_counts.items()))

    # Split into two parts
    mid_point = len(sorted_categories) // 2
    first_half = dict(list(sorted_categories.items())[:mid_point])
    second_half = dict(list(sorted_categories.items())[mid_point:])

    plt.figure(figsize=(12, 5))
    plt.bar(first_half.keys(), first_half.values())
    plt.gca().tick_params(axis='x', which='major', labelsize=7)
    plt.gca().set_xlim(left=-0.6, right=len(first_half) - 0.5)
    plt.gca().set_xticks(range(len(first_half.keys())))
    plt.gca().set_xticklabels(first_half.keys(), rotation=55, ha='right')
    plt.title("Average Abstract Word Count per Category (First Half)")
    plt.xlabel("Category")
    plt.ylabel("Average Word Count")
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 5))
    plt.bar(second_half.keys(), second_half.values())
    plt.gca().tick_params(axis='x', which='major', labelsize=7)
    plt.gca().set_xlim(left=-0.6, right=len(second_half) - 0.5)
    plt.gca().set_xticks(range(len(second_half.keys())))
    plt.gca().set_xticklabels(second_half.keys(), rotation=55, ha='right')
    plt.title("Average Abstract Word Count per Category (Second Half)")
    plt.xlabel("Category")
    plt.ylabel("Average Word Count")
    plt.tight_layout()
    plt.show()

    # 5. Number of entries with one, two, three, four or 5+ categories assigned
    def count_entries_by_category_count(categories_column):
        counts = categories_column.apply(lambda x: len(x.split()))
        bins = [1, 2, 3, np.inf]
        labels = ["1", "2", "3+"]
        return pd.cut(counts, bins=bins, labels=labels).value_counts().sort_index()

    category_distribution = count_entries_by_category_count(df["categories"])
    plt.figure(figsize=(8, 5))
    wedges, texts, autotexts = plt.pie(
        category_distribution.values, labels=None, autopct=lambda pct: ('%1.1f%%' % pct) if pct > 0 else '',
        startangle=140
    )
    plt.legend(wedges, category_distribution.index, title="Number of Categories", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("Distribution of Articles by Number of Categories")
    plt.tight_layout()
    plt.show()