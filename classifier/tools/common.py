import re
from collections import defaultdict
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
    for entry in categories_column:
        all_categories.update(entry.strip().split())
    return sorted(all_categories)


def plot_analysis(df=None):
    if df is None or df.empty:
        # resource_json_path = Path(__file__).parent.parent.parent / "data" / f"{SAMPLE_SNAPSHOT}.json"
        # resource_json_path = Path(__file__).parent.parent.parent / "data" / f"arxiv-snapshot-unseen-sample_87%-json.json"
        resource_json_path = Path(__file__).parent.parent.parent / "data" / f"arxiv-snapshot-training-json.json"
        df = pd.read_json(resource_json_path)
        print(f"---> File {resource_json_path} loaded successfully.")

    # 1. Occurrences of each grouped category
    def group_categories(category):
        return f"{category.split('.')[0]}.*"

    df["grouped_categories"] = df["categories"].apply(lambda x: " ".join([group_categories(cat) for cat in x.split()]))
    grouped_category_counts = df["grouped_categories"].str.split(expand=True).stack().value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(grouped_category_counts.index, grouped_category_counts.values)
    plt.xticks(rotation=55, ha='right')
    plt.title("Number of Articles in Grouped Categories")
    plt.xlabel("Category Group")
    plt.ylabel("Count")
    plt.yscale('symlog', linthresh=10)
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticks(y_ticks)
    plt.gca().set_yticklabels([f"{int(tick)}" for tick in y_ticks])
    plt.tight_layout()
    plt.show()

    # 2. Distribution of abstract word counts with a horizontal bar chart
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

    # 3. Average Abstract Word Count per Category
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

    # 4. Number of entries with one, two, three, four or 5+ categories assigned
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