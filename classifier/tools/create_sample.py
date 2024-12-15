import json
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

from classifier.tools.common import SNAPSHOT, get_unique_categories, plot_analysis


def create_sample(rows_per_category=None, sample_frac=0.25, save_unseen=False):
    datasets_dir = Path(__file__).resolve().parents[2] / "data" / "datasets"
    resource_json_path = datasets_dir / f"{SNAPSHOT}-training-json.json"
    df = pd.read_json(resource_json_path)
    df = df[["abstract", "categories"]]
    size = len(df)

    # Shuffle the original DataFrame
    df = df.sample(frac=1, random_state=22).reset_index(drop=True)
    plot_analysis(df, df_title="Original Dataset")

    df["id"] = df.index
    df["category_list"] = df["categories"].str.split()
    expanded_df = df.explode("category_list")

    # 1. Sample equally distributed number of entries for each category as in the original dataset
    # sampled_ids = (
    #     expanded_df.groupby("category_list", group_keys=False)
    #     .apply(lambda x: x.sample(frac=sample_frac, random_state=22), include_groups=False)
    #     .id.unique()
    # )

    # 2. Sample an equal number of entries for each category (slow)
    all_categories = get_unique_categories(df["categories"])

    if not rows_per_category:
        sample_size = int(sample_frac * size)
        rows_per_category = sample_size // len(all_categories)
    else:
        sample_size = rows_per_category * len(all_categories)

    category_counts = sorted(Counter(expanded_df['category_list']).items(), key=lambda x: x[1])

    print(f"Pick at least {rows_per_category} entries from each of the {len(all_categories)} categories if available")
    rare_categories = defaultdict(int)
    balanced_sample = pd.DataFrame(columns=df.columns)
    # remaining_categories = list(category_counts)

    for i, (category, _) in enumerate(category_counts):
        category_rows = expanded_df[expanded_df['category_list'] == category].drop_duplicates()
        category_rows = category_rows.loc[~category_rows.index.isin(balanced_sample.index)]

        if len(category_rows) < rows_per_category:
            rare_categories[category] = len(category_rows)
            remaining_rows = sample_size - len(balanced_sample) - len(category_rows)
            remaining_categories_count = len(all_categories) - i - 1
            rows_per_category = remaining_rows // remaining_categories_count if remaining_categories_count > 0 else remaining_rows

        sampled_rows = category_rows.sample(n=min(rows_per_category, len(category_rows)), random_state=42)
        balanced_sample = pd.concat([balanced_sample, sampled_rows])

    balanced_sample.reset_index(drop=True, inplace=True)

    print(f"Rare categories:\n {rare_categories.keys()}")

    remaining_sample_size = sample_size - len(balanced_sample)
    if remaining_sample_size > 0:
        remaining_rows = expanded_df.drop(balanced_sample.index).drop_duplicates()
        extra_sample = remaining_rows.sample(remaining_sample_size, random_state=42)
        balanced_sample = pd.concat([balanced_sample, extra_sample])

    sample_df = balanced_sample.drop_duplicates().reset_index(drop=True)

    # 3. Sample an equal number of entries for each category (fast)
    # sampled_ids = (
    #     expanded_df.groupby("category_list", group_keys=False)
    #     .apply( lambda x: x.sample(n=min(len(x), rows_per_category), random_state=22) )
    #     .id.unique()
    # )
    # sample_df = df[df["id"].isin(sampled_ids)][["abstract", "categories"]]

    if save_unseen:
        unseen_df = df.loc[~df.index.isin(sample_df.index)]
        unseen_df = unseen_df[["abstract", "categories"]]
        unseen_json_path = datasets_dir / f"{SNAPSHOT}-unseen-sample-json.json"
        data_list = unseen_df.to_dict(orient="records")
        with unseen_json_path.open("w", encoding="utf-8") as file:
            json.dump(data_list, file, ensure_ascii=False, indent=2)
        print(f"Number of unseen entries {len(unseen_df)} written to {unseen_json_path}")
        del unseen_df

    del df

    # Double check as we cannot have categories that appear only once
    sample_df["category_list"] = sample_df["categories"].str.split()
    expanded_df = sample_df.explode("category_list")
    unique_cats = [cat for cat, count in Counter(expanded_df['category_list']).items() if count == 1]

    if unique_cats:
        rows_with_unique_cats = sample_df["category_list"].apply(
            lambda cats: any(c in unique_cats for c in cats)
        )
        print("Categories that appear only once:", unique_cats)
        print("Number of entries that will be dropped:", rows_with_unique_cats.sum())
        sample_df = sample_df[~rows_with_unique_cats]

    del expanded_df
    sample_df = sample_df[["abstract", "categories"]]
    plot_analysis(sample_df, df_title="Sample Dataset")

    perc = round(len(sample_df)*100 / size, 2)
    print(f"Number of entries : {len(sample_df)}, {perc}%")
    perc_str = f"{str(int(perc))}" if perc > 1 else f"{str(perc).replace('.', '_')}"

    sample_json_path = datasets_dir / f"{SNAPSHOT}-sample_{perc_str}%-json.json"
    data_list = sample_df.to_dict(orient="records")
    with sample_json_path.open("w", encoding="utf-8") as file:
        json.dump(data_list, file, ensure_ascii=False, indent=2)


def convert_to_json_array(file_name):
    """Convert original file to JSON array that can be loaded by Pandas straightforward"""
    resource_path = Path(__file__).parent.parent.parent / "data" / f"{file_name}.json"
    resource_json_path = (
        Path(__file__).parent.parent.parent / "data" / f"{SNAPSHOT}-json.json"
    )

    with resource_path.open("r") as in_file:
        is_first = True
        with resource_json_path.open("w") as out_file:
            out_file.write("[")
            for line in in_file:
                if is_first:
                    out_file.write(line)
                    is_first = False
                else:
                    out_file.write(f",{line}")
            out_file.write("]\n")


if __name__ == "__main__":
    create_sample(sample_frac=0.24, save_unseen=True)
