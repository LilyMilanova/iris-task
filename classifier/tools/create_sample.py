import json
from collections import Counter
from pathlib import Path
import pandas as pd

from classifier.tools.common import SNAPSHOT


def create_sample(sample_frac=0.80):
    resource_json_path = (Path(__file__).parent.parent.parent / "data" / f"{SNAPSHOT}-training-json.json")
    df = pd.read_json(resource_json_path)
    df = df[["abstract", "categories"]]
    # print(df.info())

    if sample_frac:
        # Shuffle the original DataFrame
        df = df.sample(frac=1, random_state=22).reset_index(drop=True)

        size = len(df)
        df["id"] = df.index
        df["category_list"] = df["categories"].str.split()
        expanded_df = df.explode("category_list")

        # 1. Sample equally distributed number of entries for each category as in the original dataset
        sampled_ids = (
            expanded_df.groupby("category_list", group_keys=False)
            .apply(lambda x: x.sample(frac=sample_frac, random_state=22), include_groups=False)
            .id.unique()
        )

        # 2. Sample an equal number of entries for each category (not balanced because of entries with >1 category)
        # all_categories = {cat for cats in df["category_list"] for cat in cats}
        #
        # Sample enough to reach the desired sample fraction
        # samples_per_category = int(sample_frac * size) // len(all_categories)
        # print(f"Pick {samples_per_category} entries per each of the {len(all_categories)} categories")
        # sampled_ids = (
        #     expanded_df.groupby("category_list", group_keys=False)
        #     .apply( lambda x: x.sample(n=min(len(x), samples_per_category), random_state=22) )
        #     .id.unique()
        # )
        sample_df = df[df["id"].isin(sampled_ids)][["abstract", "categories"]]
        unseen_df = df[~df["id"].isin(sampled_ids)][["abstract", "categories"]]

        training_json_path = (Path(__file__).parent.parent.parent / "data" / f"{SNAPSHOT}-unseen-sample-json.json")
        data_list = unseen_df.to_dict(orient="records")
        with training_json_path.open("w", encoding="utf-8") as file:
            json.dump(data_list, file, ensure_ascii=False, indent=2)

        del df, unseen_df

        sample_df["category_list"] = sample_df["categories"].str.split()
        all_categories = [cat for cats in sample_df["category_list"] for cat in cats]
        category_counts = Counter(all_categories)
        unique_cats = [cat for cat, count in category_counts.items() if count == 1]

        rows_with_unique_cats = sample_df["category_list"].apply(
            lambda cats: any(c in unique_cats for c in cats)
        )
        num_rows_to_drop = rows_with_unique_cats.sum()

        print("Categories that appear only once:", unique_cats)
        print("Number of entries that will be dropped:", num_rows_to_drop)

        if num_rows_to_drop:
            sample_df = sample_df[~rows_with_unique_cats]

        sample_df = sample_df.drop(columns=["category_list"])
    else:
        sample_df = df

    # print(sample_df.info())
    percentage = (len(sample_df) / size) * 100
    print(f"Number of entries : {len(sample_df)}, {percentage}%")
    percentage_str = f"_{str(int(percentage))}%"
    training_json_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / f"{SNAPSHOT}-sample{percentage_str}-json.json"
    )
    data_list = sample_df.to_dict(orient="records")
    with training_json_path.open("w", encoding="utf-8") as file:
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
    create_sample()
