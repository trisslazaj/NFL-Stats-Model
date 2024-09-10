import os
import pandas as pd

def load_aggregate_data(data_dir):
    all_data = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            data = pd.read_csv(filepath)
            all_data.append(data) #Place holder probs dont put all data in one spot lmaoo 

    aggregated_data = pd.concat(all_data, ignore_index=True)
    return aggregated_data

def save_aggregated_data(data, output_filepath):
    data.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    data_dir = '\Users\Triss Lazaj\Desktop\nfl_stats_model\data'
    output_filepath = ""

    aggregated_data = load_aggregate_data(data_dir)

    save_aggregated_data(aggregated_data, output_filepath)

    