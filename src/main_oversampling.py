import pandas as pd
from constants import DATASET_SYNTHETIC_COLUMN, RANDOM_SEED
from deduplication import CLUSTER1_PROPORTION_COL, SemanticDeduplicator
from oversampling.vllm_oversampler import VllmOversampler
from data_loader import DataLoader
import time


SIMILARITY_THRESHOLD = 0.92
SYNTH_SIMILARITY_THRESHOLD = 0.92
N_SYNTH_SAMPLES = 5
N_RECORDS = None
DATA_PATH = "../data/processed/"
DEDUP0_MIXED_FILE = f"deduplicated_mixed_class0_{int(SIMILARITY_THRESHOLD*100)}.csv"
DEDUP1_PURE_FILE = f"deduplicated_pure_class1_{int(SIMILARITY_THRESHOLD*100)}.csv"
DEDUP1_MIXED_FILE = f"deduplicated_mixed_class1_{int(SIMILARITY_THRESHOLD*100)}.csv"
SYNTH_FILE = f"synth_pure_{N_SYNTH_SAMPLES}s.csv"
DEDUP_SYNTH_FILE = f"synth_pure_deduplicated_{N_SYNTH_SAMPLES}s_{int(SYNTH_SIMILARITY_THRESHOLD*100)}.csv"
MIN_TEXT_LENGTH = 100


def deduplicate():
    dl = DataLoader()
    df = pd.concat([dl._load_pulse_data(), dl._load_platform_data()], ignore_index=True)

    dd = SemanticDeduplicator()
    unique_df, cluster_df = dd.clustering_deduplication(df, threshold=SIMILARITY_THRESHOLD, shuffle=False, pure_only=True)
    # unique_df = unique_df[['content', 'status']]
    unique_c0 = unique_df[unique_df['status'] == 0]
    unique_c1m = unique_df[(unique_df['status'] == 1) & (unique_df[CLUSTER1_PROPORTION_COL] != 1)]
    unique_c1p = unique_df[(unique_df['status'] == 1) & (unique_df[CLUSTER1_PROPORTION_COL] == 1)]
    unique_c0.to_csv(DATA_PATH + DEDUP0_MIXED_FILE, index=False)
    unique_c1m.to_csv(DATA_PATH + DEDUP1_MIXED_FILE, index=False)
    unique_c1p.to_csv(DATA_PATH + DEDUP1_PURE_FILE, index=False)
    return df, unique_df, cluster_df

def deduplicate_synth():
    dd = SemanticDeduplicator(augment_features=False)
    df = pd.read_csv(DATA_PATH + SYNTH_FILE)
    unique_df, cluster_df = dd.clustering_deduplication(df, SYNTH_SIMILARITY_THRESHOLD, shuffle=False, pure_only=False)
    unique_df.to_csv(DATA_PATH + DEDUP_SYNTH_FILE, index=False)
    return unique_df, cluster_df

def run_oversampling():
    df = pd.read_csv(DATA_PATH + DEDUP1_PURE_FILE)
    df[DATASET_SYNTHETIC_COLUMN] = 0
    df["original_idx"] = df.index
    df = df.head(N_RECORDS) if N_RECORDS is not None else df

    os = VllmOversampler()

    start = time.time()
    print("\n")
    for index, row in df.iterrows():
        text = row['content']
        if len(text) >= MIN_TEXT_LENGTH:
            print(f"Generating sample {index}...")
            
            start_time = time.time()
            gen = os.generate(text, N_SYNTH_SAMPLES)
            print(f"Sample {index} generated in {time.time() - start_time:.2f} seconds")
            print(f"Original: {text}")

            if len(gen) != N_SYNTH_SAMPLES:
                print(f"Wrong output: {gen}")
                continue
            
            for s in gen:
                print(f"Generated: {s}")
                new_row = pd.DataFrame([{'content': s, 'status': 1, DATASET_SYNTHETIC_COLUMN: 1, 'original_idx': index}])
                df = pd.concat([df, new_row], ignore_index=True)
        else:
            print(f"Skipping short text: {text}")
    
    print(f"\nOversampling took {time.time() - start:.2f} seconds")
    print("Saving data...")
    df.to_csv(DATA_PATH + SYNTH_FILE, index=False)

if __name__ == "__main__":
    run_oversampling()
    # deduplicate_synth()