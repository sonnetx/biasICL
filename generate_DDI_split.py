import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_ddi_splits_new(data_path, test_size=0.25, random_state=42, benign_ratio=0.5):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Remove rare diseases (in a set for faster lookup)
    rare_diseases = {
        'subcutaneous-t-cell-lymphoma', 'focal-acral-hyperkeratosis', 
        'eccrine-poroma', 'inverted-follicular-keratosis', 'kaposi-sarcoma',
        'metastatic-carcinoma', 'mycosis-fungoides', 
        'acquired-digital-fibrokeratoma', 'atypical-spindle-cell-nevus-of-reed',
        'verruciform-xanthoma', 'morphea', 'nevus-lipomatosus-superficialis',
        'pigmented-spindle-cell-nevus-of-reed', 'arteriovenous-hemangioma',
        'syringocystadenoma-papilliferum', 'trichofolliculoma',
        'coccidioidomycosis', 'leukemia-cutis', 'sebaceous-carcinoma',
        'blastic-plasmacytoid-dendritic-cell-neoplasm', 'glomangioma',
        'dermatomyositis', 'cellular-neurothekeoma', 'graft-vs-host-disease',
        'xanthograngioma', 'chondroid-syringoma', 'angioleiomyoma'
    }
    df = df[~df['disease'].isin(rare_diseases)]

    # drop skin tone group '34'
    df = df[df['skin_tone'] != 34]

    # Step 1: Split data by skin tone
    skin_tone_groups = {tone: group for tone, group in df.groupby('skin_tone')}

    # check benign/malignant balance in each skin tone group
    for skin_tone, group in skin_tone_groups.items():
        malignant = group[group['malignant'] == 1]
        benign = group[group['malignant'] == 0]
        print(f"Skin tone {skin_tone}:")
        print("Malignant:", len(malignant))
        print("Benign:", len(benign))
        print("Malignant/Benign ratio:", len(malignant) / len(benign))
        print()
    
    # Step 2: Create train-test splits for each skin tone
    train_dfs = []
    test_dfs = []
    
    for skin_tone, group in skin_tone_groups.items():
        train_group, test_group = train_test_split(
            group, test_size=test_size, random_state=random_state
        )
        
        # Balance benign/malignant in training set
        malignant_train = train_group[train_group['malignant'] == 1]
        benign_train = train_group[train_group['malignant'] == 0]
        
        # Calculate target number of benign samples
        n_malignant = len(malignant_train)
        target_benign = int(n_malignant * benign_ratio / (1 - benign_ratio))
        
        if len(benign_train) < target_benign:
            # raise ValueError(
            #     f"Not enough benign samples for skin tone {skin_tone}. "
            #     f"Need {target_benign}, have {len(benign_train)}"
            # )
            print(f"Not enough benign samples for skin tone {skin_tone}. "
            f"Need {target_benign}, have {len(benign_train)}")
            target_benign = len(benign_train)
        
        # Sample benign cases to match target ratio
        benign_train = benign_train.sample(
            n=target_benign, random_state=random_state
        )
        
        train_dfs.append(pd.concat([malignant_train, benign_train]))
        test_dfs.append(test_group)
    
    # Step 3: Balance skin tones in training set
    min_train_samples = min(len(df) for df in train_dfs)
    balanced_train_dfs = [
        df.sample(n=min_train_samples, random_state=random_state)
        for df in train_dfs
    ]
    
    # Step 4: Combine and prepare final datasets
    train_df = pd.concat(balanced_train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Create output DataFrames with required columns
    columns = ['DDI_file', 'malignant', 'skin_tone']
    train_output = train_df[columns].copy()
    test_output = test_df[columns].copy()
    
    # Add benign column
    train_output['benign'] = (~train_output['malignant'].astype(bool)).astype(int)
    test_output['benign'] = (~test_output['malignant'].astype(bool)).astype(int)
    
    # Print balance statistics
    def print_balance_stats(name, df):
        print(f"\n{name} set balance:")
        print("Skin tone distribution:")
        print(df['skin_tone'].value_counts(normalize=True))
        print("\nMalignant/Benign distribution:")
        print(df['malignant'].value_counts(normalize=True))
    
    print_balance_stats("Training", train_output)
    print_balance_stats("Test", test_output)
    
    return train_output, test_output


def generate_ddi_splits(data_path, test_size=0.75, random_state=42, benign_ratio=0.5):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Remove rare diseases (in a set for faster lookup)
    rare_diseases = {
        'subcutaneous-t-cell-lymphoma', 'focal-acral-hyperkeratosis', 
        'eccrine-poroma', 'inverted-follicular-keratosis', 'kaposi-sarcoma',
        'metastatic-carcinoma', 'mycosis-fungoides', 
        'acquired-digital-fibrokeratoma', 'atypical-spindle-cell-nevus-of-reed',
        'verruciform-xanthoma', 'morphea', 'nevus-lipomatosus-superficialis',
        'pigmented-spindle-cell-nevus-of-reed', 'arteriovenous-hemangioma',
        'syringocystadenoma-papilliferum', 'trichofolliculoma',
        'coccidioidomycosis', 'leukemia-cutis', 'sebaceous-carcinoma',
        'blastic-plasmacytoid-dendritic-cell-neoplasm', 'glomangioma',
        'dermatomyositis', 'cellular-neurothekeoma', 'graft-vs-host-disease',
        'xanthograngioma', 'chondroid-syringoma', 'angioleiomyoma'
    }
    df = df[~df['disease'].isin(rare_diseases)]

    fst_12 = df[df['skin_tone'] == 12]
    fst_56 = df[df['skin_tone'] == 56]

    # Split each slice
    train_12, test_12 = train_test_split(fst_12, test_size=test_size)
    train_56, test_56 = train_test_split(fst_56, test_size=test_size)

    # Split the training split into benign and malignant
    # Function to balance the benign and malignant samples
    def balance_benign_malignant(train_df, benign_ratio, random_state):
        malignant = train_df[train_df['malignant'] == 1]
        benign = train_df[train_df['malignant'] == 0]

        # Calculate required number of benign samples based on benign_ratio
        total = len(malignant) / (1 - benign_ratio)
        target_benign_count = int(total - len(malignant))

        # Adjust the benign count to match the desired ratio
        if len(benign) >= target_benign_count:
            benign = benign.sample(target_benign_count, random_state=random_state)
        else:
            raise ValueError(f"Not enough benign samples to achieve the desired ratio. "
                             f"Required: {target_benign_count}, Available: {len(benign)}")

        return pd.concat([malignant, benign])

    # Apply balancing to the training splits
    train_12_balanced = balance_benign_malignant(train_12, benign_ratio, random_state)
    train_56_balanced = balance_benign_malignant(train_56, benign_ratio, random_state)

    # Balance patient type
    def balance_patient_type(train_df, random_state):
        fst_12 = train_df[train_df['skin_tone'] == 12]
        fst_56 = train_df[train_df['skin_tone'] == 56]

        # Calculate required number of patient samples based on the smaller group
        total = min(len(fst_12), len(fst_56))
        target_12_count = int(total)
        target_56_count = int(total)

        # Adjust the benign count to match the desired ratio
        if len(fst_12) >= target_12_count:
            fst_12 = fst_12.sample(target_12_count, random_state=random_state)
        if len(fst_56) >= target_56_count:
            fst_56 = fst_56.sample(target_56_count, random_state=random_state)

        return pd.concat([fst_12, fst_56])

    # Apply balancing to the training splits
    train_12_balanced = balance_patient_type(train_12_balanced, random_state)
    train_56_balanced = balance_patient_type(train_56_balanced, random_state)

    # Combine the splits
    demo_df = pd.concat([train_12_balanced, train_56_balanced])
    test_df = pd.concat([test_12, test_56])

    
    # Check balance
    def check_balance(df):
        fst_balance = df['skin_tone'].value_counts(normalize=True)
        malignant_balance = df['malignant'].value_counts(normalize=True)
        return fst_balance, malignant_balance
    
    train_fst_balance, train_malignant_balance = check_balance(demo_df)
    test_fst_balance, test_malignant_balance = check_balance(test_df)
    
    print("Train set balance:")
    print("FST balance:", train_fst_balance)
    print("Malignant balance:", train_malignant_balance)
    print("\nTest set balance:")
    print("FST balance:", test_fst_balance)
    print("Malignant balance:", test_malignant_balance)
    
    # Create new DataFrames with desired structure
    def create_output_df(df):
        print(df.columns)
        output_df = df[['DDI_file', 'malignant', 'skin_tone']]
        output_df.loc[:, 'benign'] = (~output_df['malignant']).astype(int)
        output_df.loc[:, 'malignant'] = output_df['malignant'].astype(int)
        print(output_df.columns)
        return output_df
    
    demo_output = create_output_df(demo_df)
    test_output = create_output_df(test_df)
    
    return demo_output, test_output

# Usage

data_path = '/home/groups/roxanad/ddi/ddi_metadata.csv'
demo_df, test_df = generate_ddi_splits_new(data_path, benign_ratio=0.75)

print("demo_df", len(demo_df))
print("test_df", len(test_df))

# Save the splits
demo_df.to_csv('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/DDI/ddi_demo.csv')
test_df.to_csv('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/DDI/ddi_test.csv')
