import pickle

def parse_pickle(file_path):
    """
    Parse and inspect the contents of a pickle file.

    Args:
        file_path (str): Path to the pickle file.
    
    Returns:
        Parsed data from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def inspect_pickle_content(data):
    """
    Inspect the structure and key information of parsed pickle data.

    Args:
        data: Data parsed from a pickle file.
    """
    if isinstance(data, dict):
        print("Keys in the pickle data:")
        for key in data.keys():
            print(f" - {key}")
            if isinstance(data[key], (list, dict)):
                print(f"   -> Substructure: {type(data[key])}, Length: {len(data[key])}")
            else:
                print(f"   -> Value: {data[key]}")
    else:
        print(f"Data structure is {type(data)}")
        print("Sample data:", data)

if __name__ == "__main__":
    file_path = "C:/Users/sonne/Documents/Github/biasICL/ddi_results/ddi_0_0_0_0_gpt-4o-2024-05-13_50.pkl"
    parsed_data = parse_pickle(file_path)
    inspect_pickle_content(parsed_data)
