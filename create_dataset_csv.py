import os
import pandas as pd

def create_csv(dataset_dir='Train', output_file='dataset.csv'):
    """
    Creates a CSV file with image filenames and their corresponding labels (folder names).
    """
    data = []
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory '{dataset_dir}' not found.")
        return

    # Walk through the directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            # Check for valid image extensions if needed, for now assuming all files in subfolders are images
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Get the label from the folder name
                label = os.path.basename(root)
                
                # Skip if the file is in the root dataset directory (no label)
                if root == dataset_dir:
                    continue
                    
                data.append({'gambar': file, 'label': label})

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully created '{output_file}' with {len(df)} entries.")

if __name__ == "__main__":
    create_csv()
