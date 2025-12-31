import pandas as pd
import glob
import os
import pickle

def main():
    #Set the directory where your extracted images and CSV are stored.
    #Adjust this path as needed
    data_dir = '/path/to/extracted/images'
    
    #Load the CSV file containing labels.
    csv_path = os.path.join(data_dir, "Data_Entry_2017_v2020.csv")
    df_csv = pd.read_csv(csv_path)
    
    #Gather all PNG file paths from subdirectories matching the pattern.
    image_files = glob.glob(os.path.join(data_dir, "images_*", "*", "*.png")) #This will find all PNG files in subdirectories of the form "images_001", "images_002", etc.
    print("Found {} image files.".format(len(image_files)))
    
    #Create a DataFrame of file paths.
    df_paths = pd.DataFrame({"file_path": image_files})
    #Extract the filename from each path (e.g., "00000003_000.png").
    df_paths["Image Index"] = df_paths["file_path"].apply(lambda x: os.path.basename(x))
    
    #Merge the file paths with the CSV on the "Image Index" column.
    merged_df = pd.merge(df_paths, df_csv, on="Image Index", how="inner")
    #Keep only relevant columns.
    merged_df = merged_df[["file_path", "Image Index", "Finding Labels"]]
    
    #Save the merged DataFrame as a pickle file for later use.
    output_pickle = os.path.join(data_dir, "merged_dataset.pkl")
    with open(output_pickle, "wb") as f:
        pickle.dump(merged_df, f)
    print("Merged dataset saved to:", output_pickle)

#The .pkl file just saves a map of filenames to labels, so you can quickly load that information later and then open the actual image files from disk when needed for training.

#The below ensures that the main() function runs only if this script is executed directly, and not when it’s imported as a module.
if __name__ == "__main__":
    main()