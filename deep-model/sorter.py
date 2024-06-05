import os
import shutil

# Source directory (you can change this to a specific path if needed)
source_directory = './data_no_blue'

# Destination directory
destination_directory = os.path.join(source_directory, 'ORGANIZED-DATA')

# Global counter for sequential naming
global_counter = 1

# Iterate through the top-level folders (different website names)
for top_folder in [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]:
    top_folder_path = os.path.join(source_directory, top_folder)

    # Iterate through the specific folders in the source directory
    for name in [d for d in os.listdir(top_folder_path) if os.path.isdir(os.path.join(top_folder_path, d)) and d != 'ORGANIZED-DATA']:
        name_path = os.path.join(top_folder_path, name)
        for windows_length in os.listdir(name_path):
            windows_length_path = os.path.join(name_path, windows_length)
            if os.path.isdir(windows_length_path):
                for overlapping in os.listdir(windows_length_path):
                    overlapping_path = os.path.join(windows_length_path, overlapping)
                    if os.path.isdir(overlapping_path):
                        for number_of_objects in os.listdir(overlapping_path):
                            number_of_objects_path = os.path.join(overlapping_path, number_of_objects)
                            if os.path.isdir(number_of_objects_path):
                                # Create a new directory for each windows_length/objects combination if it doesn't exist
                                new_folder_path = os.path.join(destination_directory, windows_length, number_of_objects)
                                if not os.path.exists(new_folder_path):
                                    os.makedirs(new_folder_path)

                                # Move files to the new directory
                                for file_name in os.listdir(number_of_objects_path):
                                    file_path = os.path.join(number_of_objects_path, file_name)
                                    if os.path.isfile(file_path):
                                        # Get the file extension and create a new sequential name
                                        file_extension = os.path.splitext(file_name)[1]
                                        new_file_name = f"{global_counter}{file_extension}"

                                        print(f'Moving file: {file_path} to {os.path.join(new_folder_path, new_file_name)}')
                                        shutil.move(file_path, os.path.join(new_folder_path, new_file_name))

                                        # Increment the global counter
                                        global_counter += 1

print("Sorting completed successfully.")