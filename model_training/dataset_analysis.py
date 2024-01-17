import os

# Specify the folder path you want to read files from
folder_path = "./dataset_after_preprocessing"

counts = dict()
overall = 0

# Check if the folder path exists
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through the files and print their names
    for file_name in sorted(file_list):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            f = open(file_path, "r")
            count = len(f.readlines())
            f.close()
            
            overall += count

            file_name = file_name.split("_")
            object_name = file_name[0]
            participant_name = file_name[1]

            if object_name not in counts:
                counts[object_name] = dict()
            
            if participant_name not in counts[object_name]:
                counts[object_name][participant_name] = [count]
            else:
                counts[object_name][participant_name].append(count)

else:
    print("The specified folder does not exist or is not a directory.")

for k,v in counts.items():
    print(k)
    for k2,v2 in v.items():
        print(k2, v2)
    print()

print("Overall: ", overall)