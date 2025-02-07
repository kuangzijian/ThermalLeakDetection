import math
import shutil
import os
import pyodbc

source_folder = "//nsight01/nsight/"
dest_img_folder = "dataset/images/"
dest_label_folder = "dataset/labels/"
dest_diff_map_folder = "dataset/diff_maps/"
server = 'SCADATESTDB-VM3'
database = 'nTerface'
username = 'sa'
password = 'Butterfly$'

# Function to write the decimal arrays to a text file
def write_to_file(decimal_arrays: list[list[float]], file_name: str) -> None:
    with open(file_name, 'w') as file:
        for array in decimal_arrays:
            # Convert each array to a string of space-separated values with a 0 at the beginning
            array_str = '0 ' + ' '.join(f'{x:.4f}' for x in array)
            # Write the string to the file followed by a newline
            file.write(array_str + '\n')

# Make directories
def create_directory(path, description):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
            print(f'Created {description} folder: {path}')
        except Exception as e:
            print(f'Error creating {description} folder: {path} - {e}')
            exit(0)

create_directory('dataset/', 'dataset')
create_directory(dest_img_folder, 'images')
create_directory(dest_label_folder, 'labels')
create_directory(dest_diff_map_folder, 'diff maps')

# ENCRYPT defaults to yes starting in ODBC Driver 18. It's good to always specify ENCRYPT=yes on the client side to avoid MITM attacks.
conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';ENCRYPT=yes;UID='+username+';PWD='+ password +';TrustServerCertificate=yes')
cursor = conn.cursor()
print('successfully connected')

# Dynamic date calculation
start_date = '2024-07-17 08:05:38.997'
end_date = '2024-07-26 04:54:28.667'

# Step 1: Calculate TotalGroups
total_groups_query = """
SELECT 
    COUNT(*) AS TotalGroups
FROM (
    SELECT DISTINCT 
        I.clientId, I.facility, I.cameraNo
    FROM 
        [nTerface].[nVision].[imagesToProcess] I
        INNER JOIN [nTerface].[nVision].[result] R ON I.imageId = R.imageId 
        LEFT JOIN [nTerface].[nVision].[detectedObjects] d ON d.imageId = I.imageId 
    WHERE 
        I.processed = 1 
        AND I.imageType = 'thermal' 
        AND R.result = 1 
        AND I.timeStamp BETWEEN ? AND ?
) AS GroupedData;
"""
cursor.execute(total_groups_query, start_date, end_date)
total_groups = cursor.fetchone()[0]

print(f'Total thermal cameras: {str(total_groups)}' )
# Calculate records per group
total_records = 1000
records_per_group = math.ceil(total_records / total_groups)

print(f'Records per thermal camera: {str(records_per_group)}' )

# Step 2: Retrieve Records
records_query = f"""
WITH RandomizedImages AS (
    SELECT 
        I.clientId, 
        I.facility, 
        I.cameraNo, 
        I.date, 
        I.imageName,
        I.imageId,
        ROW_NUMBER() OVER (PARTITION BY I.clientId, I.facility, I.cameraNo ORDER BY NEWID()) AS RowNum
    FROM 
        [nTerface].[nVision].[imagesToProcess] I
        INNER JOIN [nTerface].[nVision].[result] R ON I.imageId = R.imageId 
        LEFT JOIN [nTerface].[nVision].[detectedObjects] d ON d.imageId = I.imageId 
    WHERE 
        I.processed = 1 
        AND I.imageType = 'thermal' 
        AND R.result = 1 
        AND I.timeStamp BETWEEN ? AND ?
)
SELECT TOP {total_records}
    clientId, 
    facility, 
    cameraNo, 
    date, 
    imageName,
    imageId
FROM 
    RandomizedImages
WHERE 
    RowNum <= {records_per_group}
ORDER BY 
    NEWID();
"""
cursor.execute(records_query, start_date, end_date)

# Fetch the results
images = cursor.fetchall()
print(f'Total number: {str(len(images))}')

for img in images:
    source = f'{source_folder}/{img[0]}/{img[1]}/{img[2]}/{img[3]}/{img[4]}'
    diff_map_source = f'{source_folder}/LeakAI/diff_maps/{img[0]}/{img[1]}/{img[2]}/{img[3]}/{img[4]}'

    if not (os.path.isfile(f'{dest_img_folder}/{img[4]}')):
        try:
            new_filename = f'{img[1]}-{img[2]}-{img[4]}'
            dest_img_path = os.path.join(dest_img_folder, new_filename)
            shutil.copy2(source, dest_img_path)
            print(f'ImageID - {img[5]} downloaded image: {source} to: {dest_img_folder}')

            dest_diff_map_path = os.path.join(dest_diff_map_folder, new_filename)
            shutil.copy2(diff_map_source, dest_diff_map_path)
            print(f'ImageID - {img[5]} download diff map: {source} to: {dest_diff_map_folder}')

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print('skip image: ' + source)

    # download labels
    query = f"""SELECT TOP 10
      [boundingBox_xc]
      ,[boundingBox_yc]
      ,[boundingBoxWidth]
      ,[boundingBoxHeight]
      FROM [nTerface].[nVision].[detectedObjects]
      where imageId = '{img[5]}'"""
    with cursor.execute(query) as cursor:
        labels = cursor.fetchall()
        # Call the function to write the data to the file
        write_to_file(labels, dest_label_folder + new_filename.replace('.jpg', '.txt'))
        print(f'ImageID - {img[5]} download label to: {dest_label_folder}')


print('All images downloaded')