import os
import pyodbc
from datetime import datetime

##CONNECT TO SQL SERVER
server = 'SCADATEST-VM3'
database = 'nSight' 
username = 'sa'
password = 'Butterfly$;'
# ENCRYPT defaults to yes starting in ODBC Driver 18. It's good to always specify ENCRYPT=yes on the client side to avoid MITM attacks.
conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';ENCRYPT=yes;UID=' + username + ';PWD=' + password + ';TrustServerCertificate=yes')
cursor = conn.cursor()
FOLDER_PATH = '//nsight01/nsight/9999/'
clientId='9999'

# List all files in the folder
for facility in os.listdir(FOLDER_PATH):
    for camera in os.listdir(os.path.join(FOLDER_PATH, facility)):
        for date in os.listdir(os.path.join(FOLDER_PATH, facility, camera)):
            for imageName in os.listdir(os.path.join(FOLDER_PATH, facility, camera,date)):
                # Insert folder name and image name into the database
                cursor.execute("INSERT INTO nVision.imagesToProcess (imageType,clientId,facility,cameraNo,date,imageName,cameraType,timeStamp,processed,svSource) VALUES (?,?,?,?,?,?,?,?,?,?)",
                               'thermal', clientId,facility,camera,date,imageName,'visual', datetime.now(), 0,2)
                conn.commit()

# Close database connection
conn.close()

print("Data insertion completed.")
       
  