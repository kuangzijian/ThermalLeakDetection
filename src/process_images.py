import os
from logging.handlers import TimedRotatingFileHandler
from engine_utils import *
import multiprocessing
from multiprocessing import Process
import logging
import image_processor
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension", category=UserWarning, module="torchvision.io.image")
warnings.filterwarnings("ignore", message="Unable to retrieve source for @torch.jit._overload function:", category=UserWarning, module="torch._jit_internal")
os.environ["PYTORCH_JIT"] = "0"

# Import engine parameters from config file
batch_size, n_proc, root_dir, def_conf_thresh, max_time_to_process, wait_time, save_diff_map, save_visual_result, log_file_name = get_engine_param()

# Ensure log directory exists
log_directory = "./logs"
os.makedirs(log_directory, exist_ok=True)

# Get today's log file name
log_file_name = os.path.join(log_directory, "app.log")

# Set up the TimedRotatingFileHandler
handler = TimedRotatingFileHandler(log_file_name, when="midnight", interval=1, backupCount=7)
handler.suffix = "%Y-%m-%d"  # Suffix for rotated log files

# Create a custom formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Import MSSQL server parameters from config file
    server, database, username, password, query = get_config()

    try:
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';ENCRYPT=yes;UID=' + username + ';PWD=' + password + ';TrustServerCertificate=yes')
        cursor = conn.cursor()
        logging.info('Connection successful')
    except pyodbc.Error as e:
        logging.info('Connection failed:', e)

    while True:
        try:
            # CHECK FOR EXCEPTIONS (e.g., empty table, no new unprocessed images,etc.) AND PERFORM EXCEPTION HANDLING (e.g., wait, exit), IF NECESSARY
            check_for_exceptions(server, database, username, password, wait_time)

            # GET IMAGE BATCHES
            top_n = batch_size * n_proc  # set the number of rows to select
            select_query = query.format(top_n)
            with cursor.execute(select_query) as cursor:
                images = cursor.fetchall()

            # INITIALIZE MULTIPROCESSING
            procs = []

            if len(images) >= n_proc:
                chunk_size = len(images) // n_proc
                image_batch = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]

                # SEND IMAGES TO SUB-PROCESSES (each sub-process has its own instance of image_processor.py)
                for i in range(n_proc):
                    proc = Process(target=image_processor.read_and_process, args=(
                        root_dir, server, database, username, password, str(i), str(def_conf_thresh),
                        str(max_time_to_process), save_diff_map, save_visual_result, image_batch[i]))
                    procs.append(proc)
                    proc.start()
            else:
                proc = Process(target=image_processor.read_and_process, args=(
                    root_dir, server, database, username, password, str(0), str(def_conf_thresh),
                    str(max_time_to_process), save_diff_map, save_visual_result, images))
                procs.append(proc)
                proc.start()

            # Run the processes
            for proc in procs:
                proc.join()

        except pyodbc.Error as e:
            logging.info('Connection failed:', e)













