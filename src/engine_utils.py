import ast
from configparser import ConfigParser
from datetime import datetime
import logging
import pyodbc
import time

logging.basicConfig(level=logging.INFO)

config_object = ConfigParser()
config_object.read("config.ini")

class NoNewEntries(Exception):
    pass

# Read config.ini file for SQL server/DB info
def get_config():
    server_info = config_object["SERVERCONFIG"]

    server = server_info["server"]
    database = server_info["database"]
    username = server_info["username"]
    password = server_info["password"]
    query = server_info["query"]

    return (server, database, username, password, query)


# Read config.ini file for image processing parameters
def get_engine_param():
    engine_info = config_object["ENGINEPARAMETERS"]
    batch_size = engine_info["batch_size"]
    n_proc = engine_info["n_proc"]
    root_dir = engine_info["root_dir"]
    conf = engine_info["default_conf_thresh"]
    max_time_to_process = engine_info["max_time_to_process"]
    wait_time = engine_info["wait_time"]
    save_diff_map = engine_info["save_diff_map"]
    save_visual_result = engine_info["save_visual_result"]
    log_file_name = engine_info["log_file_name"]

    return (int(batch_size), int(n_proc), root_dir, float(conf), float(max_time_to_process), int(wait_time), bool(eval(save_diff_map)), bool(eval(save_visual_result)), log_file_name)


# Read config.ini file for classes info
def get_classes():
    class_info = config_object["CLASSES"]
    class_dict = eval(class_info["class_dict"])

    return (class_dict)

def get_device():
    engine_info = config_object["ENGINEPARAMETERS"]
    device = engine_info["device"]

    return(device)

def get_weights():
    engine_info = config_object["ENGINEPARAMETERS"]
    weights = engine_info["weights"]

    return(weights)

def get_overlap_percentage():
    engine_info = config_object["ENGINEPARAMETERS"]
    overlap_percentage = engine_info["overlap_percentage"]

    return(overlap_percentage)

def get_overlap_use_contours():
    engine_info = config_object["ENGINEPARAMETERS"]
    overlap_use_contours = engine_info["overlap_use_contours"]

    return(overlap_use_contours)


def get_inference_parameters():
    inference_parameters = config_object["INFERENCEPARAMETERS"]
    input_h = inference_parameters.getint('input_height')
    input_w = inference_parameters.getint('input_width')
    ignored_areas = inference_parameters['ignored_areas']
    ignored_areas = ast.literal_eval('[' + ignored_areas + ']')

    return input_h, input_w, ignored_areas


def extract_image_path(image_batch, i, root_dir):
    source = root_dir + (str(image_batch[i][2])).zfill(3) + '/' + image_batch[i][3] + '/' + str(
        image_batch[i][4]) + '/' + image_batch[i][5] + '/' + image_batch[i][6]

    return source


def check_presence(classes_det):
    class_dict = get_classes()
    classes = list(class_dict.keys())
    if any(class_det in classes_det for class_det in classes):
        return 1
    else:
        return 0


def get_object_class(classify):
    class_dict = get_classes()
    return class_dict.get(classify)


def check_for_time(del_t, max_time_to_process, image_name):
    if del_t > max_time_to_process:
        string = f'timeToProcess exceeds {str(max_time_to_process)} seconds when processing image - {str(image_name)}'
        logging.warning(string)
        return True
    else:
        return False


def check_for_exceptions(server, database, username, password, wait_time):
    retries = 0
    retry_limit = 10

    # ENCRYPT defaults to yes starting in ODBC Driver 18. It's good to always specify ENCRYPT=yes on the client side to avoid MITM attacks.
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';ENCRYPT=yes;UID=' + username + ';PWD=' + password + ';TrustServerCertificate=yes')
    cursor = conn.cursor()

    while True:
        try:
            query = "SELECT COUNT(*) FROM nVision.imagesToProcess WHERE processed=0"
            cursor.execute(query)
            no_unprocessed = cursor.fetchone()
            no_unprocessed = int(no_unprocessed[0])

            if no_unprocessed == 0:
                raise NoNewEntries

        except NoNewEntries:
            if retries == retry_limit:
                err_msg = f'No new unprocessed images in imagesToProcess table for {str(wait_time * retries / 60)} minutes. Restart service.'
                logging.error(err_msg)

                cursor.execute("""
                        INSERT INTO nVision.engineLogs (timeStamp, errorMessage, imageID, priority) 
                        VALUES (?,?,?,?)""", datetime.now(), err_msg, 00, 'H')
                conn.commit()
                retries = 0

            else:
                err_msg = f'No new unprocessed images in imagesToProcess table. No. of retries: {str(retries)} ... waiting for {str(wait_time)} seconds'
                logging.warning(err_msg)
                time.sleep(wait_time)
                retries += 1

        else:
            string = 'All systems nominal'
            logging.info(string)
            cursor.close()
            conn.close()
            break
