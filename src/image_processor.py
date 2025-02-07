from engine_utils import *
from datetime import datetime
import logging
from two_stage_model.dynamic_analysis import process_images_sequence
from engine_utils import get_overlap_percentage, get_overlap_use_contours, get_weights, get_inference_parameters

logging.basicConfig(level=logging.INFO)

def read_and_process(root_dir, server, database, username, password,process_no,def_conf_thresh,max_time_to_process,save_diff_map, save_visual_result, image_batch):
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';ENCRYPT=yes;UID='+username+';PWD='+ password +';TrustServerCertificate=yes')
    cursor = conn.cursor()
    input_h, input_w, ignored_areas = get_inference_parameters()
    overlap_percentage_threshold = get_overlap_percentage()
    overlap_use_contours = get_overlap_use_contours()
    weights=get_weights()
    def_conf_thresh=float(def_conf_thresh)
    max_time_to_process=float(max_time_to_process)
    logging.info(f'{str(len(image_batch))} images loaded on processor - {str(process_no)}')

    for i in range(0,len(image_batch)):

        tic=time.perf_counter()
        source=extract_image_path(image_batch,i,root_dir) #reconstruct path to image to be processed based on line extracted from imagesToProcess

        image_id=image_batch[i][0]
        client_id = image_batch[i][2]
        facility=image_batch[i][3]
        camera_no=image_batch[i][4]
        date=image_batch[i][5]
        image_name=image_batch[i][6]

        #find previous image
        query = "SELECT TOP 1 imageId,date, imageName FROM nVision.imagesToProcess WHERE facility='" + str(facility) + "' AND cameraNo=" + str(camera_no) + "AND imageName < '"+str(image_name) +"' order by imageName desc"
        cursor.execute(query)
        previous_source = cursor.fetchone()

        if previous_source is None:
            previous_source_path = source
        else:
            previous_source_path = root_dir + str(client_id) + '/' + str(facility) + '/' + str(camera_no) + '/' + str(previous_source[1]) + '/' + previous_source[2]

        #run inference using two_stage_model
        logging.info(f'ImageID - {str(image_id)} - starting processing image - {str(image_name)}')

        _, line_out=process_images_sequence([previous_source_path, source], show=False, use_model=weights, input_h=input_h, input_w=input_w, ignored_areas=ignored_areas,
                                            root_dir=root_dir, image_id=image_id, clientId=client_id, facility=facility, cameraNo=camera_no, date=date,
                                            overlap_percentage_threshold=overlap_percentage_threshold, save_diff_map=save_diff_map, save_visual_result=save_visual_result, overlap_use_contours=overlap_use_contours)

        detected_objects = []
        for j in range(0,len(line_out)):
            if line_out[j][0] != -1:  # objects detected
                classify=line_out[j][0] #obtain integer ID of object
                obj_class=get_object_class(classify) #convert integer ID of object to string literal (e.g., 0='leakage')

                #Extract object detection and inference info
                xc=line_out[j][1] #centroid x-location
                yc=line_out[j][2] #centroid y-location
                width=line_out[j][3] #bounding box width
                height=line_out[j][4] #bounding box height
                conf=line_out[j][5]#inference confidence
                ######################################################

                #append detected objects to list for writing to detectedObjects table
                if obj_class != 'Null' and obj_class is not None:
                    detected_objects.append([image_id,classify,conf,xc,yc,width,height,obj_class,'','',''])

        toc = time.perf_counter()  # stop counter for processing of image/object
        del_t = toc - tic
        time_exceeds = check_for_time(del_t, max_time_to_process, image_name)

        # log error
        if time_exceeds:
            cursor.execute(
                "INSERT INTO nVision.engineLogs (timeStamp, errorMessage, imageID, priority) VALUES (?,?,?,?)",
                [datetime.utcnow(), 'timeToProcess exceeds ' + str(max_time_to_process) + ' seconds', image_id, 'M'])
            conn.commit()

        # process results of inference
        result_handled = 0  # default
        if len(detected_objects) != 0:
            # if all detected bbox are under the mask, we set result to 0 otherwise 1
            result = int(not all(row[1] == 0 for row in detected_objects))
            result_type = 'leakage' if result == 1 else 'suppressed event'
        else:
            # if no bbox detected, we set result to 0
            result = 0
            result_type = 'non-relevant event'

        # insert inference results to result table
        logging.info(f'ImageID - {str(image_id)} - starting inserting result for image - {image_name}')
        cursor.execute(
            "INSERT INTO nVision.result (imageId, resultType, timeToProcess, result, resultHandled, timeStamp) OUTPUT INSERTED.resultId VALUES (?,?,?,?,?,?)",
            [image_id, result_type, del_t, result, result_handled, datetime.utcnow()])
        result_id = cursor.fetchone()[0]
        conn.commit()
        logging.info(f'ImageID - {str(image_id)} - result of image - {str(image_name)} is: {str(result)} on processor - {str(process_no)}')

        # insert into detectedObjects table and commit
        if len(detected_objects) != 0:
            detected_objects = [row[:1] + [str(result_id)] + row[2:] + [datetime.utcnow()] for row in detected_objects]
            cursor.executemany("""
                    INSERT INTO nVision.detectedObjects (imageId, resultId, confidenceLevel,boundingBox_xc,boundingBox_yc,
                    boundingBoxWidth,boundingBoxHeight,objectClass,previousImageId,previousObjectId,persist,timeStamp) 
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                               detected_objects)
            conn.commit()

        # update image to processed
        cursor.execute("UPDATE nVision.imagesToProcess SET processed=1 WHERE imageId=" + str(image_id))
        conn.commit()
        logging.info(f'ImageID - {str(image_id)} - image - {str(image_name)} has been processed on processor - {str(process_no)}')



