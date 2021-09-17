import numpy as np
import pandas as pd
import random
import cv2
import sys
# gets 2 sample frames from each close ups from all clips 

clip_lengths = {'1316':5548, '1382':7053, '1391':11689, '206':12533, '1370':9341 }
def manual_frame_count( cap, clip_id ):

    if clip_id in clip_lengths:
        return clip_lengths[clip_id]
    
    frames = 0;

    while True:
        ref, frame = cap.read()

        if not ref:
            break

        frames += 1

    clip_lengths[clip_id] = frames

    return frames

# picks close up frame
def pick_close_ups(start, end, i):
    if start[i] == end[i]:
        f = start[i]
    elif start[i] < end[i]:
        f = random.randint(start[i], end[i])
    else: # bail because there aren't any good choices
        f = -1
    return f

def main(argv):
    if len(argv) < 3:
        print("Usage: python3 %s <src data> <output dir>" % (argv[0]))
        return
    
    data_dir = argv[1]
    analysis_dir = argv[2]

    #read in the annotations 
    df = pd.read_csv("kinotool/master_annotations.csv")
    annotations  = df.to_numpy()
    annotations = annotations[:,1:]
    clip_titles = annotations[:,0]
    titles = annotations[:,6]
    code1s = annotations[:,12]
    start = annotations[:,18]
    end = annotations[:,19]
    clip_ids = annotations[:,4]
    fps_vals = annotations[:,20]
    start_f = np.asarray(start)
    end_f = np.asarray(end)
    ids = {}
    
    # make a dictionary of clip ids which are not used in the test set
    for id in clip_ids:
        if id not in ids:
            ids[id] = 0
        else:
            ids[id] += 1
    print("number of movie clips for the train set: ", len(ids))
    

    # convert the dictionary into list
    clip_list = [(k,v) for k,v in ids.items()]
    total_cus = 0
    
    # get 2000 close-ups
    close_up_frames = []
    clip_id = 0

    how_many = []

    cu_anlysis_dir = analysis_dir + "/CU"
    for clip in clip_list:
        if clip[0] == '206':
            print("We aren't using Pan's Labyrinth")
            clip_id += 1
            continue
        
        if clip[0] not in how_many:
            how_many.append(clip[0])
        
        # gets clip indices in annotation file
        where = np.where(clip_ids == clip[0])
        if len(where[0]) == 0:
            print("CANT FIND INDEX FOR", clip[0])
            continue
        indices = where[0]
        indices = indices.astype(int)
        
        start_c = start_f[indices]
        
        end_c = end_f[indices]
        clip_title = clip_titles[indices[0]]
        clip_title = clip_title.replace('"',"")
        film_title = titles[indices[0]]
        code_1 = code1s[indices[0]]
        fps = fps_vals[indices[0]]
        
        if code_1 == "":
            code_1 = "Unclassified"     
        
        filename = data_dir+ "/" + str(clip[0]) + ".mp4"
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print("File is not opened!") 
            continue

        # loops from first close ups to last close ups
        # picks close-up frame and write the image and annotations
        for x in range(2):
            for i in range(0, len(start_c)):
                f = pick_close_ups(start_c, end_c, i)
                if f < 0: # can't pick a good frame from this close-up example
                    continue

                if not f in close_up_frames:
                    # double-check if it can successfully move to that frame #
                    success = cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    if not success:
                        continue

                    actual_next_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if actual_next_frame != f:
                        print("unable to get frame f")
                        continue

                    ret, frame = cap.read()
                    if not ret: # couldn't grab this image
                        continue

                    img_name = "%s/%s_%06d.jpg"%(cu_anlysis_dir,clip[0],f)
                    cv2.imwrite(img_name, frame)
                    information = np.array([film_title,clip_title,clip[0], f ,img_name, code_1])
                    information = information.reshape((1,information.shape[0]))
                    if len(close_up_frames) == 0:
                        close_up_info =  information
                    else:
                        close_up_info = np.vstack((close_up_info, information))
                    close_up_frames.append(f)
        cap.release()
        
    print("Wrote frames from", len(how_many), "clips")

    # save the test annotations in csv file to the directory
    np.savetxt(analysis_dir+"/"+"analysis_set_annotations.csv", close_up_info, delimiter = ",", header ="Film Title,Clip Title, Clip ID,Frame Number,Image Name,Close-up code I",fmt=['%s','%s','%s','%s','%s','%s'],comments = '')
        
    
if __name__ == '__main__':
    main(sys.argv)
