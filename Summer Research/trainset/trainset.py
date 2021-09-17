import numpy as np
import pandas as pd
import random
import cv2
import sys
# gets a total of 3000 test frames- 1500 close-ups and 1500 non close-ups

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
    #elif start[i]+24 < end[i]-24:
        #f = random.randint(start[i]+24, end[i]-24)
    elif start[i] < end[i]:
        f = random.randint(start[i], end[i])
    else: # bail because there aren't any good choices
        f = -1
    return f

# picks non close up frames
# start is a list of closeup start frames in the clip
# end is a list of closeup end frames in the clip
# i is the index of a closeup
# total_frames is the total number of frames in the clip
def pick_non_close_ups(start, end, i, total_frames,fps):
    if i == 0 and 0 < start[i]-fps: # start section
        f = random.randint(0, start[i]-fps) 
    elif i == len(end)-1 and end[i]+fps < total_frames: # end section
        f = random.randint(end[i]+fps, total_frames-1)
    elif (len(start) > i+1) and (end[i]+fps < start[i+1]-fps): # more than 2 spaces
        f = random.randint(end[i]+fps, start[i+1]-fps)
    else: # bail because there aren't any good choices
        f = -1
    if f >= total_frames:
        print("BAD BAD BADDDD", i, total_frames, f)
        f = -1
    return f

def main(argv):
    set_size = 1500
    if len(argv) < 3:
        print("Usage: python3 %s <src data> <output dir>" % (argv[0]))
        return
    
    data_dir = argv[1]
    train_dir = argv[2]

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
    ids = {}

    #read in the test set annotations 
    testset = pd.read_csv("testset/test_set_annotations.csv")
    testset_stuffs = testset.to_numpy()
    t_film_titles = testset_stuffs[:,0]
    t_clip_titles = testset_stuffs[:,1]
    t_clip_ids = testset_stuffs[:,2]
    t_frame_no = testset_stuffs[:,3]
    t_img_name = testset_stuffs[:,4]
    t_code1s = testset_stuffs[:,5]
    t_ids = {}
    start_f = np.asarray(start)
    end_f = np.asarray(end)

    # make a dictionary of clip ids used in the test set
    for id in t_clip_ids:
        if id not in t_ids:
            t_ids[id] = 0
        else:
            t_ids[id] += 1
    print("number of movie clips used in the test set:  ", len(t_ids))

    
    # make a dictionary of clip ids which are not used in the test set
    for id in clip_ids:
        if not id in t_ids:
            if id not in ids:
                ids[id] = 0
            else:
                ids[id] += 1
        else:
            continue
    print("number of movie clips for the train set: ", len(ids))
    

    # convert the dictionary into list
    clip_list = [(k,v) for k,v in ids.items()]
    random_clips = []
    total_cus = 0
    num_clips = 40
    
    # get 1500 close-ups
    close_up_frames = []
    clip_id = 0

    how_many = []

    cu_train_dir = train_dir + "/CU"
    while len(close_up_frames) < set_size:
        # gets all clips and close-ups for the first round
        if clip_id < len(clip_list):
            clip = clip_list[clip_id]
            if clip[0] == '206':
                print("We aren't using Pan's Labyrinth")
                clip_id += 1
                continue
            
            clip_id += 1
        # randomly selects clips and close-ups
        else:
            clip = random.choice(clip_list)

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

        # go through all close ups 
        if clip_id < len(clip_list):
            first_cu = 0
            last_cu = len(start_c)

        # randomly selects close up
        else:
            first_cu = random.randint(0, len(start_c)-1)
            last_cu = first_cu+1        
        
        filename = data_dir+ "/" + str(clip[0]) + ".mp4"
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print("File is not opened!") 
            continue

        # loops from first close ups to last close ups
        # picks close-up frame and write the image and annotations 
        for i in range(first_cu, last_cu):
            f = pick_close_ups(start_c, end_c, i)
            if f < 0: # can't pick a good frame from this close-up example
                continue
            
            if f not in close_up_frames:
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

                img_name = "%s/%s_%06d.jpg"%(cu_train_dir,clip[0],f)
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
    #exit()
    # get 1500 non close_ups
    frame_num = 0
    non_close_up_frames = []
    clip_id = 0
    non_cu_train_dir = train_dir + "/non-CU"
    while len(non_close_up_frames) < set_size:
        # gets all clips and non close-ups for the first round
        if clip_id < len(clip_list):
            clip = clip_list[clip_id]
            clip_id += 1
        # randomly selects clips and non close-ups
        else:
            clip = random.choice(clip_list)
            
        # gets clip indices in annotation file
        where = np.where(clip_ids == clip[0])
        if len(where[0]) == 0:
            continue
        indices = where[0]
        indices = indices.astype(int)
        
        start_c = start_f[indices]
        end_c = end_f[indices]
        
        clip_title = clip_titles[indices[0]]
        clip_title = clip_title.replace('"',"")
        film_title = titles[indices[0]]
        code_1 = "N/A"

        first_cu = 0
        last_cu = len(start_c)
            
        filename = data_dir + "/"+ str(clip[0]) + ".mp4"
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            continue
            print("File is not opened!") 

        # loops from first close ups to last close ups
        # picks non close-up frame and write the image and annotations
        total_frames = manual_frame_count( cap, clip[0] ) # int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i = random.randint(first_cu, last_cu - 1)
        f = pick_non_close_ups(start_c, end_c, i, total_frames, fps)
        #print( "picked %s total frames %d clip %d frame %d" % (clip[0], total_frames, i, f ) )
        if f < 0:
            continue

        if f not in non_close_up_frames: 
            success = cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            if not success:
                continue
            
            actual_next_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if actual_next_frame != f:
                print('Unable to get frame f')
                continue
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            img_name = "%s/%s_%06d.jpg"%(non_cu_train_dir,clip[0],f)
            cv2.imwrite(img_name, frame)
            information = np.array([film_title,clip_title,clip[0], f ,img_name, code_1])
            information = information.reshape((1,information.shape[0]))
            if len(non_close_up_frames) == 0:
                non_close_up_info = information
            else:
                non_close_up_info = np.vstack((non_close_up_info, information))
            non_close_up_frames.append(f)

        # see if releasing the video helps
        cap.release()

    # save the test annotations in csv file to the directory
    info = np.vstack((close_up_info, non_close_up_info))
    np.savetxt(train_dir+"/"+"train_set_annotations.csv", info, delimiter = ",", header ="Film Title,Clip Title, Clip ID,Frame Number,Image Name,Close-up code I",fmt=['%s','%s','%s','%s','%s','%s'],comments = '')
        
    
if __name__ == '__main__':
    main(sys.argv)
