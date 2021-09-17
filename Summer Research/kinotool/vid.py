import numpy as np
import pandas as pd
import cv2 as cv
import sys
import argparse
import os

# space bar = pause
# right key = skip forward
# left key = skip backward
# f = play forward
# b = play backward
# i = info
# q = exit 
clip_lengths = {}

def manual_frame_count( cap, clip_id ):

    if clip_id in clip_lengths:
        return clip_lengths[clip_id]
    
    frames = 0;

    while True:
        ref, frame = cap.read()

        if not ref:
            break

        frames += 1

    current = cap.get(cv.CAP_PROP_POS_FRAMES)
    print("current %d frames %d" % (current, frames) )
    clip_lengths[clip_id] = frames
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)


    return frames

def time_to_frame(time, frames_per_sec):
    #print(time)
    if len(time) == 4:
        seconds = ((int)(time[0]))*60 + ((int)(time[2]))*10 + ((int)(time[3]))
        frames = seconds*frames_per_sec
    else:
        seconds = ( (((int)(time[0]))*10) + ((int)(time[1])) )* 60 + ((int)(time[3]))*10 + ((int)(time[4]))
        frames = seconds*frames_per_sec    
    return frames

def main(): 
    parser = argparse.ArgumentParser(description="List video and annotation file")
    parser.add_argument("video",help="List video file you want to play")
    parser.add_argument("annotations",help="List annotation file")
    parser.add_argument("-s",nargs = '?', help="Write frame per second. Default = 30", const = 30)

    args = parser.parse_args()
    print("Using video      : ", args.video)
    print("Using annotations: ", args.annotations)
    
    cap = cv.VideoCapture(args.video)

    file_id = os.path.basename(args.video)
    file_id = os.path.splitext(file_id)[0]

    frames_per_sec = 30
    if args.s != None:
        frames_per_sec = int(args.s)
    print("Using %d frames per second" % (frames_per_sec))
        
    #read in the annotations 
    df = pd.read_csv(args.annotations)
    annotations  = df.to_numpy()
    annotations = annotations[:,1:]
    clip_ids = annotations[:,4]
    x = np.where(clip_ids == int(file_id) ) 
    indices = x[0]
    idx = x[0][0]
    clip_title = annotations[idx,0]
    title = annotations[idx,6]
    code1 = [] 
    start = []
    end = []

    for idx in indices:
        code1.append(annotations[idx,12])
        start.append(annotations[idx,18])
        end.append(annotations[idx,19])


    #for i in range(len(start)):
	#start[i] = int(start[i])
	#end[i] = int(end[i])
    #print(start[1])
    print("film title: ", title)
    print("clip title: ",clip_title)
    print("close-up code 1: ",code1 )
    print("start time: ",start )
    print("end time", end)

    if not cap.isOpened():
        print("File is not opened!") 
        exit()
    current = cap.get(cv.CAP_PROP_POS_FRAMES)
    flag = True
    reverse = False
    refresh = False
    wait_divisor = 1

    total_frames = manual_frame_count( cap, '1316' )
    print("total_frames %d", total_frames)
    
    while cap.isOpened():
        key = cv.waitKey(int(1000/frames_per_sec) // wait_divisor) # adjust speed to fps
        #if key != 255:
            #print(key)

        # handle the case of not being paused
        if flag == True:

            # if moving backward
                # set current to current - 1 (or current -2)
                # use cap.set to set the frame to the prior one
            if reverse:
                if current > 0:
                    current = current - 2
                cap.set(cv.CAP_PROP_POS_FRAMES, current)

            current = cap.get(cv.CAP_PROP_POS_FRAMES)

            try:
                oldret, oldframe = ret, frame
            except:
                pass
                
            ret, frame = cap.read()
            
            if not ret:
                ret, frame = oldret, oldframe
                current -= 1
                
        else: # paused

            if refresh:
                current = cap.get(cv.CAP_PROP_POS_FRAMES)

                try:
                    oldret, oldframe = ret, frame
                except:
                    pass
                    
                ret, frame = cap.read()

            if refresh and not ret:
                ret, frame = oldret, oldframe
                current -= 1
            
            refresh = False
            
        if key == ord('q'): # quit the program
            break
        
        elif key == ord(' '): # pause / unpause the clip
            flag = not(flag)

        elif key == ord('f'): # play the film forwards
            reverse = False

        elif key == ord('j'): # jump to the start of the next closeup
            count = 0
            while start[count] < current:
                if count >= len(start)-1:
                    break
                count+=1   
            cap.set(cv.CAP_PROP_POS_FRAMES, start[count])
            refresh = True

        elif key == ord('k'): # jump to the start of the prior closeup
            count = 0
            while count < len(end) and end[count] < current:
                count += 1
            if count == 0:
                cap.set(cv.CAP_PROP_POS_FRAMES, start[count])
            else:
                cap.set(cv.CAP_PROP_POS_FRAMES, start[count-1])
            refresh = True
                

        elif key == 81: # move backward one frame
            cap.set(cv.CAP_PROP_POS_FRAMES, current - 1)
            refresh = True

        elif key == 82: # move to the start of the clip
            wait_divisor = 2
            
        elif key == 83: # move forward one frame
            refresh = True

        elif key == 84: # play at normals peed
            wait_divisor = 1
            
        elif key == ord('b'): # play the file backwards
            reverse = True

        elif key == ord('a'): # go to the start of the file
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            refresh = True
            

        elif key == ord('i'): # show the clip info window
            cv.namedWindow('Clip Info', cv.WINDOW_AUTOSIZE)
            # Create a black image
            img = np.ones((250,250,3), np.uint8)

            # Write some Text

            font                   = cv.FONT_HERSHEY_SIMPLEX
            org                    = (0,50)
            org2                   = (0,100)
            fontScale              = 0.5
            fontColor              = (255,255,255)
            lineType               = 1
            time_min = int(current) // int(frames_per_sec)
            time_sec = int(current) % int(frames_per_sec)
            
            txt = 'Current Time / Frame: '+ str(time_min) + " : " + str(time_sec) + "  " + str(current)
            txt2 = 'Video Name: ' + args.video
            cv.putText(img,txt, 
                        org, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
            cv.putText(img,txt2, 
                        org2, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)

            #Display the image
            cv.imshow('Clip Info', img)

        elif key == ord('c'): # remove the clip info window
            cv.destroyWindow('Clip Info')

        # end of handling key strokes

        # put frame number at the top 
        font                   = cv.FONT_HERSHEY_SIMPLEX
        org                    = (50,55)
        fontScale              = 0.5
        fontColor              = (255,255,255)
        lineType               = 1
        time_min = int(current) // int(60 * frames_per_sec)
        time_sec = (int(current) // int(frames_per_sec)) % 60
            
        txt = "Current Time / Frame: %3d:%02d / %5d" % (time_min, time_sec, current )
        cv.putText(frame, txt, 
                   org, 
                   font, 
                   fontScale,
                   fontColor,
                   lineType)
            
        # manage annotations for the current frame
        center = (50,50)
        radius = 20
        color = (0,255,0)
        thickness = 3    
        font                   = cv.FONT_HERSHEY_SIMPLEX
        org                    = (90,55)
        fontScale              = 0.5
        fontColor              = (0,255,0)
        lineType               = 2

        # check if any annotation is active right now
        for i in range( len( start ) ):
            # checks the case where start < end time
            if (current >= int(start[i])+1) and (current < int(end[i])+1):
                circle = cv.circle(frame, center, radius,color, thickness )
                try:
                    cv.putText(frame,code1[i], 
                               org, 
                               font, 
                               fontScale,
                               fontColor,
                               lineType)
                except:
                    cv.putText(frame,"Unclassified", 
                               org, 
                               font, 
                               fontScale,
                               fontColor,
                               lineType)

                #print(code1[i])
            # checks the case where start == end time
            if start[i] == int(end[i])+1 and current >= int(start[i])+1 and current < int(start[i]) +2:
                circle = cv.circle(frame, center, radius,color, thickness )
                cv.putText(frame,code1[i], 
                           org, 
                           font, 
                           fontScale,
                           fontColor,
                           lineType)

        
        # show the current frame
        cv.imshow('frame', frame)

        # gets the current frame #
        #current = cap.get(cv.CAP_PROP_POS_FRAMES)
        
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
