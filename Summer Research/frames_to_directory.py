import numpy as np
import pandas as pd
import cv2 as cv
import sys
import argparse
import os
# opens the clip, runs it, when pressed "w", saves the image and annotations in jpg and csv file 

# converts time to frame
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
    parser.add_argument("-f",help="List video file you want to play")
    parser.add_argument("-a", help = "List annotation file")
    
    args = parser.parse_args()
    print("Video file entered: ", args.f)
    print("Annotatin file: ", args.a)
    
    cap = cv.VideoCapture(args.f)
    file_id = os.path.splitext(args.f)[0]

    # reads in the annotations 
    df = pd.read_csv(args.a)
    annotations  = df.to_numpy()
    annotations = annotations[:,1:]
    clip_ids = annotations[:,4]
    x = np.where(clip_ids == ( (int)(file_id)) )
    indices = x[0]
    idx = x[0][0]
    clip_title = annotations[idx,0]
    title = annotations[idx,6]
    code1 = []
    start = []
    end = []
    for idx in indices:
        code1.append(annotations[idx,12])
        start.append(time_to_frame(annotations[idx,9],30))
        end.append(time_to_frame(annotations[idx,10],30))

    if not cap.isOpened():
        print("File is not opened!") 
        exit()
    
    current = cap.get(cv.CAP_PROP_POS_FRAMES)
    flag = True
    reverse = False
    while cap.isOpened():
        key = cv.waitKey(25)

        # handle the backward case
        if flag == True:
            if reverse:
                if current > 0:
                    current = current - 2
                cap.set(cv.CAP_PROP_POS_FRAMES, current)

            ret, frame = cap.read()
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
        if key == ord('q'):
            break
        
        elif key == ord(' '):
            flag = not(flag)

        elif key == ord('f'):
            reverse = False

        elif key == 81:
            cap.set(cv.CAP_PROP_POS_FRAMES, current - 50)
            
        elif key == 83:
            cap.set(cv.CAP_PROP_POS_FRAMES, current + 50)    
            
        elif key == ord('b'):
            reverse = True

        # writes the frame and the annotations in jpg and csv file
        elif key == ord('w'):
            
            # image 
            img_name = "%s_%06d.jpg"%(file_id,int(current))
            cv.imwrite(img_name, frame)
            print(img_name, "written to a directory...")

            # annotations 
            for i in range(len(start)):
                if current >= start[i] and current <= end[i]:
                    code = code1[i]
                else:
                    code = 'N/A'

            clip_title = clip_title.replace('"',"")
            information = np.array([title,str(clip_title),str(int(current)),code])
            information = information.reshape((1,information.shape[0]))
            csv_name = "%s_%06d.csv"%(file_id,int(current))
            np.savetxt(csv_name, information, delimiter = ",", header ="Film Title,Clip Title,Frame Number,Close-up code I",fmt=['%s','%s','%s','%s'],comments = '')
            print(csv_name, "written to a directory...")

        elif key == ord('i'):
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
            txt = 'Current Time Frame: '+ str(current)
            txt2 = 'Video Name: ' + args.f
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

        elif key == ord('c'):
            cv.destroyWindow('Clip Info')
        
        # show the current frame
        cv.imshow('frame', frame)

        # gets the current frame #
        current = cap.get(cv.CAP_PROP_POS_FRAMES)
        
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
