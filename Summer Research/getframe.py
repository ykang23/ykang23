import numpy as np
import pandas as pd
import cv2 as cv
import argparse
import os
# writes the frame and its annotations at the frame number specified by the user to the directory 

# converts time to frame
def time_to_frame(time, frames_per_sec):
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
    parser.add_argument("-t", help = "Enter the frame you want to save")
    parser.add_argument("-a", help = "Enter the annotation file")
    
    args = parser.parse_args()
    cap = cv.VideoCapture(args.f)
    if not cap.isOpened():
        print("File is not opened!") 
        exit()
    file_id = os.path.splitext(args.f)[0]

    # sets the frame
    cap.set(cv.CAP_PROP_POS_FRAMES,int(args.t))
    ret, frame = cap.read()

    # writes the image
    img_name = "%s_%06d.jpg"%(file_id,int(args.t))
    cv.imwrite(img_name, frame)
    print(img_name, "written to a directory...")

    # writes the annotations 
    csv_name = "%s_%06d.csv"%(file_id,int(args.t))
 
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

    for i in range(len(start)):
        if start[i]>=int(args.t) and int(args.t) <= end[i]:
            code = code1[i]
        else:
            code = 'N/A'
            
    information = np.array([title,str(clip_title),int(args.t) ,code])
    information = information.reshape((1,information.shape[0]))
    np.savetxt(csv_name, information, delimiter = ",", header ="Film Title,Clip Title,Frame Number,Close-up code I",fmt=['%s','%s','%s','%s'],comments = '')
    print(csv_name, "written to a directory...")
        
    cap.release()
    cv.destroyAllWindows()
    

if __name__ == '__main__':
    main()
