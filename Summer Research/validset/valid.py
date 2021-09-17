'''randomly picked 500 '''
import os 
import random 

def main():
    # gets to the trainset directory 
    path = '/Volumes/Personal/ykang23/Summer/trainset'
    CU_dir = os.path.join(path, "CU")
    non_CU_dir = os.path.join(path, "non-CU")
    CU_files = os.listdir(CU_dir)
    non_CU_files = os.listdir(non_CU_dir)
    #print(type(CU_files))

    set_size = 250
    # randomly picks files from CU directory 
    random_CUs = random.sample(CU_files, set_size)

    # randomly picks files from non_CU directory
    random_non_CUs = random.sample(non_CU_files, set_size)

    # move selected files to validset
    for file in random_CUs:
        #print(file)
        cmd = "mv " + CU_dir + "/"+ file + " /Volumes/Personal/ykang23/Summer/validset/CU" 
        os.system(cmd)
    for file in random_non_CUs:
        #print(file)
        cmd = "mv " + non_CU_dir +"/"+ file + " /Volumes/Personal/ykang23/Summer/validset/non-CU"
        os.system(cmd)

if __name__ == '__main__':
    main()