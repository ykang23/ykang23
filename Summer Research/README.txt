Author: Yeeun Kang
Upload date: 09/17/2021 
Written: 05/2021 - 07/2021

Project Description
Project: Automatic Identification of Close-up Shots in Film
Project @ Roux Institute at Northeastern University in collaboration with Kinolab, Bowdoin College
Under the supervision of Professor Bruce Maxwell
Background: The project is aimed to develop an automatic detection and labelling of shot-types in film to help film studies at kino lab. 
Methods: Transfer Learning 
* Refer to the Presentation.pptx file for more details

Files: 
getframe.py : writes the frame and its annotations at the frame number specified by the user to the directory
frames_to_directory.py : opens the clip, runs it, when pressed "w", saves the image and annotations in jpg and csv file 
kinotool : a tool for viewing video and video annotations as part of the Kinolab project for detecting shot types.
analysisset: creates analysis set
validset: creates validation set
trainset: creates train set
testset: creates test set
