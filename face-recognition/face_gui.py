from tkinter import * 
import cv2
import os
from matplotlib import pyplot as plt
import cv2 as cv

class run_gui:
    def capture_face(self):
        os.system('python3 ./create_dataset.py')

    def train_face(self):
        os.system('./tasks/train.sh  ./images/')

    def verify_face(self):
        os.system('python3 verify_face.py')
    
    def run(self):
        root = Tk()

        root.geometry('1000x1000')

        create_dataset=Button(root, text="Create your face dataset",fg="red",command=self.capture_face,height=5,width=20).place(x=450,y=250)
        train_dataset=Button(root, text="Train your face dataset",fg="red",command=self.train_face,height=5,width=20).place(x=450,y=350)
        test_dataset=Button(root, text="Verify your face",fg="red",command=self.verify_face,height=5,width=20).place(x=450,y=450)
        root.configure(bg='black')

        root.mainloop()

if __name__ == '__main__':
    run_gui().run()