from tkinter import Text,Button,simpledialog,messagebox
from PIL import ImageFont, ImageDraw, Image
from bidi.algorithm import get_display
from PIL import Image, ImageTk
import face_recognition
import arabic_reshaper
import tkinter as tk
import numpy as np
import cv2
import os
import glob
import sys


width, height = 600, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
root.title("برنامج التعرف على الوجوه")
root.resizable(False, False)  
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (width/2))
y_cordinate = int((screen_height/2) - (height/2))

root.geometry("{}x{}+{}+{}".format(width, height, x_cordinate, y_cordinate))

lmain = tk.Label(root)
lmain.pack(side=tk.TOP, anchor=tk.NW,padx=15,pady=15)




def captureImage():
    USER_INP = simpledialog.askstring(title="ادخال الاسم",
                                  prompt="ما هو اسمك؟",parent =root)
    if USER_INP is not None:
        if len(USER_INP)==0:
            messagebox.showerror("خطأ", "يرجى ادخال اسم")
        else:
        
            ret, frame = cap.read()
            img_name = "data/faces/"+USER_INP+".jpg"
            is_success, im_buf_arr = cv2.imencode(".jpg", frame)
            im_buf_arr.tofile(img_name)
            
        return True

def recognizeFace():
    root.destroy()
    #lmain.destroy()
    faces_encodings = []
    faces_names = []
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'data\\faces\\')
    list_of_files = [f for f in glob.glob(path+'*.jpg')]
    number_files = len(list_of_files)
    names = list_of_files.copy()


    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
        faces_encodings.append(globals()['image_encoding_{}'.format(i)])
    # Create array of known names
        x=names[i]
        names[i] = names[i].replace(path, "").removesuffix('.jpg')
        faces_names.append(names[i])

    face_locations = []
    face_encodings = []
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        name=""
        if process_this_frame:
            face_locations = face_recognition.face_locations( rgb_small_frame)
            face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
            face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces (faces_encodings, face_encoding)
            face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]
            face_names.append(name)
            process_this_frame = not process_this_frame
        if len(name)>0:
            text="مرحبا بك يا "+name
        else:
            text=u"غير معروف"
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text) 
        fontpath = "arial.ttf"   
        font = ImageFont.truetype(fontpath, 32)

        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((50, 80),bidi_text, font = font)
        img1 = np.array(img_pil)
        cv2.imshow('Face Recognition', img1) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.getWindowProperty('Face Recognition',cv2.WND_PROP_VISIBLE) < 1:   
            break 
    return True


def show_frame():
   
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)



show_frame()

btnCaptureImage = Button(root, text = "تخزين صورة",command = captureImage,width=20,height=5) 
btnCaptureImage.pack(pady=10)

btnRecognizeFace = Button(root, text = "اختبار ",command = recognizeFace,width=20,height=5) 
btnRecognizeFace.pack(pady=10)


root.mainloop()