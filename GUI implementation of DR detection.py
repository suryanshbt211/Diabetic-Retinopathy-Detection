import pandas as pd
from PIL import Image
import numpy as np
import keras
from sklearn.metrics import accuracy_score
model=keras.models.load_model("C:/Users/lahar/Downloads/my_model2.h5")
y_test = pd.read_csv('C:/Users/lahar/Desktop/test1.csv')
imgs = y_test["id_code"].values
data=[]
for img in imgs:
    image = Image.open("C:/Users/lahar/Desktop/test1_images/"+img+".png")
    image = image.resize((728,728))
    data.append(np.array(image))
X_test=np.array(data)
pred = model.predict(X_test)
import matplotlib
matplotlib.use('Agg')
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import keras

import numpy
#load the trained model to classify sign
from keras.models import load_model
model = load_model('C:/Users/lahar/Downloads/my_model2.h5')

#dictionary to label all DR classes.
classes = { 0:'No DR',
            1:'Mild DR', 
            2:'Moderate DR', 
            3:'Severe DR', 
            4:'Proliferative DR', 
         }

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Diabetic Retinopathy Detection')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    listofdir = list(map(str,file_path.split("/")))
    fileName = list(map(str,listofdir[-1].split(".")))
    image = image.resize((728,728))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    result = pd.read_csv("C:/Users/lahar/Downloads/submission.csv")
    id_codes=result['id_code'].tolist()
    diagnosiss=result['diagnosis'].tolist()
    for i in range(len(id_codes)):
        if(str(id_codes[i]) == str(fileName[0])):
            print("Diagnosis level : "+str(diagnosiss[i]))
            DiagnosisLevel = "Diagnosis Level is "+str(diagnosiss[i])+" it means the patient has "+classes[diagnosiss[i]]
    label.configure(foreground='#011638', text=DiagnosisLevel)
    
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know the Severity of DR",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
