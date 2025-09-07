import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

file_path=""
Resnet = load_model('resnet50v2_adam_model.keras')  

predicted_label = ""
# Initialize CustomTkinter
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"

# Function to upload image and display it
def upload_image():
    global file_path 
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.jpeg;*.png")])
    if file_path:
        print(f"Image uploaded: {file_path}")
        
        # Load the image
        uploaded_image = Image.open(file_path)
        resized_image = uploaded_image.resize((200, 200), Image.Resampling.LANCZOS)

        # Convert image to PhotoImage for tkinter
        uploaded_image_tk = ImageTk.PhotoImage(resized_image)

        # Create or update the image label to display the selected image
        if hasattr(upload_image, 'image_label'):
            upload_image.image_label.config(image=uploaded_image_tk)
        else:
            upload_image.image_label = tk.Label(root, image=uploaded_image_tk)
            upload_image.image_label.place(x=270, y=530)  # Place the image at desired location
        
        upload_image.image_label.image = uploaded_image_tk  # Keep a reference to avoid garbage collection

# Functions for model testing (no changes here)
def predict_with_vgg16():
    global predicted_label  # Use global to modify the predicted_label variable
    print(file_path)
    vgg16 = load_model('vgg16_model.h5')  
    img = image.load_img(file_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    predictions = vgg16.predict(img_array)

    predicted_class = np.argmax(predictions)  # Class with highest probability
    confidence = predictions[0][predicted_class]  # Confidence score

    class_labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro'] 
    predicted_label = class_labels[predicted_class]
    prediction_label.configure(text=f"{predicted_label}")  

    print("Predicting with VGG16")



    print("Predicting with VGG16")

def predict_with_resnet50():
    global predicted_label  # Use global to modify the predicted_label variable
    print(file_path)
    img = image.load_img(file_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    predictions = Resnet.predict(img_array)

    predicted_class = np.argmax(predictions)  # Class with highest probability
    confidence = predictions[0][predicted_class]  # Confidence score

    class_labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro'] 
    predicted_label = class_labels[predicted_class]
    prediction_label.configure(text=f"{predicted_label}")
    print("Predicting with ResNet50")

def predict_with_mobilenet():
    global predicted_label  # Use global to modify the predicted_label variable
    print(file_path)
    mobileNet = load_model('mobilenetv2_rice_disease_model_finetuned.h5')  
    img = image.load_img(file_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    predictions = mobileNet.predict(img_array)

    predicted_class = np.argmax(predictions)  # Class with highest probability
    confidence = predictions[0][predicted_class]  # Confidence score

    class_labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro'] 
    predicted_label = class_labels[predicted_class]
    prediction_label.configure(text=f"{predicted_label}") 
    print("Predicting with MobileNet")

def predict_with_custom_cnn():
    global predicted_label  # Use global to modify the predicted_label variable
    print(file_path)
    cnn1 = load_model('cnn1_model.h5')  
    img = image.load_img(file_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    predictions = cnn1.predict(img_array)

    predicted_class = np.argmax(predictions)  # Class with highest probability
    confidence = predictions[0][predicted_class]  # Confidence score

    class_labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro'] 
    predicted_label = class_labels[predicted_class]
    prediction_label.configure(text=f"{predicted_label}") 
    print("Predicting with Custom CNN (RiceNet)")

def predict_with_cnn():
    global predicted_label  # Use global to modify the predicted_label variable
    print(file_path)
    cnn2 = load_model('cnn2_model.h5')  
    img = image.load_img(file_path, target_size=(256, 256)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    predictions = cnn2.predict(img_array)

    predicted_class = np.argmax(predictions)  # Class with highest probability
    confidence = predictions[0][predicted_class]  # Confidence score

    class_labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro'] 
    predicted_label = class_labels[predicted_class]
    prediction_label.configure(text=f"{predicted_label}") 
    print("Predicting with CNN")

# Create the main window
root = ctk.CTk()
root.title("Plant Disease Prediction")
root.state("zoomed")  # Automatically fit the screen size

# Load the background image
background_image = Image.open("rice2 (1).png")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
background_image = background_image.resize((screen_width, screen_height))
background_photo = ImageTk.PhotoImage(background_image)

# Set the background
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Add a button to upload an image
upload_button = ctk.CTkButton(root, text="Upload Image", command=upload_image, font=("Arial", 14), width=150, height=60, fg_color="#15281C", hover_color="#264833")
upload_button.place(x=84, y=680)

# Add buttons for each model (no changes here)
vgg16_button = ctk.CTkButton(root, text="choose", command=predict_with_vgg16, font=("Arial", 14), width=90, height=4, fg_color="#67B685", hover_color="#15281C")
vgg16_button.place(x=1450, y=170)

resnet50_button = ctk.CTkButton(root, text="choose", command=predict_with_resnet50, font=("Arial", 14), width=90, height=4, fg_color="#46805C", hover_color="#15281C")
resnet50_button.place(x=1450, y=310)

mobilenet_button = ctk.CTkButton(root, text="choose", command=predict_with_mobilenet, font=("Arial", 14), width=90, height=4, fg_color="#376247", hover_color="#15281C")
mobilenet_button.place(x=1450, y=460)

custom_cnn_button = ctk.CTkButton(root, text="choose", command=predict_with_custom_cnn, font=("Arial", 14), width=90, height=4, fg_color="#264833", hover_color="#15281C")
custom_cnn_button.place(x=1450, y=615)

cnn_button = ctk.CTkButton(root, text="choose", command=predict_with_cnn, font=("Arial", 14), width=90, height=4, fg_color="#15281C", hover_color="#264833")
cnn_button.place(x=1450, y=755)

prediction_label = ctk.CTkLabel(root, text="", font=("Arial", 51), bg_color="white", text_color="black")
prediction_label.place(x=830, y=355)



# Run the app
root.mainloop()