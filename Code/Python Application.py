# import libraries
import nibabel as nib
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserListView
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image as PILimage
#Define App class
class AD_predictor(App):
    def build(self):
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.6, 0.7)
        self.window.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        self.window.add_widget(Image(source="logo.png"))
        self.loadImage = Label(
            text="UpLoad a NIfTI file .....",
            font_size=50,
            color="#ffffff",
            bold=True
        )
        self.window.add_widget(self.loadImage)
        self.load_button = Button(text='Load Image',
                             size_hint = (0.5, 0.5),
                             bold = True,
                             font_size = 30)
        self.file_chooser = FileChooserListView(filters=['*.nii', '*.nii.gz'])
        # bind button event
        self.load_button.bind(on_press=self.load_image)
        # add widgets to layout
        self.window.add_widget(self.file_chooser)
        self.window.add_widget(self.load_button)
        return self.window

    def load_image(self, instance):
        file_path = self.file_chooser.selection[0]
        img = nib.load(file_path).get_fdata()
        for i in range(50, 170):
            # extract the ith sagittal slice
            sagittal_slice = np.rot90(img[i, :, :])
            # save the sagittal slice as a new image file
            plt.imsave(".\slices\sagittal_slice_{}.jpg".format(i), sagittal_slice, cmap="gray")
        # Create a label widget for the first number
        self.label1 = Label(text='Enter the starting number: ')
        self.window.add_widget(self.label1)
        # Create a text input widget for the second number
        self.num1 = TextInput(multiline=False, input_filter='int', input_type='number')
        self.window.add_widget(self.num1)
        # Create a label widget for the second number
        self.label2 = Label(text='Enter the Ending number: ')
        self.window.add_widget(self.label2)
        # Create a text input widget for the second number
        self.num2 = TextInput(multiline=False, input_filter='int', input_type='number')
        self.window.add_widget(self.num2)
        # Create proceed button to get output
        self.process_button = Button(
            text="Proceed",
            size_hint=(0.5, 0.5),
            bold=True,
            font_size=30
        )
        self.process_button.bind(on_press=self.predictAD)
        self.window.add_widget(self.process_button)

    def predictAD(self, instance):
        #Loading the saved model
        model = tf.keras.models.load_model('model')
        class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
        img_scores = []
        # import medial temporal lobe images as per the user's input range
        try:
            num1 = int(self.num1.text)
            num2 = int(self.num2.text)
            for i in range(num1, num2):
                img = PILimage.open(".\slices\sagittal_slice_{}.jpg".format(i))
                # rotate image by -90 degree
                img = img.transpose(PILimage.ROTATE_90)
                # resize the image into 128*128
                img = img.resize((128, 128))
                # convert to NumPy array
                img_array = np.array(img)
                # Perform contrast stretching
                p1, p99 = np.percentile(img_array, (1, 99))
                stretched = (img_array - p1) * (255.0 / (p99 - p1))
                # Convert the stretched array back to an image
                img = PILimage.fromarray(np.uint8(stretched))
                img.save("img_contrast.jpg")
                img = cv2.imread("img_contrast.jpg")
                # Apply Gaussian blur with kernel size 5x5 and sigma value of 0
                img = cv2.GaussianBlur(img, (5, 5), 0)
                # prediction using the model
                predictions = model.predict(tf.expand_dims(img, 0))
                score = tf.nn.softmax(predictions[0])
                img_scores.append(np.argmax(score))
        except ValueError:
            print("Please enter a valid integer!")
        freq_dict = {}
        for num in img_scores:
            if num in freq_dict:
                freq_dict[num] += 1
            else:
                freq_dict[num] = 1
        img_scores.clear()
        max_freq = max(freq_dict.values())
        for num in freq_dict:
            if freq_dict[num] == max_freq:
                self.loadImage.text = class_names[num]
                n = num
                break
        freq_dict.clear()
        if n == 0:
            self.loadImage.color = (1, 0.647, 0, 1)
        if n == 1:
            self.loadImage.color = (1, 0, 0, 1)
        if n == 2:
            self.loadImage.color = (0, 1, 0, 1)
        if n == 3:
            self.loadImage.color = (1, 1, 0, 1)
        self.reset_b = Button(text='Reset', on_press=self.reset)
        self.window.add_widget(self.reset_b)

    def reset(self, instance):
        self.window.remove_widget(self.num1)
        self.window.remove_widget(self.num2)
        self.window.remove_widget(self.process_button)
        self.window.remove_widget(self.label1)
        self.window.remove_widget(self.label2)
        self.loadImage.text = "UpLoad a NIfTI file ....."
        self.window.remove_widget(self.reset_b)

if __name__ == '__main__':
    AD_predictor().run()