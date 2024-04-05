import tkinter as tk
import cv2
import numpy as np
from keras.models import load_model


def tahmin():
    model = load_model('model0.h5')

    image = cv2.imread('uu_harf.png', cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    image = cv2.resize(image, (32, 32))  # Resize image
    inputdata = image.reshape(-1, 32, 32, 1).astype('float32') / 255.0  # Normalize etme

    predictions = model.predict(inputdata)
    predicted_class_index = np.argmax(predictions)

    labels = ['A', 'B', 'C', 'Ç', 'D', 'E', 'F', 'G', 'Ğ', 'H', 'I', 'İ', 'J', 'K', 'L', 'M', 'N', 'O', 'Ö', 'P',
              'R',
              'S', 'Ş', 'T', 'U', 'Ü', 'V', 'Y', 'Z']  # Modelin öğrendiği sınıf etiketlerinizin listesi
    predicted_class = labels[predicted_class_index]
    print("Tahmin edilen sınıf:", predicted_class)


root = tk.Tk()
root.title("Tahmin Uygulaması")

# Buton oluşturma
button = tk.Button(root, text="Tahmin Yap", command=tahmin)
button.pack(pady=10)

# Sonuç gösterme alanı oluşturma
label_sonuc = tk.Label(root, text="")
label_sonuc.pack()

# Pencereyi çalıştırma
root.mainloop()
