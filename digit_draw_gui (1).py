import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Window
window = tk.Tk()
window.title("Digit Recognizer")

canvas_width = 280
canvas_height = 280

# Canvas
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# Image for drawing
image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)

# Draw function
def paint(event):
    x1, y1 = event.x - 10, event.y - 10
    x2, y2 = event.x + 10, event.y + 10
    canvas.create_oval(x1, y1, x2, y2, fill='black')
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

# Clear canvas
def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    result_label.config(text="Draw a digit")

# Predict digit
def predict():
    img = image.resize((28, 28))
    img = ImageOps.invert(img)
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    result_label.config(text=f"Predicted Digit: {digit}")

# Buttons
btn_predict = tk.Button(window, text="Predict", command=predict)
btn_predict.pack()

btn_clear = tk.Button(window, text="Clear", command=clear)
btn_clear.pack()

result_label = tk.Label(window, text="Draw a digit", font=("Arial", 16))
result_label.pack()

window.mainloop()
