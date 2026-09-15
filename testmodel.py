import json
import numpy as np
import tensorflow as tf
import tkinter as tk

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "testmodel/cocoscan_efficientnetb4_offline.keras"
LABEL_PATH = "testmodel/class_labels.json"

IMG_SIZE = 456


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# =========================================================
# LOAD CLASS LABELS
# =========================================================

with open(LABEL_PATH, "r") as f:
    label_map = json.load(f)

class_names = {
    int(index): class_name
    for class_name, index in label_map.items()
}

print("\nClasses:")

for index in sorted(class_names):
    print(index, "=", class_names[index])


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(img_path):

    # Load image
    img = image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    # Convert image to NumPy array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Same preprocessing used during training
    img_array = preprocess_input(
        img_array
    )

    # Get predictions
    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    # Highest probability
    predicted_index = np.argmax(
        predictions
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index] * 100
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# =========================================================
# SELECT IMAGE
# =========================================================

def select_image():

    file_path = filedialog.askopenfilename(
        title="Select Coconut Image",

        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp *.webp"
            )
        ]
    )

    if not file_path:
        return

    # Display selected image
    display_image(file_path)

    # Run prediction
    try:

        predicted_class, confidence, predictions = (
            predict_image(file_path)
        )

        # Display main prediction
        prediction_label.config(
            text=(
                f"Prediction:\n"
                f"{predicted_class}"
            )
        )

        confidence_label.config(
            text=(
                f"Confidence: "
                f"{confidence:.2f}%"
            )
        )

        # =================================================
        # DISPLAY ALL PROBABILITIES
        # =================================================

        probability_text.delete(
            "1.0",
            tk.END
        )

        sorted_predictions = sorted(
            enumerate(predictions),
            key=lambda x: x[1],
            reverse=True
        )

        for index, probability in sorted_predictions:

            probability_text.insert(
                tk.END,
                f"{class_names[index]:25s} "
                f"{probability * 100:6.2f}%\n"
            )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


# =========================================================
# DISPLAY IMAGE
# =========================================================

def display_image(img_path):

    img = Image.open(img_path)

    # Make a copy so the original isn't modified
    img = img.copy()

    # Resize while maintaining aspect ratio
    img.thumbnail(
        (500, 400)
    )

    photo = ImageTk.PhotoImage(
        img
    )

    image_label.config(
        image=photo
    )

    # Keep reference so Python doesn't delete it
    image_label.image = photo


# =========================================================
# RESET
# =========================================================

def reset():

    image_label.config(
        image=""
    )

    prediction_label.config(
        text="Prediction:\n---"
    )

    confidence_label.config(
        text="Confidence: ---"
    )

    probability_text.delete(
        "1.0",
        tk.END
    )


# =========================================================
# CREATE GUI
# =========================================================

root = tk.Tk()

root.title(
    "Cocoscan AI - Coconut Disease Detection"
)

root.geometry(
    "900x700"
)

root.resizable(
    False,
    False
)


# =========================================================
# TITLE
# =========================================================

title_label = tk.Label(
    root,
    text="Cocoscan AI",
    font=(
        "Arial",
        26,
        "bold"
    )
)

title_label.pack(
    pady=(20, 5)
)


subtitle_label = tk.Label(
    root,
    text=(
        "EfficientNetB4 Coconut Disease "
        "Classification"
    ),
    font=(
        "Arial",
        12
    )
)

subtitle_label.pack(
    pady=(0, 15)
)


# =========================================================
# IMAGE DISPLAY
# =========================================================

image_frame = tk.Frame(
    root
)

image_frame.pack(
    pady=10
)

image_label = tk.Label(
    image_frame,
    text="No image selected",
    width=60,
    height=20,
    relief=tk.SOLID,
    borderwidth=1
)
image_label.pack()


# =========================================================
# BUTTONS
# =========================================================

button_frame = tk.Frame(
    root
)
button_frame.pack(
    pady=15
)


select_button = tk.Button(
    button_frame,
    text="Select Image",
    font=(
        "Arial",
        12,
        "bold"
    ),
    width=18,
    command=select_image
)

select_button.pack(
    side=tk.LEFT,
    padx=10
)


reset_button = tk.Button(
    button_frame,
    text="Reset",
    font=(
        "Arial",
        12
    ),
    width=12,
    command=reset
)

reset_button.pack(
    side=tk.LEFT,
    padx=10
)


# =========================================================
# PREDICTION RESULT
# =========================================================

prediction_label = tk.Label(
    root,
    text="Prediction:\n---",
    font=(
        "Arial",
        18,
        "bold"
    )
)

prediction_label.pack(
    pady=(5, 0)
)


confidence_label = tk.Label(
    root,
    text="Confidence: ---",
    font=(
        "Arial",
        14
    )
)

confidence_label.pack(
    pady=5
)


# =========================================================
# ALL PROBABILITIES
# =========================================================

probability_title = tk.Label(
    root,
    text="Class Probabilities",
    font=(
        "Arial",
        13,
        "bold"
    )
)

probability_title.pack(
    pady=(10, 3)
)


probability_text = tk.Text(
    root,
    width=55,
    height=8,
    font=(
        "Consolas",
        10
    )
)

probability_text.pack(
    pady=5
)


# =========================================================
# START APPLICATION
# =========================================================

print("Starting Cocoscan AI interface...")

root.mainloop()