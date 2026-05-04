import pickle
import cv2 as cv
import numpy as np
import streamlit as st

# - - - - main functions - - - -

IMAGE_SIZE = 128
CLASSES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
# Neighboring pixels' coordinates of the center pixel
P_COORDS = [(-1, -1), (-1, 0), (-1, 1), # (top left) (top center) (top right)
            ( 0,  1),                   # (center right)
            ( 1,  1), ( 1, 0), (1, -1), # (bottom right) (bottom center) (bottom left)
            ( 0, -1)]                   # (center left)

def load_model(filename):
    with open(f"output/{filename}", "rb") as f:
        return pickle.load(f)

def preprocess(img):
    # Resize image into same resolutions
    img_resized = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Convert image from BGR (3-channels) to grayscale (1-channel)
    img_grayscaled = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
    return(img_grayscaled)

# Convert binary list value to decimal representation
def get_decimal(binary):
    transition = 0
    decimal = 0

    # Assign each binary digit to its decimal weight
    for i in range(8):
        # Count transitions by comparing current bit with the next bit
        next_bit = (i+1) % 8
        # Add the transition count if those two bit values are different
        transition += binary[i] != binary[next_bit]
        # Convert every bit of a binary list into its decimal value
        decimal = decimal * 2 + binary[i]

    return decimal if(transition <= 2) else 256

# Get histogram representation of image feature
def get_histogram(img, non_uniform=0):
    histogram, _ = np.histogram(a=img, bins=[x for x in range(257)])
    uniform_histogram = []

    # Delete zero values throughout the list
    uniform_histogram = np.delete(histogram, np.where(histogram == 0))
    # Append non uniform value into 59th element of the uniform pattern histogram 
    uniform_histogram = np.append(uniform_histogram, non_uniform)

    return uniform_histogram

def get_lbp(img, uniform=True):
    height, width = img.shape
    # Add padding to image edges with the value of 0
    img_pad = np.pad(img, pad_width=1, mode="constant", constant_values=0)
    img_lbp = np.zeros((height, width), np.uint8)
    non_uniform = 0

    for i in range(height):
        for j in range(width):
            # Center pixel on padded image
            c_i, c_j = i+1, j+1
            c = img_pad[c_i, c_j]

            # Center pixel's binary representation
            c_bin = []

            for p_coord in P_COORDS:
                # Neighboring pixels of center pixel
                p = img_pad[c_i + p_coord[0], c_j + p_coord[1]]

                # Threshold and append bit value to pixel representation
                c_bin.append(1 if(p >= c) else 0)

            # Convert center pixel's binary representation to its decimal value
            decimal = get_decimal(c_bin)
            # Filter non-uniform pattern
            if(decimal == 256): non_uniform += 1
            # Assign uniform decimal value
            else: img_lbp[i][j] = decimal

    # Get relevant feature vector histogram based on the created feature image
    hist_lbp = get_histogram(img_lbp, non_uniform)

    return img_lbp, hist_lbp

# - - - - userR interRface - - - -

st.set_page_config(page_title="brrrRRAain", layout="centered", page_icon="🧟‍♂️")
st.title("🧠 we want brrrRRAainsss!! 🧟‍♂️")

model = load_model("best_svm.pkl")

img = st.file_uploader("send us yourR brRain pic and we will analyze its quality ggrRRrr", type=["jpg", "jpeg", "png"])
    
if(img):
    img_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    img_cv = cv.imdecode(img_bytes, cv.IMREAD_COLOR)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(cv.cvtColor(img_cv, cv.COLOR_BGR2RGB), caption="yourR brRain: rRawww", use_container_width=False)

    img_gray = preprocess(img_cv)

    with col2:
        st.image(img_gray, caption="yourR brRain: prReprRocesseddd", use_container_width=False)
        
    img_lbpu, hist_lbpu = get_lbp(img_gray, uniform=True)

    with col3:
        st.image(img_lbpu, caption="yourR brRain: extrRacted featurResss", use_container_width=False)

    prediction = model["model"].predict([hist_lbpu])[0]
    label = CLASSES[prediction].title()

    st.subheader("🍽️ ourR analyzerR rResult 😋")
    st.success(f"yourR brRain has {label.lower().replace("_", " ")}")

    st.subheader("😱 prRobability of ourR brRain analyzerR prRediction ⁉️")
    st.caption("ourR confidence level for each pRredictions of your brRainnn")
    if(hasattr(model["model"].best_estimator_, "predict_proba")):
        proba = model["model"].predict_proba([hist_lbpu])[0]
        st.bar_chart({CLASSES[i].title().lower(): proba[i] for i in range(4)}, horizontal=True)