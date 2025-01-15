ORIGINAL_GOLIATH_CLASSES = (
    "Background",
    "Apparel",
    "Chair",
    "Eyeglass_Frame",
    "Eyeglass_Lenses",
    "Face_Neck",
    "Hair",
    "Headset",
    "Left_Foot",
    "Left_Hand",
    "Left_Lower_Arm",
    "Left_Lower_Leg",
    "Left_Shoe",
    "Left_Sock",
    "Left_Upper_Arm",
    "Left_Upper_Leg",
    "Lower_Clothing",
    "Lower_Spandex",
    "Right_Foot",
    "Right_Hand",
    "Right_Lower_Arm",
    "Right_Lower_Leg",
    "Right_Shoe",
    "Right_Sock",
    "Right_Upper_Arm",
    "Right_Upper_Leg",
    "Torso",
    "Upper_Clothing",
    "Visible_Badge",
    "Lower_Lip",
    "Upper_Lip",
    "Lower_Teeth",
    "Upper_Teeth",
    "Tongue",
)

ORIGINAL_GOLIATH_PALETTE = [
    [50, 50, 50],
    [255, 218, 0],
    [102, 204, 0],
    [14, 0, 204],
    [0, 204, 160],
    [128, 200, 255],
    [255, 0, 109],
    [0, 255, 36],
    [189, 0, 204],
    [255, 0, 218],
    [0, 160, 204],
    [0, 255, 145],
    [204, 0, 131],
    [182, 0, 255],
    [255, 109, 0],
    [0, 255, 255],
    [72, 0, 255],
    [204, 43, 0],
    [204, 131, 0],
    [255, 0, 0],
    [72, 255, 0],
    [189, 204, 0],
    [182, 255, 0],
    [102, 0, 204],
    [32, 72, 204],
    [0, 145, 255],
    [14, 204, 0],
    [0, 128, 72],
    [204, 0, 43],
    [235, 205, 119],
    [115, 227, 112],
    [157, 113, 143],
    [132, 93, 50],
    [82, 21, 114],
]

## 6 classes to remove
REMOVE_CLASSES = (
    "Eyeglass_Frame",
    "Eyeglass_Lenses",
    "Visible_Badge",
    "Chair",
    "Lower_Spandex",
    "Headset",
)

## 34 - 6 = 28 classes left
GOLIATH_CLASSES = tuple(
    [x for x in ORIGINAL_GOLIATH_CLASSES if x not in REMOVE_CLASSES]
)
GOLIATH_PALETTE = [
    ORIGINAL_GOLIATH_PALETTE[idx]
    for idx in range(len(ORIGINAL_GOLIATH_CLASSES))
    if ORIGINAL_GOLIATH_CLASSES[idx] not in REMOVE_CLASSES
]