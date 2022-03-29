import cv2
import itertools
import os
import numpy as np
import shutil
from PIL import Image

GAMMA, GAMMA_THRESHOLD = 0.90, 0.001

radio_choices = ["no", ".bmp", ".jpg", ".png", ".webp"]

# File types supported by OpenCV
CV2_FILETYPES = [
    ".bmp",
    ".dib",
    ".jp2",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".pbm",
    ".pgm",
    ".png",
    ".ppm",
    ".ras",
    ".sr",
    ".tif",
    ".tiff",
    ".webp",
]

# File types supported by Pillow
PILLOW_FILETYPES = [
    ".eps",
    ".gif",
    ".icns",
    ".ico",
    ".im",
    ".msp",
    ".pcx",
    ".sgi",
    ".spi",
    ".xbm",
]

COMBINED_FILETYPES = CV2_FILETYPES + PILLOW_FILETYPES

INPUT_FILETYPES = COMBINED_FILETYPES + [s.upper() for s in COMBINED_FILETYPES]

# "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
proto_txt_path = "weights/deploy.prototxt.txt"
# "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
# "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
caffe_model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

caffe_model = cv2.dnn.readNetFromCaffe(proto_txt_path, caffe_model_path)


# #################_DEFINE FUNCTIONS_################# #
def reorient_image(im):
    try:
        image_exif = im._getexif()
        image_orientation = image_exif[274]
        if image_orientation in (2, '2'):
            return im.transpose(Image.FLIP_LEFT_RIGHT)
        elif image_orientation in (3, '3'):
            return im.transpose(Image.ROTATE_180)
        elif image_orientation in (4, '4'):
            return im.transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (5, '5'):
            return im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (6, '6'):
            return im.transpose(Image.ROTATE_270)
        elif image_orientation in (7, '7'):
            return im.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (8, '8'):
            return im.transpose(Image.ROTATE_90)
        else:
            return im
    except (KeyError, AttributeError, TypeError, IndexError):
        return im


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def intersect(v1, v2):
    a1, a2 = v1
    b1, b2 = v2
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denominator = np.dot(dap, db).astype(float)
    num = np.dot(dap, dp)
    # return (num / denominator) * db + b1
    try:
        x = (num / denominator) * db + b1
    except RuntimeWarning:
        x = (num / 0.01) * db + b1
    return x


def distance(pt1, pt2):
    """Returns the euclidean distance in 2D between 2 pts."""
    distance_to = np.linalg.norm(pt2 - pt1)
    return distance_to


def gamma(img, correction):
    """Simple gamma correction to brighten faces"""
    img = cv2.pow(img / 255.0, correction)
    return np.uint8(img * 255)


def check_underexposed(img, gray):
    """
    Returns the (cropped) image with GAMMA applied if underexposing
    is detected.
    """
    under_exp = cv2.calcHist([gray], [0], None, [256], [0, 256])
    if sum(under_exp[-26:]) < GAMMA_THRESHOLD * sum(under_exp):
        img = gamma(img, GAMMA)
    return img


def check_positive_scalar(num):
    """Returns True if value if a positive scalar."""
    if num > 0 and not isinstance(num, str) and np.isscalar(num):
        return int(num)
    raise ValueError("A positive scalar is required")


def open_file(input_filename):
    """Given a filename, returns a numpy array"""
    extension = os.path.splitext(input_filename)[1].lower()

    if extension in CV2_FILETYPES:
        # Try with cv2
        x = cv2.imread(input_filename)
        assert not isinstance(x, type(None)), 'image not found'
        return x  # cv2.imread(input_filename)

    if extension in PILLOW_FILETYPES:
        # Try with PIL
        with Image.open(input_filename) as img_orig:
            return np.array(img_orig)
    return None


def _determine_safe_zoom(img_h, img_w, x, y, w, h, percent_face):
    """
    Determines the safest zoom level with which to add margins
    around the detected face. Tries to honor `self.face_percent`
    when possible.

    Parameters:
    -----------
    img_h: int | Height (px) of the image to be cropped
    img_w: int | Width (px) of the image to be cropped
    x: int | Leftmost coordinates of the detected face
    y: int | Bottom-most coordinates of the detected face
    w: int | Width of the detected face
    h: int | Height of the detected face

    Diagram:
    --------
    i / j := zoom / 100

              +
    h1        |         h2
    +---------|---------+
    |      MAR|GIN      |
    |         (x+w, y+h)|
    |   +-----|-----+   |
    |   |   FA|CE   |   |
    |   |     |     |   |
    |   ├──i──┤     |   |
    |   |  cen|ter  |   |
    |   |     |     |   |
    |   +-----|-----+   |
    |   (x, y)|         |
    |         |         |
    +---------|---------+
    ├────j────┤
              +
    """
    # Find out what zoom factor to use given self.aspect_ratio
    corners = itertools.product((x, x + w), (y, y + h))
    center = np.array([x + int(w / 2), y + int(h / 2)])
    im = np.array(
        [(0, 0), (0, img_h), (img_w, img_h), (img_w, 0), (0, 0)]
    )  # image_corners
    image_sides = [(im[n], im[n + 1]) for n in range(4)]

    corner_ratios = [percent_face]  # Hopefully we use this one
    for c in corners:
        corner_vector = np.array([center, c])
        a = distance(*corner_vector)
        intersects = list(intersect(corner_vector, side) for side in image_sides)
        for pt in intersects:
            if (pt >= 0).all() and (pt <= im[2]).all():  # if intersect within image
                dist_to_pt = distance(center, pt)
                corner_ratios.append(100 * a / dist_to_pt)
    return max(corner_ratios)


def _crop_positions(img_h, img_w, x, y, w, h, percent_face, wide, high):
    """
    Returns the coordinates of the crop position centered
    around the detected face with extra margins. Tries to
    honor `self.face_percent` if possible, else uses the
    largest margins that comply with required aspect ratio
    given by `self.height` and `self.width`.

    Parameters:
    -----------
    img_h: int | Height (px) of the image to be cropped
    img_w: int | Width (px) of the image to be cropped
    x: int | Leftmost coordinates of the detected face
    y: int | Bottom-most coordinates of the detected face
    w: int | Width of the detected face
    h: int | Height of the detected face
    """
    # aspect ratio
    aspect = wide / high

    zoom = _determine_safe_zoom(img_h, img_w, x, y, w, h, percent_face)

    # Adjust output height based on percent
    if high >= wide:
        height_crop = h * 100.0 / zoom
        width_crop = aspect * float(height_crop)
    else:
        width_crop = w * 100.0 / zoom
        height_crop = float(width_crop) / aspect

    # Calculate padding by centering face
    x_pad = (width_crop - w) / 2
    y_pad = (height_crop - h) / 2

    # Calc. positions of crop
    h1 = x - x_pad
    h2 = x + w + x_pad
    v1 = y - y_pad
    v2 = y + h + y_pad

    return [int(v1), int(v2), int(h1), int(h2)]


def box_detect(img_path, padding, wide, high, conf, face_perc):
    if isinstance(img_path, str):
        img = open_file(img_path)
    else:
        img = img_path

    # get width and height of the image
    h_, w_ = img.shape[:2]

    # preprocess the image: resize and performs mean subtraction
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # set the image into the input of the neural network
    caffe_model.setInput(blob)

    # perform inference and get the result
    output = np.squeeze(caffe_model.forward())

    left = padding
    right = padding
    top = padding
    bottom = padding
    conf = conf / 100
    for i in range(0, output.shape[0]):
        # get the confidence
        confidence = output[i, 2]
        # if confidence is above 50%, then draw the surrounding box
        if confidence > conf:
            # get the surrounding box coordinates and upscale them to original image
            box = output[i, 3:7] * np.array([w_, h_, w_, h_])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(int)

            pos = _crop_positions(
                h_, w_, start_x - left, start_y - bottom, end_x - start_x + right, end_y - start_y + top, face_perc,
                wide, high)

            cv2.waitKey(0)
            return pos[0], pos[1], pos[2], pos[3]
        else:
            return None


def process(image, source, destination, padding, width, height, confidence, face, user_gamma, radio):
    path = source + "\\" + image
    bounding_box = box_detect(path, padding, width, height, confidence, face)
    # Save the cropped image with PIL if a face was detected
    if bounding_box is not None:
        vect0, vect1, vect2, vect3 = bounding_box  # Unpack corner coordinates
        pic = Image.open(path)  # Open image
        pic = reorient_image(pic)  # Check exif orientation and rotate accordingly
        cropped_pic = pic.crop((vect2, vect0, vect3, vect1))  # pic
        pic_array = cv2.cvtColor(np.array(cropped_pic), cv2.COLOR_BGR2RGB)  # Colour correct as Numpy array
        cropped_image = cv2.resize(pic_array, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        cropped_image = gamma(cropped_image, user_gamma / 1000)
        if radio[0]:
            cv2.imwrite(destination + "\\" + image, cropped_image)
        else:
            for n in range(1, len(radio_choices)):
                if radio[n]:
                    name, extension = os.path.splitext(image)[0], radio_choices[n]
                    cv2.imwrite(destination + "\\" + name + extension, cropped_image)

    else:
        my_file = source + "\\" + image
        reject = destination + "\\" + "reject"
        if not os.path.exists(reject):
            os.mkdir(reject, mode=0o666)
            to_file = reject + "\\" + image
        else:
            to_file = reject + "\\" + image
        shutil.copy(my_file, to_file)
