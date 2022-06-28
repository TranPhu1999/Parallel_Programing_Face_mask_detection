# init file tuần tự
from absl import flags
from absl.flags import FLAGS
import numpy as np
import numba
from numba import jit
import tensorflow as tf

import sys

yolo_max_boxes = 10
yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5
# customize model through the following parameters
flags.DEFINE_integer('yolo_max_boxes', 10,
                     'maximum number of detections at one time')
flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def BatchNormalization_forward(input, gamma, beta, moving_mean, moving_variance, epsilon=0.001):

    mean_x = moving_mean.copy()
    var_x = moving_variance.copy()

    var_x += epsilon
    stddev_x = np.sqrt(var_x)
    x_minus_mean = input - mean_x
    standard_x = x_minus_mean / stddev_x
    return gamma * standard_x + beta


# Phép correlate tham khảo từ đây https://numpy.org/doc/stable/reference/generated/numpy.convolve.html,
# https://docs.scipy.org/doc//scipy-1.3.0/reference/generated/scipy.signal.correlate2d.html
@jit(nopython=True)
def correlate2d(input, kernel, stride=1, padding="valid"):
    h_i, w_i = input.shape
    h_k, w_k = kernel.shape

    s_h = stride
    s_w = stride

    if padding == 'valid':
        p_h = 0
        p_w = 0

    if padding == 'same':
        p_h = int((h_k - 1)/2)
        p_w = int((w_k - 1)/2)

    behind = np.zeros((h_i+2*p_h, w_i+2*p_w))
    behind[p_h:h_i+p_h, p_w:w_i+p_w] = input
    input = behind

    h_out = int((h_i - h_k + 2*p_h)/stride + 1)
    w_out = int((w_i - w_k + 2*p_w)/stride + 1)

    output_conv = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            for i_kernel in range(h_k):
                for j_kernel in range(w_k):
                    output_conv[i, j] += input[i_kernel+i*stride,
                                               j_kernel+j*stride]*kernel[i_kernel, j_kernel]

    return output_conv

# Tích chập tiến
# @jit(nopython=True)


def Convolution_forward(input, kernel, filters, use_batchnorm=True, bias=[[[]]], stride=1, padding="valid"):

    _, input_height, input_width, input_depth = input.shape
    kernel_height, kernel_witdh, _, _ = kernel.shape
    # input shape is [batch_size, height, width, input_depth]
    # kernel shape is [kernel_size,kernel_size,input_depth,filters]
    # output shape is [height, width, filters]

    output_shape = (int(input_height/stride), int(input_width/stride), filters)
    output = np.zeros(output_shape)

    if use_batchnorm == False:
        for i in range(filters):
            output[:, :, i] += bias[i]
    # if use_batchnorm == False:
    #   for h in range(len(output)):
    #     for k in range(len(output[h])):
    #       for i in range(filters):
    #         output[h,k,i] += bias[i]
    for i in range(filters):
        temp = np.zeros(output_shape[:-1])
        for j in range(input_depth):
            temp += correlate2d(input[0, :, :, j], kernel[:,
                                :, j, i], stride=stride, padding=padding)
        output[:, :, i] += temp

    output = np.expand_dims(output, 0)
    return output

# hàm kích hoạt leakyReLU


def npLeakyReLU(x, alpha=0.01):
    return np.where(x >= 0, x, x*alpha)

# Layer Darknet Conv bao gồm 1 layer convole đi kèm với batch normalization và leakyReLU
def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        print(x.shape)
        x = np.pad(x, ((0, 0), (1, 0), (1, 0), (0, 0)),
                   'constant')  # top left half-padding
        padding = 'valid'

# Load weight------------------------------------------------------------

    global offset_read_weight
    global weight_file

    bias = None
    if batch_norm is False:
        # read bias weight of convolutional layer if there is no batch normalization
        bias = np.fromfile(weight_file, dtype=np.float32, count=filters)
    else:
        # read batch normalization layer weight
        bn_weights = np.fromfile(
            weight_file, dtype=np.float32, count=4*filters)
        bn_weights = bn_weights.reshape((4, filters))
        beta, gamma, moving_mean, moving_variance = bn_weights

    # read kernel weight
    conv_shape = (filters, x.shape[-1], size, size)
    kernel = np.fromfile(weight_file, dtype=np.float32,
                         count=np.product(conv_shape))
    kernel = kernel.reshape(conv_shape).transpose([2, 3, 1, 0])
# ------------------------------------------------------------------------
    x = Convolution_forward(input=x, kernel=kernel, filters=filters,
                            bias=bias, stride=strides, padding=padding, use_batchnorm=batch_norm)
    if batch_norm:
        x = BatchNormalization_forward(
            x, gamma, beta, moving_mean, moving_variance)
        x = npLeakyReLU(x, alpha=0.1)
    return x

def DarknetResidual(x, filters):
    prev = x  # Skip connection, giúp các mạng neural có cấu trúc quá sâu giảm thiểu mất mát feature khi đi xuống
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = prev + x
    return x

# Mỗi Darknet Block gồm 1 Darknet convole và n Darknet Residual
def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x

def Darknet(inputs):
    x = inputs
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return x_36, x_61, x

# Block các layer riêng của YOLO dùng cho object detection

def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            # inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = x_in

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = np.kron(x, np.ones((2, 1))).repeat(2, axis=1).astype(int)
            # x = Concatenate()([x, x_skip])
            x = np.concatenate((x, x_skip), axis=3)
        else:
            x = x_in

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        # return Model(inputs, x, name=name)(x_in)
        return x
    return yolo_conv

# Layer trả output

def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = x_in
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = (lambda x: np.reshape(x, (-1, np.shape(x)[1], np.shape(x)[2],
                                      anchors, classes + 5)))(x)
        return x
    return yolo_output

# Output của YOLO có lưu xác xuất bounding box thuộc các class
# (ví dụ như có 3 class thì sẽ có một list độ dài 3 lưu xác xuất box đó có thuộc class đó không)
# vậy nên cần dùng sigmoid để trả về giá trị từ (0-1)


# @jit(nopython=True)
def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig

def yolo_boxes(pred, anchors, classes):
    grid_size = np.shape(pred)[1]

    box_xy, box_wh, objectness, class_probs = np.split(
        pred, (2, 4, 5), axis=-1)

    box_xy = sigmoid(box_xy)
    objectness = sigmoid(objectness)
    class_probs = sigmoid(class_probs)

    pred_box = np.concatenate((box_xy, box_wh), axis=-1)

    grid = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    grid = np.expand_dims(np.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + grid.astype(np.float32)) / float(grid_size)
    box_wh = np.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = np.concatenate([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box

# reference: https://towardsdatascience.com/non-maxima-suppression-139f7e00f0b5

def combined_non_max_suppression(boxes, scores, iou_threshold, score_threshold):
    # Return an empty list, if no boxes given
    if np.shape(boxes)[0] == 0:
        return []

    x1 = boxes[:, :, :, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, :, :, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, :, :, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, :, :, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    global size
    # We add 1, because the pixel at the start as well as at the end counts
    areas = (x2 - x1 + 1/size) * (y2 - y1 + 1/size)
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.ones((np.shape(boxes)[0], np.shape(boxes)[1])) > 0
    classes = np.zeros((np.shape(boxes)[0], np.shape(boxes)[1]))
    for batch in range(np.shape(boxes)[0]):
        # indices[batch] = scores[batch,:] < score_threshold
        for i, box in enumerate(boxes[batch]):
            if np.all(scores[batch, i] < score_threshold):
                indices[batch, i] = False
            #   continue
        for i, box in enumerate(boxes[batch]):
            classes[batch, i] = np.where(
                scores[batch, i] == scores[batch, i].max())[0][0]
            scores[batch, i] = scores[batch, i].max()
            # Create temporary indices
            temp_indices = indices[batch].copy()
            temp_indices[i] = False
            # Find out the coordinates of the intersection box
            xx1 = np.maximum(box[0, 0], boxes[batch][temp_indices, 0, 0])
            yy1 = np.maximum(box[0, 1], boxes[batch][temp_indices, 0, 1])
            xx2 = np.minimum(box[0, 2], boxes[batch][temp_indices, 0, 2])
            yy2 = np.minimum(box[0, 3], boxes[batch][temp_indices, 0, 3])
            # Find out the width and the height of the intersection box
            w = np.maximum(0, xx2 - xx1 + 1/size)
            h = np.maximum(0, yy2 - yy1 + 1/size)
            # compute the ratio of overlap
            overlap = (w * h) / areas[batch, temp_indices, 0]
            # print(overlap)

            # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index
            if np.any(overlap > iou_threshold):
                indices[batch] = temp_indices

    # return only the boxes at the remaining indices
    boxes_result = boxes[indices]
    scores_result = scores[indices]
    classes_result = classes[indices]
    return boxes_result[:, 0, :], scores_result[:, 0], classes_result

def yolo_nms(outputs, anchors, masks, classes):
    # boxes, objectness, class_probs
    b, c, t = [], [], []

    for o in outputs:
        b.append(np.reshape(o[0], (np.shape(o[0])[0], -1, np.shape(o[0])[-1])))
        c.append(np.reshape(o[1], (np.shape(o[1])[0], -1, np.shape(o[1])[-1])))
        t.append(np.reshape(o[2], (np.shape(o[2])[0], -1, np.shape(o[2])[-1])))

    bbox = np.concatenate(b, axis=1)
    confidence = np.concatenate(c, axis=1)
    class_probs = np.concatenate(t, axis=1)

    scores = confidence * class_probs

    boxes, scores, classes = combined_non_max_suppression(
        boxes=np.reshape(bbox, (np.shape(bbox)[0], -1, 1, 4)),
        scores=np.reshape(
            scores, (np.shape(scores)[0], -1, np.shape(scores)[-1])),
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )

    return boxes, scores, classes

def YoloV3(inputs, size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=3):

    import time
    start1 = time.time()

    x = inputs
    x_36, x_61, x = Darknet(x)

    x = YoloConv(512)(x)
    output_0 = YoloOutput(512, len(masks[0]), classes)(x)

    x = YoloConv(256)((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes)(x)

    x = YoloConv(128)((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes)(x)

    end1 = time.time()
    print("Conv: ", end1 - start1)

    boxes_0 = (lambda x: yolo_boxes(x, anchors[masks[0]], classes))(output_0)
    boxes_1 = (lambda x: yolo_boxes(x, anchors[masks[1]], classes))(output_1)
    boxes_2 = (lambda x: yolo_boxes(x, anchors[masks[2]], classes))(output_2)
    end2 = time.time()
    print("Yolo Box: ", end2-end1)
    outputs = (lambda x: yolo_nms(x, anchors, masks, classes))(
        (boxes_0[:3], boxes_1[:3], boxes_2[:3]))
    end3 = time.time()
    print("Non-max supperession: ", end3-end2)

    return outputs


size = 416                 # size images are resized to for model
num_classes = 3            # number of classes in model
class_names = ["mask_weared_incorrect,", "with_mask", "without_mask"]

# Reference: https://meghal-darji.medium.com/implementing-bilinear-interpolation-for-image-resizing-357cbb2c2722

def img_resize(original_img, new_h, new_w):
    import numpy as np
    import math
    # get dimensions of original image
    old_h, old_w, c = original_img.shape
    # create an array of the desired shape.
    # We will fill-in the values later.
    resized = np.zeros((new_h, new_w, c))
    # Calculate horizontal and vertical scaling factor
    w_scale_factor = (old_w) / (new_w) if new_h != 0 else 0
    h_scale_factor = (old_h) / (new_h) if new_w != 0 else 0
    for i in range(new_h):
        for j in range(new_w):
            # map the coordinates back to the original image
            x = i * h_scale_factor
            y = j * w_scale_factor
            # calculate the coordinate values for 4 surrounding pixels.
            x_floor = math.floor(x)
            x_ceil = min(old_h - 1, math.ceil(x))
            y_floor = math.floor(y)
            y_ceil = min(old_w - 1, math.ceil(y))

            if (x_ceil == x_floor) and (y_ceil == y_floor):
                q = original_img[int(x), int(y), :]
            elif (x_ceil == x_floor):
                q1 = original_img[int(x), int(y_floor), :]
                q2 = original_img[int(x), int(y_ceil), :]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif (y_ceil == y_floor):
                q1 = original_img[int(x_floor), int(y), :]
                q2 = original_img[int(x_ceil), int(y), :]
                q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
            else:
                v1 = original_img[x_floor, y_floor, :]
                v2 = original_img[x_ceil, y_floor, :]
                v3 = original_img[x_floor, y_ceil, :]
                v4 = original_img[x_ceil, y_ceil, :]

                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)

            resized[i, j, :] = q
    return resized.astype(np.uint8)


# Chuẩn hóa ảnh đầu vào về kích thước 416x416 và giá trị pixel trong khoản (0,1)
def transform_images(x_train, size):
    import cv2
    x_train = img_resize(x_train, size, size)
    x_train = x_train / 255
    return x_train


def draw_outputs(img, outputs, class_names):
    import cv2
    from seaborn import color_palette
    from PIL import Image, ImageDraw
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    boxes, objectness, classes = outputs
    # boxes, objectness, classes = boxes[0], objectness[0], classes[0]
    wh = np.flip(img.shape[0:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for i in range(len(boxes)):
        color = colors[int(classes[i])]
        x1y1 = ((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = ((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        thickness = (img.size[0] + img.size[1]) // 200
        x0, y0 = x1y1[0], x1y1[1]
        for t in np.linspace(0, 1, thickness):
            x1y1[0], x1y1[1] = x1y1[0] - t, x1y1[1] - t
            x2y2[0], x2y2[1] = x2y2[0] - t, x2y2[1] - t
            draw.rectangle([x1y1[0], x1y1[1], x2y2[0], x2y2[1]],
                           outline=tuple(color))
        confidence = '{:.2f}%'.format(objectness[i]*100)
        text = '{} {}'.format(class_names[int(classes[i])], confidence)
        text_size = draw.textsize(text)
        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0],
                       fill=tuple(color))
        draw.text((x0, y0 - text_size[1]), text, fill='black')
    rgb_img = img.convert('RGB')
    img_np = np.asarray(rgb_img)
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return img


def Main():

    import time
    import cv2

    if len(sys.argv) < 3:
        print('Missing input file')
        return

    img_raw = cv2.imread(sys.argv[2])
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    img = transform_images(img_raw, size)
    img = np.expand_dims(img, 0)

    t1 = time.time()
    boxes, scores, classes = YoloV3(img, classes=num_classes)
    t2 = time.time()

    print('time: {}'.format(t2 - t1))

    print('detections:')

    class_names_local = class_names

    for i in range(len(boxes)):
        print('\t{}, {}, {}'.format(class_names_local[int(
            classes[i])], np.array(scores[i]), np.array(boxes[i])))
    img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes), class_names_local)

    return img


weight_file = open(sys.argv[1], "rb")
major, minor, revision, seen, _ = np.fromfile(
    weight_file, dtype=np.float32, count=5)

if __name__ == "__main__":
    img = Main()
    from google.colab.patches import cv2_imshow
    cv2_imshow(img)
