import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from src.dnn import InceptionV3

# CHANGE
REPEAT_CLASSIFY = False


def show_summary(classifications: dict):
    plt.cla()
    total = np.zeros(3)

    labels = classifications[0].keys()

    for i, batch in enumerate(classifications):
        values = list(batch.values())
        total += values

        # plt.title(f"Batch No. {i + 1}")
        # plt.pie(values, labels=labels)
        # plt.show(block=True)

        print(f"Batch No. {i + 1} summary:\n True: {values[0]}\n "
              f"Target: {values[1]}\n Other: {values[2]}")


    # plt.title("Total Summary")
    # plt.pie(total, labels=labels)
    # plt.show(block=True)

    print(f"Total summary:\n True: {total[0]}\n "
          f"Target: {total[1]}\n Other: {total[2]}")



def update_counts(label, true, target):
    if label in true:
        classified[-1]["true"] += 1
    elif label in target:
        classified[-1]["target"] += 1
    else:
        classified[-1]["other"] += 1


def draw_graph(arr):
    plt.cla()
    labels, probs = zip(*arr)
    index = np.arange(len(arr))

    plt.bar(index, probs)
    plt.xticks(index, labels)
    plt.ylim(0, 1)

    plt.draw()


def show_text(msg):
    output = np.ones((250, 700)) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, msg, (70, 100),
                font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("prediction", output)


def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[starty:starty+cropy, startx:startx+cropx, :]


def make_new_dict():
    return {
        "true": 0,
        "target": 0,
        "other": 0
    }


plt.ion()  # make plot interactive
plt.show()

classified = []
# CHANGE HERE!
# Add the indexes of the true and classes (e.g., 504 for coffee mug, 363
# for an armadillo, etc.) You can use more than one label as shown here:
true_labels = [504, 968]
target_labels = [363]

cap = cv2.VideoCapture(1)
dnn = InceptionV3()
input_shape = dnn.get_input_shape()

sess = tf.Session()

inp = tf.placeholder(dtype=np.float32, shape=(*input_shape, 3))

processed_images = tf.expand_dims(inp, 0)
logits, probs = dnn.get_logits_prob(processed_images)

sess.run(tf.global_variables_initializer())
dnn.init_session(sess)
try:
    while True:
        ret, frame = cap.read()

        # Our operations on the frame come here
        cropped_frame = crop_center(frame, *input_shape)
        assert(cropped_frame.shape == (*input_shape, 3))  # for inception
        # Display the resulting frame

        cv2.imshow('crop', cropped_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # press 's' to start/pause classification
            if not REPEAT_CLASSIFY:  # start new batch
                classified.append(make_new_dict())
                print(f"[*] batch no. {len(classified)} started")

            REPEAT_CLASSIFY = not REPEAT_CLASSIFY

        if REPEAT_CLASSIFY or key == ord('c'):
            normalized_frame = cropped_frame / 255
            probabilities = sess.run([probs], {inp: normalized_frame})
            probabilities = probabilities[0]

            predictions = dnn.get_k_top_with_probs(probabilities, k=3)
            print(predictions[0])

            top = probabilities.argsort()[-3:][::-1]

            update_counts(top[0], true=true_labels, target=target_labels)
            # draw_graph(predictions)

            show_text(predictions[0][0])

        if key == ord('q'):
            break
finally:
    cap.release()
    show_summary(classified)
    cv2.destroyAllWindows()
