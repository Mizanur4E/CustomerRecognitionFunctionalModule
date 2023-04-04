import cv2


def draw_boxes(img, box, name, scale):

    #cv2.rectangle(img, (540, 640), (1040, 1240), (255, 0, 0), 10)
    if len(box) == 4:
        [xmin, ymin, xmax, ymax] = box
        xmin, ymin, xmax, ymax = int(xmin // scale), int(ymin // scale), int(xmax // scale), int(ymax // scale)
        if name == 'unknown':
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
        else:
            print(img.shape, box, xmin, xmax, ymin, ymax)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)

            cv2.rectangle(img, (xmin, ymin - 35), (xmin + len(name) * 35, ymin - 2), (0, 255, 255), -1)
            cv2.putText(img, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), thickness=2,
                        lineType=1)
        return img


