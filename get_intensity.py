# importing the module
import cv2
import os


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(img[y, x]), (x, y), font,
                    1, (139, 0, 0), 1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (139, 0, 0), 2)
        cv2.imshow('image', img)


# driver function
if __name__ == "__main__":
    path = 'D:/Sources/Python/DO AN 2/SOURCE CODE/BANGDIEMRIENG'  # path to image folder
    filename = '116298-1.png'
    img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
