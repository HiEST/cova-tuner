import numpy as np
import cv2
from mss import mss
from PIL import Image

monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

class CaptureScreen:
    def __init__(self):
        self.sct = mss()

    def capture_screen(self):
        im = np.array(self.sct.grab(monitor))
        im = np.flip(im[:, :, :3], 2)  # 1
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 2
        return True, im

    def read(self):
        return self.capture_screen()

def main():
    cap = CaptureScreen()

    while True:
        # sct.get_pixels(mon)
        ret, img = cap.read()
        #img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        cv2.imshow('test', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()