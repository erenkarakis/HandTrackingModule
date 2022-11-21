import mediapipe as mp
import cv2
import time 


class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionConf=0.5, trackConf=0.5, zAxisMultiplier = 100000000):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity;
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.zAxisMultiplier = zAxisMultiplier

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionConf,
                                        self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handPoint=8, draw=True):
        lmlist = []

        if self.results.multi_hand_landmarks:
            selectedHandPoint = self.results.multi_hand_landmarks[handPoint]
            for id, lms in enumerate(selectedHandPoint.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lms.x * w), int(lms.y * h), lms.z * ((w / h) * self.zAxisMultiplier)
                #print(id, cx, cy, cz)
                lmlist.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (lmlist[handPoint][1], lmlist[handPoint][2]), 15, (255, 255, 0), cv2.FILLED)

        return lmlist

def main():
    previousTime = 0
    currentTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[0])

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

        cv2.imshow("RawFrame", img)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
