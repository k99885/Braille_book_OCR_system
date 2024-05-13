# Braille Translation Program for Beginners
시각 장애인들이 점자책을 이용하는데 여러 가지 불편함이 존재할 것 같아 점자책을 영상처리 기술을 사용하여 인식하고 이를 기반으로 음성파일로 변환 시켜주는 알고리즘을 개발하였습니다.

## 1. 영상 취득
![9_low](https://github.com/k99885/Braille-Translation-Program-for-Beginners/assets/157681578/8c15d613-2ba9-4703-a8d2-60fd70057967)

점자책을 직접 촬영하여 얻은 영상입니다. 책을 카메라로 찍었을때 대상이 평평하지않고 불균일 하기때문에 이러한 점을 고려하여 진행 하였습니다.

## 2. 영상처리

### 2.1 특징점 검출

```
params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 240
params.thresholdStep = 1

params.minDistBetweenBlobs=5

params.filterByArea = True
params.minArea = 2
params.maxArea = 6.5

params.filterByColor = True
params.blobColor=255

params.filterByInertia = True
params.minInertiaRatio = 0.1

params.filterByCircularity = True
params.minCircularity = 0.4

params.filterByConvexity = True
params.minConvexity = 0.7

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(gray)
```

opencv의 SimpleBlobDetector_Params() 함수를 사용하여 blob 특징점을 검출하여 keypoint로 나타내는 작업을 진행하였습니다.

```
img = cv2.drawKeypoints(image, keypoints,None, (0,0,255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
```

drawKeypoints() 를 사용하여 검출한 keypoint들을 시각화 하였습니다.

![책 BLOB](https://github.com/k99885/Braille-Translation-Program-for-Beginners/assets/157681578/6c59d503-264f-45b9-a6e5-bef59893366b)


