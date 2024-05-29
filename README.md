# Braille book recognition program

시각 장애인들이 점자책을 이용하는데 여러 가지 불편함이 존재할 것 같아 점자책을 영상처리 기술을 사용하여 인식하고 이를 기반으로 음성파일로 변환 시켜주는 알고리즘을 개발하였습니다.

   2. [특징점 리스트화](#1-특징점-리스트화)


## 1. 영상 취득
![9_low](https://github.com/k99885/Braille-Translation-Program-for-Beginners/assets/157681578/8c15d613-2ba9-4703-a8d2-60fd70057967)

점자책을 직접 촬영하여 얻은 영상입니다. 책을 카메라로 찍었을때 대상이 평평하지않고 불균일 하기때문에 이러한 점을 고려하여 진행 하였습니다.

![10_hi](https://github.com/k99885/Braille-Translation-Program-for-Beginners/assets/157681578/c7b5573b-ae7a-403f-9ec9-2395d5ddd73c)

또한 실사용을 고려하여 회전된 이미지도 사용하였습니다.

## 2. 영상 처리
![점자 규격](https://github.com/k99885/Braille-Translation-Program-for-Beginners/assets/157681578/77ab501a-2c68-4a74-9b58-35117cc69ff8)

점자는 일정한 규격을 가지므로 일정하게 정렬시켜 주어야 합니다.

### 2.1 특징점 검출

## 1.1 특징점 리스트화
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

### 2.2 특징점 리스트화

```
    contours,_=cv2.findContours(img2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):

        x,y,width_fst,height_fst = cv2.boundingRect(cnt)

        list1_1[i]=i,x+40,y+20
```

2.1 에서 검출한 blob로 원을 그려 새롭게 나타낸 이미지에서 cv2.findContours()을 사용하여 원의 특징점을 저장하고

cv2.boundingRect() 함수를 사용하여 원의 시작 좌표들을 추출하여 리스트에 저장하였습니다.

```
    for i in range(len(list1_1)):
        x = list1_1[i][1]
        y = list1_1[i][2]
        img2_2[y, x] = 255
```

list1_1 (리스트)에 저장한 점자들의 좌표로 새로운 이미지(img2_2)에 나타내었습니다.

![img2_2](https://github.com/k99885/Braille-Translation-Program-for-Beginners/assets/157681578/e2ce3c98-b613-4491-8fb6-3089587d76cf)

이때 이미지를 임의로 회전시켜서 회전된 이미지에 대한 보정도 진행 하였습니다.

### 2.3 이미지 회전보정

```
    lines_z = cv2.HoughLines(img2_2, rho_z, theta_z, 10)
    img3_2_4,_,angle = draw_houghLines_1(img3_2_4, lines_z, 0, 3.14)
    average = sum(angle) / len(angle)


```
![img3_2_4](https://github.com/k99885/Braille-Translation-Program-for-Beginners/assets/157681578/6104a72e-c74a-4588-b4c6-a8000da347b1)

img2_2를  cv2.HoughLines()의 허프변환 함수를 사용하여 일정 thres 개수를 넘어가는 line들을 저장해주고 이들의 평균 각도를 구하였습니다.

```
hei_1, wid_1 = img3_2_4.shape[:2]
rotation_center = (wid_1 // 2, hei_1 // 2)
rotation_angle = math.degrees(average)

# 회전 매트릭스 계산
rotation_matrix = cv2.getRotationMatrix2D(rotation_center, -(90-rotation_angle), 1.0)

# 이미지 회전
img3_2_3 = cv2.warpAffine(img3_2_3, rotation_matrix, (wid_1, hei_1))
```
회전된 평균 각도를 이용하여 cv2.getRotationMatrix2D() 함수로 회전 매트릭스를 구하고  cv2.warpAffine() 매스릭스를 이용한 어파인 변환으로 이미지를 회전시켜주었습니다.

![img3_2_3](https://github.com/k99885/Braille-Translation-Program-for-Beginners/assets/157681578/b329ffc1-f4ea-4ef3-b6f3-2100d57df6dd)

img3_2_3 또한 이전의 방법과 같이  리스트(list1)에 점자들의 좌표를 저장했습니다.

### 2.4 불균인한 영상 보정 

#### 2.4.1 수평방향(가로방향) 보정
이전 단계(2.3) 에서 얻은 img3_1은 이미지의 휘어짐과 왜곡이 존재하기 떄문에 일정 각도,길이 이내의 좌표들을 수평방향으로 연결하보았습니다.

![img3_1_line](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/2b8cceea-9bb0-4a37-ad95-ef984e22f9e8)

책의 가운데 부분(영상의 왼쪽)은 두께로인한 휘어짐이 발생하므로 이부분에 대한 보정이 필요하였고

또한 점자는 일정한 규격과 규칙을 가지고 있기 떄문에 전체적으로 일정한 규격으로 만들어주어야합니다.


```
        img3_3_1 = img3_1[0:heigt_total, div6 * 0:div6 * 1]
        img3_3_2 = img3_1[0:heigt_total, div6 * 1:div6 * 2]
        img3_3_3 = img3_1[0:heigt_total, div6 * 2:div6 * 3]
        img3_3_4 = img3_1[0:heigt_total, div6 * 3:div6 * 4]
        img3_3_5 = img3_1[0:heigt_total, div6 * 4:div6 * 5]
        img3_3_6 = img3_1[0:heigt_total, div6 * 5:div6 * 6]
        img3_3_7 = img3_1[0:heigt_total, div6 * 6:div6 * 7]
```
![img3_1_1](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/75e3ff8a-a295-4d4d-a0d4-31f028f1e216)
->첫번째 이미지

우선 전체 이미지를 일정한 너비로 끊어서 여러개의 이미지를 만들었고 첫번째 이미지부터 차례대로 보정을 진행하였습니다.

```
contours3_1, _ = cv2.findContours(img3_3_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
y_max3 = 0
y_cut_1 = []
for a in range(len((contours3_1))):
    if len(contours3_1[a]) > 1:
        for b in range(len(contours3_1[a])):
            if contours3_1[a][b][0][1] > y_max3:
                y_max3 = contours3_1[a][b][0][1]
        for b in range(len(contours3_1[a])):
            contours3_1[a][b][0][1] = y_max3
        y_cut_1.append(y_max3 + 1)
        y_max3 = 0
y_cut_1 = sorted(y_cut_1)
```
첫번째 조각에서 cv2.findContours() 함수를 이용하여 같은 행으로 예측되는 부분들을 하나의 외곽선으로 인식하여 그룹을지어 contours3_1에 정보를 저장하고 

contours3_1에 저장된 정보 y값(contours3_1[a][b][0][1])으로 y기준 좌표들을 y_cut_1에 저장하였습니다.

![y_cut_1](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/78d3e8aa-1fb7-432e-a95b-b6e88f1ef494)

```
a = 0
while a != len(y_cut_1):
    if y_cut_1[a + 1] - y_cut_1[a] < 28:
        a = a + 3
    else:
        y_cut_1.insert(a + 1, y_cut_1[a] + 15)
        a = a + 3
for contour in contours3_1:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img3_3_1, (x, y), (x + w, y + h), (0, 0, 255), 1)
for a in range(len(y_cut_1) - 1):
    if (y_cut_1[a + 1] - y_cut_1[a]) < 12:
        y_cut_1[a + 1] = y_cut_1[a] + 13
for a in y_cut_1:
    cv2.line(img3_3_1, (0, a), (img3_3_1.shape[1], a), (255, 0, 0), 1)

```
잘려진 이미지같은 경우에는 점자가 3행모두 존재하지 않은 경우가 있기때문에 3개의 행을 만들어주었습니다.

![img3_1_1](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/93a1bb58-eb0e-4dbe-84f8-24981482baac)
흰색->빨간색->파란색

![수평기준2](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/ccdcacd5-421e-4797-ba07-a594e703ad74)

첫번째 과정과 마찬가지로 순차적으로 나머지 이미지들도 기준점들을 정해주었습니다.

```
min_y1 = 999
num_y1=0
    for i in range(len(list1)):
        x, y = list1[i][1:]
        x, y = int(x), int(y)
        if x < div6:
            for a in range(len(y_cut_1)):
                if (y_cut_1[a] - y) > 0 and (y_cut_1[a] - y) < min_y1:
                    min_y1 = y_cut_1[a] - y
                    num_y1=a
            list1[i][2]=y_cut_2[num_y1]
 
```
우선 왜곡이 심한  첫번쨰,두번째 이미지에서 점자가 저장된 list1에서 y값들을 기준좌표로 매핑 시켜주었습니다.

```
min_cut=999
    for b in range(len(y_cut_3)):
            if y_cut_3[b]<min_cut:
                min_cut=y_cut_3[b]
            if y_cut_4[b]<min_cut:
                min_cut=y_cut_4[b]
            if y_cut_5[b]<min_cut:
                min_cut=y_cut_5[b]
            if y_cut_6[b]<min_cut:
                min_cut=y_cut_6[b]
            if y_cut_7[b] < min_cut:
                min_cut = y_cut_7[b] 
            y_cut_3[b]=y_cut_4[b]=y_cut_5[b]=y_cut_6[b]=y_cut_7[b]=min_cut
            min_cut = 999

    for i in range(len(list1)):
        x, y = list1[i][1:]
        x, y = int(x), int(y)
        # for a in range(2,6):
        if x >= div6 * 2 and x < div6 * (7) :
            y=find_closest_value(y_cut_3,y)
            list1[i][2] = y
```
왜곡이 비교적 양호한 이미지들도 min_cut으로 매핑 시켜주고,

```
    for b in range(len(y_cut_3)):
        min_cut=int((y_cut_3[b]+y_cut_2[b])/2)
        y_cut_2[b]=y_cut_3[b]=y_cut_4[b]=y_cut_5[b]=y_cut_6[b]=y_cut_7[b]=min_cut
        #min_cut = 999

    for i in range(len(list1)):
        x, y = list1[i][1:]
        x, y = int(x), int(y)
        y = find_closest_value(y_cut_3, y)
        list1[i][2] = y
```
첫번째,두번째의 이미지와 나머지 이미지도 하나의 기준으로 통일시켜 전체이미지를 리매핑 시켜 주었습니다. 
![수평기준4](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/029769dc-2cf2-44c7-96da-80d903879ae0)

![image](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/c393cf70-aef2-4e5b-a3d9-d2bf9e83cb07)


이로서 수평방향으로 보정을 진행하였습니다.

#### 2.4.1 수직방향(세로방향) 보정

```
    for i in range(len(list1)):
        for j in range(i + 1, len(list1)):
            x1,y1 = list1[i][1:]
            x2,y2 = list1[j][1:]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            if (x2 - x1) != 0:
                angle_rad = (y2 - y1) / (x2 - x1)
            else:
                angle_rad = 0
            angle_radians = math.atan(angle_rad)
            angle_degrees = math.degrees(angle_radians)
            if x1<div6*4 and x2<div6*4 and abs(x2 - x1) <13and abs(y2 - y1) <200:
               if angle_degrees<-68 and angle_degrees>=-90:
                   cv2.line(img3_2f, (x1, y1), (x2, y2), (0,255,0), 1,8)
```
수직방향으로 왜곡이 심한 왼쪽부분을 먼저 일정한 간격,각도를 정하여 하나의 직선으로 이어주었습니다.

![img3_2f](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/6dc6a931-caff-48d0-952b-3d2d99e26209)


```
    contours3, _ = cv2.findContours(img3_2f, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    y_max4 = 0
    for a in range(len((contours3))):
        for b in range(len(contours3[a])):
            if contours3[a][b][0][0] > y_max4:
                y_max4 = contours3[a][b][0][0]
        for b in range(len(contours3[a])):
            if img3_2s[contours3[a][b][0][1], contours3[a][b][0][0]]==255:
                img3_2[contours3[a][b][0][1], y_max4]=255
        y_max4 = 0
```
수직방향으로 이어진 부분들을 x좌표를 하나로 통일시켜 주었습니다.

![수직방향 왜곡이 심한부분보정](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/0cf852fe-35e9-47a9-bab3-b6be4f024fc1)

```
    contours4, _ = cv2.findContours(img3_2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    list2 = np.zeros((len(contours4), 3), np.uint16)
    print(len(contours4))
    for i, cnt in enumerate(contours4):
        x, y, width, height = cv2.boundingRect(cnt)
        list2[i] = i, x, y
```
수직방향으로 왜곡이 심한부분을 보정해주었고 이것을 리스트화 하여 list2[]에 저장하였습니다.

```
    list_y = []
    for i in range(len(y_cut_3)):
        str1 = []
        list_y.append(str1)

    for j in range(len(y_cut_3)):
        for i in range(len(list2)):
            if list2[i][2] == y_cut_3[j]:
                list_y[j].append(list2[i][1])

    for j in range(len(y_cut_3)):
        list_y[j] = np.sort(list_y[j])
```
이전에 과정을 통하여 수평방향으로는 정렬이 완벽하게 되었기때문에 list_y[]에 점자들의 y값 기준좌표 리스트를 생성하였습니다.

```
    for j in range(len(list_y) - 1):
        for i in range(len(list_y[j])):
            for x in range(len(list_y) - j):
                if j < len(list_y) - x:
                    for m in range(len(list_y[j + x])):
                        if abs(int(list_y[j][i]) - int(list_y[j + x][m])) <6:
                            #list_y[j][i] = list_y[j + x][m]
                            cv2.line(img3_3s, (list_y[j][i], y_cut_3[j]), (list_y[j + x][m], y_cut_3[j + x]),(0,255,0), 1,8)
```
이전의 과정을 통하여 수직방향으로 왜곡이 심한부분은 보정하였지만 나머지 부븐을 보정하기위하여 일정한 간격을 설정하여 수직방향으로 연결 시켜주었습니다.

![img3_3s](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/16ff2146-88bc-4a49-b022-6413864c2602)

```
    contours5, _ = cv2.findContours(img3_3s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    y_max4 = 0

    for a in range(len((contours5))):
        for b in range(len(contours5[a])):
            if contours5[a][b][0][0] > y_max4:
                y_max4 = contours5[a][b][0][0]
        for b in range(len(contours5[a])):
            if img3_3[contours5[a][b][0][1], contours5[a][b][0][0]]==255:
                img3_4[contours5[a][b][0][1], y_max4]=255
        y_max4 = 0
```
이전의 과정과 똑같이 하나의 x값들로 통일시켜 주었습니다.

![img3_4_line](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/d5e60e94-46c5-4dbd-b220-6fc67a7e3063)

```
    list_y2 = []
    for i in range(len(y_cut_3)):
        str1 = []
        list_y2.append(str1)

    for j in range(len(y_cut_3)):
        for i in range(len(list3)):
            if list3[i][2] == y_cut_3[j]:
                list_y2[j].append(list3[i][1])

    for j in range(len(y_cut_3)):
        list_y2[j] = np.sort(list_y2[j])
```

정렬완료된 좌표들을  list_y2[]에 저장하였습니다.

### 2.4 점자 규격화

```
  rholist_x = []
    result_rholist_x1 = []
    for j in range(len(list_y2)):
        for i in range(len(list_y2[j])):
            x = list_y2[j][i]
            y = y_cut_3[j]
            cv2.rectangle(img4, (x, y), (x + 7, y + 7), (255, 255, 255), 1)
            result_rholist_x1.append(x)
    [rholist_x.append(x) for x in result_rholist_x1 if x not in rholist_x]

    rholist_x = np.sort(rholist_x)
```
정렬이된 점자들을 2x3으로 규격화 하기위해서 기준좌표 리스트 rholist_x[] 을 만들어주고 점자들을 rectangle으로 시각화 하였습니다.

```
    for j in range(0, len(y_cut_3), 3):
        for i in range(0, len(rholist_x), 2):
            cv2.rectangle(img4, (rholist_x[i], y_cut_3[j]), (rholist_x[i + 1] + 8, y_cut_3[j + 2] + 8), (0, 255, 0),1)
```
기준점 리스트 y_cut_3,rholist_x을 이용하여 2x3으로 점자들을 규격화 하였습니다..

![img4_1](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/0b04089f-a624-4863-b08d-a9209e95c965)

### 2.5 이진부호로 변환
```
result = np.zeros((int(len(rholist_x) / 2), int(len(y_cut_3) / 3) + 1), np.uint8)
    l, k, m, j = 0, 0, 0, 0
    y = 1
    for x in range(0, len(y_cut_3), 3):
        for i in range(0, len(rholist_x), 2):
            #######1행############################
            if rholist_x[i] == list_y2[x][m]:
                s1 = 1
                if m < len(list_y2[x]) - 1:
                    m += 1
            else:
                s1 = 0

            if rholist_x[i + 1] == list_y2[x][m]:
                s2 = 1
                if m < len(list_y2[x]) - 1:
                    m += 1
            else:
                s2 = 0
            #######2행############################
            if rholist_x[i] == list_y2[x + 1][k]:
                s3 = 1
                if k < len(list_y2[x + 1]) - 1:
                    k += 1
            else:
                s3 = 0

            if rholist_x[i + 1] == list_y2[x + 1][k]:
                s4 = 1
                if k < len(list_y2[x + 1]) - 1:
                    k += 1
            else:
                s4 = 0
            #######3행############################
            if rholist_x[i] == list_y2[x + 2][l]:
                s5 = 1
                if l < len(list_y2[x + 2]) - 1:
                    l += 1
            else:
                s5 = 0

            if rholist_x[i + 1] == list_y2[x + 2][l]:
                s6 = 1
                if l < len(list_y2[x + 2]) - 1:
                    l += 1
            else:
                s6 = 0

            res1 = str(s1) + str(s2) + str(s3) + str(s4) + str(s5) + str(s6)
            # print(res1)
            res2 = int(res1, 2)
            # result.append(res1)
            result[j][y] = res2
            result[j][0] = j
            j += 1
        l, k, m, j = 0, 0, 0, 0
        y = y + 1
```
한줄에 2x3의 점자가 있으므로 rholist_x를 3행으로 나누어서 이진부호로 인식하고 10진수로 변환시켜 result[][]에 결과값을 저장하였습니다.

![result](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/49b0d76a-fe93-41ae-8071-0062395f38e6)

이진 변환을 열으로 진행하고 하나의 열이 변환완료 되었으면 다음 열으로 이동하는 방식으로 진행하여 사진과 같은 result[][]를 얻었습니다.

```
    for j in range(0, len(y_cut_3), 3):
        for i in range(0, len(rholist_x), 2):
            y = int(j / 3 + 1)
            x = int(i / 2)
            cv2.putText(img4, str(result[x][y]), (rholist_x[i], y_cut_3[j] + 50), cv2.FONT_HERSHEY_PLAIN, 1,
                       (0, 0, 255))
```

10진수로 변환된 점자들을 해당하는 구역에 디스플레이 해주었습니다.

![img4](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/edfccf09-a5e0-4260-9c8a-04519c3f3a27)

영상 처리가 완료된 점자들을 보기좋게 새로 만들어주었습니다.

![결과](https://github.com/k99885/Braille-book-recognition-program/assets/157681578/abacfe66-b642-425a-ac5f-da434a02a615)

## 3. 점자 데이터(10진수) 자연어로 변환


2차원 리스트로 이루어진 result를 1차원 리스트인 result1[]으로 변경해주고 

![result1](https://github.com/k99885/Braille_book_recognition_program/assets/157681578/75728860-858c-4303-824a-3a1a677f6891)

```
import transe   
result2 = transe.trans_data(result1)
```
 
trans_data함수를 사용하여 한글로 변환시켜주었습니다.

![result2](https://github.com/k99885/Braille_book_recognition_program/assets/157681578/a72e875d-de5e-47dc-97a1-4c9f9a732b78)

- transe.py

