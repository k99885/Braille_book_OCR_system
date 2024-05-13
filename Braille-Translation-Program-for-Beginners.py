import cv2,math
import numpy as np
import matplotlib.pyplot as plt
import transe
#image = cv2.imread("images2/3.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("images2/10_low_3.jpg",cv2.IMREAD_COLOR)# 컬러 영상 읽기
image2=cv2.GaussianBlur(image,(5,5),0)
gray=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
if image is None: raise Exception("영상 파일 읽기 오류")

def find_closest_value(numbers, target):
    return min(numbers, key=lambda x: abs(x - target))
def draw_houghLines_1(src, lines,min,max):
    rholist = []
    radians=[]
    for i in range(len(lines)):
        rho, radian = lines[i, 0, 0:2]  # 수직거리 , 각도 - 3차원 행렬임
        if radian>=min and radian<max:
            rholist.append(int(rho))
            radians.append(radian)
            #print(radian)
            # if radian>=0 and radian<0.01:
            #     print(i,"번쨰","rho:",rho,"radian:",radian)
            a, b = math.cos(radian), math.sin(radian)
            pt = (a * rho, b * rho)  # 검출 직선상의 한 좌표 계산
            delta = (-1000 * b, 1000 * a)  # 직선상의 이동 위치
            pt1 = np.add(pt, delta).astype('int')
            pt2 = np.subtract(pt, delta).astype('int')
            cv2.line(src, tuple(pt1), tuple(pt2), (0, 0,255), 1, cv2.LINE_AA)

    return src,rholist,radians
def draw_houghLines(src, lines,min,max):
    rholist = []
    for i in range(len(lines)):
        rho, radian = lines[i, 0, 0:2]  # 수직거리 , 각도 - 3차원 행렬임
        if radian>=min and radian<max:
            rholist.append(int(rho))
            # if radian>=0 and radian<0.01:
            #     print(i,"번쨰","rho:",rho,"radian:",radian)
            a, b = math.cos(radian), math.sin(radian)
            pt = (a * rho, b * rho)  # 검출 직선상의 한 좌표 계산
            delta = (-2000 * b, 2000 * a)  # 직선상의 이동 위치
            pt1 = np.add(pt, delta).astype('int')
            pt2 = np.subtract(pt, delta).astype('int')
            cv2.line(src, tuple(pt1), tuple(pt2), (0, 0,255), 1, cv2.LINE_AA)

    return src,rholist

def onparams(value):
    th[0] = cv2.getTrackbarPos("th[0]", "Keypoints")
    th[1] = cv2.getTrackbarPos("th[1]", "Keypoints")
    th[2] = cv2.getTrackbarPos("th[2]", "Keypoints")
    th[3] = cv2.getTrackbarPos("th[3]", "Keypoints")
    th[4] = cv2.getTrackbarPos("th[4]", "Keypoints")

    params = cv2.SimpleBlobDetector_Params()

    # 경계값 조정 ---②
    params.minThreshold = 10
    params.maxThreshold = 240
    params.thresholdStep = 1

    params.minDistBetweenBlobs=5

    params.filterByArea = True
    params.minArea = 2#th[0]
    params.maxArea = 6.5#th[1]

    params.filterByColor = True
    params.blobColor=255

    params.filterByInertia = True
    params.minInertiaRatio = 0.1#th[2]/10

    params.filterByCircularity = True
    params.minCircularity = 0.4#th[3]/10

    params.filterByConvexity = True
    params.minConvexity = 0.7#th[4]/10

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    rows, cols = image.shape[:2]
    img3_1 = np.zeros((rows+40, cols+40), np.uint8)
    img3_2 = np.zeros((rows + 40, cols + 40), np.uint8)
    img3_2f = np.zeros((rows + 40, cols + 40), np.uint8)
    img3_2_1 = np.zeros((rows + 40, cols + 40), np.uint8)
    img3_3 = np.zeros((rows + 40, cols + 40), np.uint8)
    img3_4 = np.zeros((rows + 40, cols + 40), np.uint8)
    img4 = np.zeros((rows + 40, cols + 40), np.uint8)
    img2_2 = np.zeros((rows + 40, cols + 40), np.uint8)

    img2 = np.zeros((rows, cols), np.uint8)



    img = cv2.drawKeypoints(image, keypoints,None, (0,0,255),cv2.DRAW_MATCHES_FLAGS_DEFAULT) #cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2=cv2.drawKeypoints(img2, keypoints,None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Keypoints", img)


    contours,_=cv2.findContours(img2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    print("image size:", (rows,cols))
    print("개수:",len(contours))

    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    list1_1 = np.zeros((len(contours), 3), np.uint16)

    for i, cnt in enumerate(contours):

        x,y,width_fst,height_fst = cv2.boundingRect(cnt)

        list1_1[i]=i,x+40,y+20

    for i in range(len(list1_1)):
        x = list1_1[i][1]
        y = list1_1[i][2]
        img2_2[y, x] = 255

    #cv2.imshow("img2_2",img2_2)

    rho_z, theta_z = 2, np.pi /78
    lines_z = cv2.HoughLines(img2_2, rho_z, theta_z, 10)

    img2_2 = cv2.cvtColor(img2_2, cv2.COLOR_GRAY2BGR)
    img3_2_4 = img2_2.copy()
    img3_2_3 = img2_2.copy()
    img3_2_4,_,angle = draw_houghLines_1(img3_2_4, lines_z, 0, 3.14)
    #print(angle)
    average = sum(angle) / len(angle)
    #print(average)
    #cv2.imshow("img3_2_4", img3_2_4)

    hei_1, wid_1 = img3_2_4.shape[:2]
    rotation_center = (wid_1 // 2, hei_1 // 2)
    #print("기울기보정전:",hei_1, wid_1)
    rotation_angle = math.degrees(average)
    #print(90-rotation_angle)

    # 회전 매트릭스 계산
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, -(90-rotation_angle), 1.0)

    # 이미지 회전
    img3_2_3 = cv2.warpAffine(img3_2_3, rotation_matrix, (wid_1, hei_1))
    #cv2.imshow("img3_2_3", img3_2_3)
    img3_2_3 = cv2.cvtColor(img3_2_3, cv2.COLOR_BGR2GRAY)
    contours2, _ = cv2.findContours(img3_2_3, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    list1 = np.zeros((len(contours2), 3), np.uint16)

    for i, cnt in enumerate(contours2):
        x, y, width_sec, height_sec = cv2.boundingRect(cnt)

        list1[i] = i, x, y
    hei_2, wid_2 = img3_2_4.shape[:2]
    #print("보정후:",hei_2, wid_2)
    img3_2 = cv2.cvtColor(img3_2, cv2.COLOR_GRAY2BGR)
    img3_1 = cv2.cvtColor(img3_1, cv2.COLOR_GRAY2BGR)
    for i in range(len(list1)):
        x = list1[i][1]
        y = list1[i][2]
        #img3_2[y, x] = 255
        img3_1[y, x] = 255
    #cv2.imshow("img3_2", img3_2)

    count1, count2 = 0, 0
    angle_sum1, angle_sum2 = 0, 0
    x_max,x_max1, x_max2 = 0, 0,0
    x_min,x_min1, x_min2 = 9999, 9999,9999
    y_max,y_max1, y_max2 = 0, 0,0
    y_min,y_min1, y_min2 = 9999, 9999,9999


######################수평으로의 연결############################################

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
            #count1 += 1
            angle_sum1 += angle_degrees
            # if abs (y2 - y1)<1 and abs(x2 - x1)<150:
            #     cv2.line(img3_1, (x1,y1), (x2,y2), (0, 0, 255), 1)
            if abs(y2 - y1) >= 2and abs(y2 - y1) < 10 and abs(x2 - x1) <100:
                #cv2.line(img3_1, (x1, y1), (x2, y2), (0, 255, 0), 1)
                #print("pt1:",(x1,y1),"pt2:",(x2,y2),"angle_deg:",angle_deg)
                if angle_degrees>0:
                    cv2.line(img3_1, (x1, y1), (x2, y2), (0,  255,0), 1)
                    if x2 >= x_max:
                        x_max=x2
                    if y2>=y_max:
                        y_max=y2
                    if y1<=y_min:
                        y_min=y1
                    if x1 <= x_min:
                        x_min =x1

    div3 = int(x_max / 3)
    img3_1_1 = img3_1[0:rows, 0:div3-1]
    img3_1_2 = img3_1[0:rows, div3:(div3 * 2)-1]
    img3_1_3 = img3_1[0:rows, div3 * 2:(div3 * 3)-1]
#############첫번쨰조각##################################
    img3_1_1 = cv2.cvtColor(img3_1_1, cv2.COLOR_BGR2GRAY)
    contours1,_ = cv2.findContours(img3_1_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img3_1_1 = cv2.cvtColor(img3_1_1, cv2.COLOR_GRAY2BGR)
    y_max3=0
    y_cut_1=[]
    y_cut_2=[]
    for a in range(len((contours1))):
        if  len(contours1[a])>1:
            #print(a,":",contours1[a])
            #print(19, ":", contours1[19][0][0][1])
            for b in range(len(contours1[a])):
                if contours1[a][b][0][1]>y_max3:
                    y_max3=contours1[a][b][0][1]
            for b in range(len(contours1[a])):
                contours1[a][b][0][1]=y_max3

            #print(a,":",contours1[a])

            y_cut_1.append(y_max3+1)
            y_max3 = 0
    y_cut_1=sorted(y_cut_1)
    #print(y_cut_1)
    a = 0
    while a != len(y_cut_1):
        if y_cut_1[a + 1] - y_cut_1[a] < 28:
            a = a + 3
        else:
            y_cut_1.insert(a + 1, y_cut_1[a] + 15)
            a = a + 3
    for contour in contours1:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img3_1_1, (x, y), (x + w, y + h), (0, 0, 255), 1)
    for a in range(len(y_cut_1)-1):
        if (y_cut_1[a+1]-y_cut_1[a])<12:
            y_cut_1[a + 1]=y_cut_1[a]+13
    for a in y_cut_1:
        cv2.line(img3_1_1,(0,a),(img3_1_1.shape[1],a),(255,0,0),1)



###########두번째조각#####################################
    img3_1_2 = cv2.cvtColor(img3_1_2, cv2.COLOR_BGR2GRAY)
    contours2, _ = cv2.findContours(img3_1_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img3_1_2 = cv2.cvtColor(img3_1_2, cv2.COLOR_GRAY2BGR)
    #print(len(contours2))
    y_max3 = 0

    for a in range(len((contours2))):
        if len(contours2[a]) > 1:
            for b in range(len(contours2[a])):
                if contours2[a][b][0][1] > y_max3:
                    y_max3 = contours2[a][b][0][1]
            for b in range(len(contours2[a])):
                contours2[a][b][0][1] = y_max3
            y_cut_2.append(y_max3+2)
            y_max3 = 0
    y_cut_2 = sorted(y_cut_2)
    #print("수정전:",y_cut_2)

    a=0
    while a != len(y_cut_2):
        if y_cut_2[a+1]-y_cut_2[a]<28:
            a=a+3
        else :
            y_cut_2.insert(a+1,y_cut_2[a]+15)
            a=a+3
        
    #print("수정후:",y_cut_2)

    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img3_1_2, (x, y), (x + w, y + h), (0, 0, 255), 1)
    for a in range(len(y_cut_2)-1):
        if (y_cut_2[a+1]-y_cut_2[a])<12:
            y_cut_2[a + 1]=y_cut_2[a]+15
    for a in y_cut_2:
        cv2.line(img3_1_2,(0,a),(img3_1_2.shape[1],a),(255,0,0),1)

####첫번째,두번째 이미지 매핑 ###########################
    min_y1=999
    for i in range(len(list1)):
        x,y=list1[i][1:]
        x,y=int(x),int(y)
        if x<div3:
            for a in range(len(y_cut_1)):
                if (y_cut_1[a]-y)>0 and (y_cut_1[a]-y)<min_y1:
                    min_y1=y_cut_1[a]-y
            list1[i][2]=y+min_y1+y_cut_2[a]-y_cut_1[a]
            min_y1=999
        if x>=div3 and x<div3*2:
            for a in range(len(y_cut_2)):
                if (y_cut_2[a]-y)>0 and (y_cut_2[a]-y)<min_y1:
                    min_y1=y_cut_2[a]-y
            list1[i][2]=y+min_y1
            min_y1 = 999

    #cv2.imshow("img3_1_1",img3_1_1)
    #cv2.imshow("img3_1_2", img3_1_2)
    #cv2.imshow("img3_1_3", img3_1_3)
##################3_2f에 새롭게 매핑###########################
    img3_2f = cv2.cvtColor(img3_2f, cv2.COLOR_GRAY2BGR)
    for i in range(len(list1)):
        x = list1[i][1]
        y = list1[i][2]
        img3_2f[y, x] = 255

##########선분 연결(수평)#####################################
    # for i in range(len(list1)):
    #     for j in range(i + 1, len(list1)):
    #         x1,y1 = list1[i][1:]
    #         x2,y2 = list1[j][1:]
    #         x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    #         if (x2 - x1) != 0:
    #             angle_rad = (y2 - y1) / (x2 - x1)
    #         else:
    #             angle_rad = 0
    #         angle_radians = math.atan(angle_rad)
    #         angle_degrees = math.degrees(angle_radians)
    #         angle_sum1 += angle_degrees
    #         if abs(y2 - y1) < 10 and abs(x2 - x1) <100:
    #             if angle_degrees>=0:
    #                 cv2.line(img3_2, (x1, y1), (x2, y2), (0,  255,0), 1)
    ##########선분 연결(수직)#####################################
    img3_2s = img3_2f.copy()
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
            angle_sum1 += angle_degrees
            if x1<div3*3 and x2<div3*3 and abs(x2 - x1) <14and abs(y2 - y1) <250:
               if angle_degrees<-68 and angle_degrees>-89 :
                    cv2.line(img3_2f, (x1, y1), (x2, y2), (0, 255, 0), 1,4)
    img3_2f = cv2.cvtColor(img3_2f, cv2.COLOR_BGR2GRAY)
    contours3, _ = cv2.findContours(img3_2f, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    img3_2f = cv2.cvtColor(img3_2f, cv2.COLOR_GRAY2BGR)
    y_max4 = 0

    img3_2s = cv2.cvtColor(img3_2s, cv2.COLOR_BGR2GRAY)
    img3_2 = cv2.cvtColor(img3_2, cv2.COLOR_BGR2GRAY)

    for a in range(len((contours3))):
        for b in range(len(contours3[a])):
            if contours3[a][b][0][0] > y_max4:
                y_max4 = contours3[a][b][0][0]
        for b in range(len(contours3[a])):
            if img3_2s[contours3[a][b][0][1], contours3[a][b][0][0]]==255:
                img3_2[contours3[a][b][0][1], y_max4]=255
        y_max4 = 0

    #img3_2[:,div3*3:]=img3_2s[:,div3*3:]

    cv2.imshow("img3_2f", img3_2f)
    cv2.imshow("img3_2s", img3_2s)
    cv2.imshow("img3_2", img3_2)



    ####################본문###########################
    #img3_2 = cv2.cvtColor(img3_2, cv2.COLOR_BGR2GRAY)

    rho_y, theta_y = 6,np.pi/10#th[0],np.pi / th[1]#6,np.pi/10#
    lines_y = cv2.HoughLines(img3_2, rho_y, theta_y, 0)

    #rho_x, theta_x = th[0], np.pi / th[1]  # th[0],np.pi / th[1]
    #lines_x = cv2.HoughLines(img3_2, rho_x, theta_x, 0)


    img3_2 = cv2.cvtColor(img3_2, cv2.COLOR_GRAY2BGR)
    img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2BGR)
    img3_2_1 = img3_2.copy()
    #img3_2_2 = img3_2.copy()

    img3_2_1, rholist_y = draw_houghLines(img3_2_1, lines_y, 1.57, 1.58)
    #img3_2_2, rholist_x = draw_houghLines(img3_2_2, lines_x, 0, 0.01)
    img3_2 = cv2.cvtColor(img3_2, cv2.COLOR_BGR2GRAY)

    contours4, _ = cv2.findContours(img3_2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    list2 = np.zeros((len(contours4), 3), np.uint16)
    print(len(contours4))
    print(rholist_y)
    for i, cnt in enumerate(contours4):
        x, y, width, height = cv2.boundingRect(cnt)
        list2[i] = i, x, y
    # print(list2)
    # print(rholist_y)
    # cv2.imshow("im3_1", img3_1)
    #cv2.imshow("img3_2_1", img3_2_1)
    #cv2.imshow("img3_2_2", img3_2_2)

    #####################최소거리로 매핑#######################(y좌표들만)
    for i in range(len(contours4)):
        list2[i][2] = find_closest_value(rholist_y, list2[i][2])
    # for i in range(len(contours)):
    #     list2[i][1] = find_closest_value(rholist_x, list1[i][1])

    rholist_y = np.sort(rholist_y)

    #####################img3_3 =새롭게 정렬된 점자들#######################
    for j in range(len(list2)):
        x = list2[j][1]
        y = list2[j][2]
        img3_3[y, x] = 255

    cv2.imshow("img3_2_1", img3_2_1)
    cv2.imshow("img3_3", img3_3)
    # cv2.imshow("img4", img4)
    # print("rholist_y:", rholist_y)

    ###########################행별 리스트 생성###########################
    list_y = []
    for i in range(len(rholist_y)):
        str1 = []
        list_y.append(str1)

    for j in range(len(rholist_y)):
        for i in range(len(list2)):
            if list2[i][2] == rholist_y[j]:
                list_y[j].append(list2[i][1])

    for j in range(len(rholist_y)):
        list_y[j] = np.sort(list_y[j])

    for j in range(len(rholist_y)):
        print("list_y", j, ":", list_y[j])

    ################x값좌표들 통일#####################################
    img3_3 = cv2.cvtColor(img3_3, cv2.COLOR_GRAY2BGR)

    for j in range(len(list_y) - 1):
        for i in range(len(list_y[j])):
            for x in range(len(list_y) - j):
                if j < len(list_y) - x:
                    for m in range(len(list_y[j + x])):
                        if abs(int(list_y[j][i]) - int(list_y[j + x][m])) <5:
                            list_y[j][i] = list_y[j + x][m]
                            #cv2.line(img3_3, (list_y[j][i], rholist_y[j]), (list_y[j + x][m], rholist_y[j + x]),(0,255,0), 1)
    ######################x기준좌표 생성(rholist_x)######################
    rholist_x = []
    result_rholist_x1 = []
    for j in range(len(list_y)):
        for i in range(len(list_y[j])):
            x = list_y[j][i]
            y = rholist_y[j]
            # img3_4[y, x] = 255
            cv2.rectangle(img4, (x, y), (x + 7, y + 7), (255, 255, 255), 1)
            result_rholist_x1.append(x)
    [rholist_x.append(x) for x in result_rholist_x1 if x not in rholist_x]

    rholist_x = np.sort(rholist_x)
    print(len(rholist_x))
    ############################점자 2x3격자 생성#######################
    # img3_3 = cv2.cvtColor(img3_3, cv2.COLOR_GRAY2BGR)
    for j in range(0, len(rholist_y), 3):
        for i in range(0, len(rholist_x), 2):
            cv2.rectangle(img4, (rholist_x[i], rholist_y[j]), (rholist_x[i + 1] + 8, rholist_y[j + 2] + 8), (0, 255, 0),1)
    ################### 이진 부호로 변환#################################################

    result = np.zeros((int(len(rholist_x) / 2), int(len(rholist_y) / 3) + 1), np.uint8)
    l, k, m, j = 0, 0, 0, 0
    y = 1
    for x in range(0, len(rholist_y), 3):
        for i in range(0, len(rholist_x), 2):
            #######1행############################
            if rholist_x[i] == list_y[x][m]:
                s1 = 1
                if m < len(list_y[x]) - 1:
                    m += 1
            else:
                s1 = 0

            if rholist_x[i + 1] == list_y[x][m]:
                s2 = 1
                if m < len(list_y[x]) - 1:
                    m += 1
            else:
                s2 = 0
            #######2행############################
            if rholist_x[i] == list_y[x + 1][k]:
                s3 = 1
                if k < len(list_y[x + 1]) - 1:
                    k += 1
            else:
                s3 = 0

            if rholist_x[i + 1] == list_y[x + 1][k]:
                s4 = 1
                if k < len(list_y[x + 1]) - 1:
                    k += 1
            else:
                s4 = 0
            #######3행############################
            if rholist_x[i] == list_y[x + 2][l]:
                s5 = 1
                if l < len(list_y[x + 2]) - 1:
                    l += 1
            else:
                s5 = 0

            if rholist_x[i + 1] == list_y[x + 2][l]:
                s6 = 1
                if l < len(list_y[x + 2]) - 1:
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

    print("result:", result)
    # x,y=0,0
    for j in range(0, len(rholist_y), 3):
        for i in range(0, len(rholist_x), 2):
            y = int(j / 3 + 1)
            x = int(i / 2)
            cv2.putText(img4, str(result[x][y]), (rholist_x[i], rholist_y[j] + 50), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 255))
    # cv2.imshow("img3_1", img3_1)
    # cv2.imshow("img3_2", img3_2)
    cv2.imshow("img3_3", img3_3)
    cv2.imshow("img4", img4)

    ########################새로 점자 생성###########################################
    img5 = np.zeros((len(rholist_y) * 18, len(rholist_x) * 18), np.uint8)
    img5 = cv2.cvtColor(img5, cv2.COLOR_GRAY2BGR)
    for j in range(0, len(rholist_y), 3):
        for i in range(0, len(rholist_x), 2):
            y = int(j / 3 + 1)
            x = int(i / 2)

            cv2.putText(img5, str(result[x][y]), (x * 30, y * 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            cv2.rectangle(img5, (x * 30, (y - 1) * 50), (x * 30 + 23, (y - 1) * 50 + 37), (0, 255, 0), 1)

            ###10진수에서 이진수(6비트)######
            binary_representation = bin(result[x][y])[2:]
            padding_length = 6 - len(binary_representation)
            six_bit_binary = '0' * padding_length + binary_representation
            six_bit_binary = str(six_bit_binary)

            for i in range(len(six_bit_binary)):
                if six_bit_binary.startswith("1", i):
                    x1 = int((x * 30) + (i % 2) * 12)
                    y1 = int((y - 1) * 50 + (i // 2) * 13)
                    cv2.rectangle(img5, (x1 + 1, y1 + 1), (x1 + 10, y1 + 10), (255, 255, 255), 1)
    cv2.imshow("img5", img5)

    #################2차원리스트->1차원으로변환하여 한글로변환##############
    result1 = []
    for i in range(1, int(len(rholist_y) / 3) + 1):
        for j in range(int(len(rholist_x) / 2)):
            result1.append(result[j][i])
    print(result1)
    result2 = transe.trans_data(result1)
    print(result2)

th=[1,1,1,1,1]
cv2.namedWindow("Keypoints")
cv2.createTrackbar("th[0]", "Keypoints", th[0], 50, onparams)
cv2.createTrackbar("th[1]", "Keypoints", th[1], 360, onparams)
cv2.createTrackbar("th[2]", "Keypoints", th[2], 180, onparams)
cv2.createTrackbar("th[3]", "Keypoints", th[3], 10, onparams)
cv2.createTrackbar("th[4]", "Keypoints", th[4], 30, onparams)

cv2.waitKey(0)
cv2.destroyAllWindows()
