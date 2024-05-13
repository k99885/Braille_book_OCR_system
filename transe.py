# import Runcode # 파일간 연결
#from typing import List, Any
from gtts import gTTS
import os
all_list = [[-1 for col_n in range(4)] for row_n in range(4)]

choseong=0
jungseong=0
jongseong = 0  # 종성이 없는 경우

# 한글 문자 조합``
hangul_char =chr(choseong * 588 + jungseong * 28 + jongseong + 44032)
#print(hangul_char)


# 출력시 저장되는 데이터 정보
# cho_uni (0~18)
#{ 'ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'}
# joong_uni (0~20)
#{'ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ'}
# jong_uni  (0~26)
#{'\0', 'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ', 'ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'}

data12=[0,17,16,38,0,20,35,42,8,52,38,9,38,16,38,24,35,0,28,24,13,0,24,35,8,20,35,9,38,16,38,29,0,42,4,41,15,0,28,61,44,46,4,41,36,33,0,1,50,33,39,0,37,17,31,28,38,0,56,0,42,8,29,0,20,26,4,38,9,20,6,39,0,17,1,62,27,0,24,35,8,20,35,36,37,0] #자기 보살핌
data13=[0, 17, 16, 38, 0, 20, 35, 42, 8, 52, 38, 9, 38, 16, 38, 24, 35, 0, 28, 24, 13, 0, 24, 35, 8, 20, 35, 9, 38, 16, 38, 29, 0, 42, 4, 41, 15, 0, 28, 61, 44, 46, 4, 41, 36, 33, 0, 1, 50, 33, 39, 0, 37, 17, 31, 28, 38, 0, 50, 0, 42, 8, 29, 0, 20, 26, 4, 38, 9, 20, 6, 39, 0, 17, 1, 62, 27, 0, 24, 35, 8, 20, 35, 36, 37, 0, 42, 15, 28, 16, 35, 0, 38, 18, 48, 39, 0, 21, 26, 38, 24, 13, 0, 28, 33, 1, 50, 57, 0, 1, 50, 33, 27, 0, 42, 0, 15, 28, 16, 38, 20, 35, 24, 0, 14, 7, 15, 28, 10, 18, 16, 17, 2, 34, 0, 1, 50, 33, 16, 35, 0, 34, 1, 34, 0, 28, 33, 1, 50, 4, 27, 0, 42, 4, 41, 15, 28, 48, 39, 0, 21, 26, 38, 0, 41, 48, 38, 4, 41, 0, 45, 24, 50, 0]


def number(data1,i):
    n=0
    if data1[i]==32:
        n=1
    if data1[i] == 40:
        n = 2
    if data1[i] == 48:
        n = 3
    if data1[i] == 52:
        n = 4
    if data1[i] == 36:
        n = 5
    if data1[i]==56 or data1[i]==50:
        n=6
    if data1[i] == 60:
        n = 7
    if data1[i] == 44:
        n = 8
    if data1[i]==24:
        n=9
    if data1[i]==28:
        n=0
    return n

def trans_data(data1):
    res1 = [[-1 for y in range(4)] for x in range(len(data1))]

    #i=2


    for i in range(len(data1)):
    ###########################초성#################################################
        if data1[i] == 1:  # code_3rd.all_list[0][0] == 0:     된소리 ={0 0 0 0 0 1}
            res1[i][0]=19

        if data1[i] == 16:  # code_3rd.all_list[0][0] == 0:       # ㄱ ={0 0 0 1 0 0}
            res1[i][0]=0
        if data1[i] == 16 and res1[i - 1][0] == 19: # ㄲ ={0 0 0 0 0 1} {0 0 0 1 0 0} #OR24
            res1[i][0] =1
        if data1[i] == 48:  # code_3rd.all_list[0][0] == 2:       # ㄴ ={1 0 0 1 0 0}
            res1[i][0]=2
        if data1[i] == 24:  # ㄷ ={0 1 0 1 0 0}
            res1[i][0]=3
        if data1[i] == 24 and res1[i-1][0]==19:   # ㄸ ={0 0 0 0 0 1} {0 1 0 1 0 0}
            res1[i][0] =4
        if data1[i] == 4:  # ㄹ ={0 1 0 1 0 0}
            res1[i][0]=5
        if data1[i] == 36:  # ㅁ ={1 0 0 0 1 0}
            res1[i][0]=6
        if data1[i] == 20:  # ㅂ ={0 0 0 1 1 0}
            res1[i][0]=7
        if data1[i] == 20 and res1[i-1][0]==19:# ㅃ ={0 0 0 0 0 1} {0 0 0 1 1 0}
            res1[i][0] =8
        if data1[i] == 1:  # ㅅ ={0 0 0 0 0 1}
            res1[i][0]=9
        if data1[i] == 1 and res1[i-1][0]==19:  # ㅆ ={0 0 0 0 0 1} {0 0 0 0 0 1}
            res1[i][0] =10
        if data1[i] == 60:  # ㅇ ={1 1 0 1 1 0}(첫소리에 오면 생략가능?)
            res1[i][0] =11
        if data1[i] == 17:  # ㅈ ={0 0 0 1 0 1}
            res1[i][0]=12
        if data1[i] == 17 and res1[i-1][0]==19:  # ㅉ ={0 0 0 0 0 1} {0 0 0 1 0 1}
            res1[i][0]=13
        if data1[i] == 5:  # ㅊ ={0 0 0 0 1 1}
            res1[i][0]=14
        if data1[i] == 56:  # ㅋ ={1 1 0 1 0 0}
            res1[i][0]=15
        if data1[i] == 44:  # ㅌ ={1 1 0 0 1 0}
            res1[i][0]=16
        if data1[i] == 52:  # ㅍ ={1 0 0 1 1 0}
            res1[i][0]=17
        if data1[i] == 28:  # ㅎ ={0 1 0 1 1 0}
            res1[i][0]=18

        ############################중성##############################################
        if data1[i] == 41:  # ㅏ ={1 1 0 0 0 1}
            res1[i][1] =0
        if data1[i] == 46:  # ㅐ ={1 1 1 0 1 0}
            res1[i][1] =1
        if data1[i] == 22:  # ㅑ ={0 0 1 1 1 0}
            res1[i][1] =2
        if res1[i-1][1] ==2 and data1[i] == 46:  # ㅒ ={0 0 1 1 1 0} {1 1 1 0 1 0}
            res1[i-1][1] = 21
            res1[i][1] =3
        if data1[i] == 26:  # ㅓ ={0 1 1 1 0 0}
            res1[i][1] =4
        if data1[i] == 54:  # ㅔ ={1 0 1 1 1 0}
            res1[i][1] =5
        if data1[i] == 37:  # ㅕ ={1 0 0 0 1 1}
            res1[i][1] =6
        if data1[i] == 18  :  # ㅖ ={0 0 1 1 0 0}
            res1[i][1] =7
        if data1[i] == 35:  # ㅗ ={1 0 1 0 0 1}
            res1[i][1] =8
        if data1[i] == 43:  # ㅘ ={1 1 1 0 0 1}
            res1[i][1] =9
        if res1[i-1][1] ==9 and data1[i] == 46:  # ㅙ ={1 1 1 0 0 1} {1 1 1 0 1 0}
            res1[i - 1][1] = 21
            res1[i][1] =10
        if data1[i] == 55:  # ㅚ ={1 0 1 1 1 1}
            res1[i][1] =11
        if data1[i] == 19:  # ㅛ ={0 0 1 1 0 1}
            res1[i][1] =12
        if data1[i] == 50:  # ㅜ ={1 0 1 1 0 0}
            res1[i][1] =13
        if data1[i] == 58:  # ㅝ ={1 1 1 1 0 0}
            res1[i][1] =14
        if res1[i-1][1] ==14 and data1[i] == 46:  # ㅞ ={1 1 1 1 0 0} {1 1 1 0 1 0}
            res1[i - 1][1] = 21
            res1[i][1] = 15
        if res1[i-1][1] ==13 and data1[i] == 46:  # ㅟ ={1 0 1 1 0 0} {1 1 1 0 1 0}
            res1[i - 1][1] = 21
            res1[i][1] = 16
        if data1[i] == 49:  # ㅠ ={1 0 0 1 0 1}
            res1[i][1] =17
        if data1[i] == 25:  # ㅡ ={0 1 0 1 0 1}
            res1[i][1] =18
        if data1[i] == 29:  # ㅢ ={0 1 0 1 1 1}
            res1[i][1] =19
        if data1[i] == 38:  # ㅣ ={1 0 1 0 1 0}
            res1[i][1] =20

    ##########################종성##################################################

        if data1[i] ==32:  # ㄱ
            res1[i][2] =1
        if res1[i-1][2] ==1 and data1[i] == 32:   #ㄲ
            res1[i-1][2] =28
            res1[i][2] =2
        if res1[i-1][2] ==19 and data1[i] == 32:   #ㄳ
            res1[i-1][2] = 28
            res1[i][2] =3
        if data1[i] ==12:  # ㄴ
            res1[i][2] =4
        if res1[i-1][2] ==22 and data1[i] == 12:  #ㄵ
            res1[i-1][2] = 28
            res1[i][2] =5
        if res1[i-1][2] ==27 and data1[i] == 12:  #ㄶ
            res1[i - 1][2] = 28
            res1[i][2] =6
        if data1[i] ==6:  # ㄷ
            res1[i][2] =7
        if data1[i] ==8:  # ㄹ
            res1[i][2] =8
        if res1[i-1][2] ==1 and data1[i] == 8:  #ㄺ
            res1[i - 1][2] = 28
            res1[i][2] =9
        if res1[i-1][2] ==16 and data1[i] == 8: #ㄻ
            res1[i - 1][2] = 28
            res1[i][2] =10
        if res1[i-1][2] ==17 and data1[i] == 8: #ㄼ
            res1[i - 1][2] = 28
            res1[i][2] =11
        if res1[i-1][2] ==19 and data1[i] == 8:# ㄽ
            res1[i - 1][2] = 28
            res1[i][2] =12
        if res1[i-1][2] ==25 and data1[i] == 8:#ㄾ
            res1[i - 1][2] = 28
            res1[i][2] =13
        if res1[i-1][2] ==26 and data1[i] == 8:#ㄿ
            res1[i - 1][2] = 28
            res1[i][2] =14
        if res1[i-1][2] ==27 and data1[i] == 8:#ㅀ
            res1[i - 1][2] = 28
            res1[i][2] =15
        if data1[i] ==9:  # ㅁ
            res1[i][2] =16
        if data1[i] ==40:  # ㅂ
            res1[i][2] =17
        if data1[i] ==2:  # ㅅ
            res1[i][2] =19
        if data1[i] ==18:  # ㅆ
            res1[i][2] =20
        if data1[i] ==15:  # ㅇ
            res1[i][2] =21
        if data1[i] ==34:  # ㅈ
            res1[i][2] =22
        if data1[i] ==10:  # ㅊ
            res1[i][2] =23
        if data1[i] ==14:  # ㅋ
            res1[i][2] =24
        if data1[i] ==11:  # ㅌ
            res1[i][2] =25
        if data1[i] ==13 and data1[i+1] !=0:  # ㅍ
            res1[i][2] =26
        if data1[i] ==7:  # ㅎ
            res1[i][2] =27
    ##################################약자#################################
        if data1[i] ==57:
            res1[i][0] = 0
            res1[i][1] = 0
            # 가

        if data1[i] ==48:
            res1[i][0] = 2
            res1[i][1] = 0
            # 나

        if data1[i] ==24:
            res1[i][0] = 3
            res1[i][1] = 0
            # 다

        if data1[i] ==36:
            res1[i][0] = 6
            res1[i][1] = 0
            #마

        if data1[i] ==20:
            res1[i][0] = 7
            res1[i][1] = 0
            # 바

        if data1[i] ==42:
            res1[i][0] = 9
            res1[i][1] = 0
            # 사

        if data1[i] ==17:
            res1[i][0] = 12
            res1[i][1] = 0
            # 자

        if data1[i] == 56:
            res1[i][0] =15
            res1[i][1] = 0
            # 카

        if data1[i] == 44:
            res1[i][0] =16
            res1[i][1] = 0
            # 타

        if data1[i] == 52:
            res1[i][0] =17
            res1[i][1] = 0
            # 파

        if data1[i] == 28:
            res1[i][0] =18
            res1[i][1] = 0
            # 하

        if data1[i] == 53:
            res1[i][0] =11
            res1[i][1] = 4
            res1[i][2]=1
            # 억

        if data1[i] == 31:
            res1[i][0] =11
            res1[i][1] = 4
            res1[i][2]=4
            # 언

        if data1[i] == 30:
            res1[i][0] =11
            res1[i][1] =4
            res1[i][2] =8
            # 얼

        if data1[i] == 33:
            res1[i][0] =11
            res1[i][1] = 6
            res1[i][2]=4
            # 연

        if data1[i] == 45:
            res1[i][0] =11
            res1[i][1] =6
            res1[i][2]=8
            # 열

        if data1[i] == 61:
            res1[i][0] =11
            res1[i][1] = 6
            res1[i][2]=21
            # 영

        if data1[i] == 51:
            res1[i][0] =11
            res1[i][1] = 8
            res1[i][2]=1
            # 옥

        if data1[i] == 47:
            res1[i][0] =11
            res1[i][1] = 8
            res1[i][2]=4
            # 온

        if data1[i] == 63:
            res1[i][0] =11
            res1[i][1] = 8
            res1[i][2]=21
            # 옹

        if data1[i] == 60:
            res1[i][0] =11
            res1[i][1] =13
            res1[i][2] =4
            # 운

        if data1[i] == 59:
            res1[i][0] =11
            res1[i][1] = 13
            res1[i][2]=8
            # 울

        if data1[i] == 39:
            res1[i][0] =11
            res1[i][1] = 18
            res1[i][2]=4
            # 은

        if data1[i] == 27:
            res1[i][0] =11
            res1[i][1] = 18
            res1[i][2]=8
            # 을

        if data1[i] == 62:
            res1[i][0] =11
            res1[i][1] = 20
            res1[i][2]=4
            # 인

        if data1[i-1] == 21 and data1[i] == 26:
            res1[i][0] =0
            res1[i][1] =4
            res1[i][2]=19
            # 것  ############   21확인필요

        #res1[i][0]=cho_uni_1.pop()
        #res1[i][1]=joong_uni.pop()
        #jong_uni.pop()


    print(res1)
    res3=[]
    cho,joong,jong=0,0,0
    #i=3
    for i in range(len(data1)):
        if data1[i] == 0:
            res2=" "
            #print(res2)
            res3.append(res2)
            continue

        if res1[i][0] != -1 and res1[i][3]!=1: ##초성이 존재(i)
            cho=res1[i][0]
            if cho==19:   ## 초성-된소리

                continue
            if res1[i][1] == -1:  ##중성이 없을때(i)

                if res1[i+1][0] == -1: ##초성이 없을때(i+1)
                    joong=res1[i+1][1]
                    if res1[i+2][2] != -1 and res1[i+2][0]==-1:  #종성이 있을때(i+2)
                        jong=res1[i+2][2]
                    else:                   #종성이 없을때(i+2)
                        jong=0

            elif res1[i+1][0] ==-1 and res1[i+1][1] !=-1 and res1[i][2] ==-1:  ##중성이 있을떄(i),(i+1의 중성이 있을때)(i+1의 초성이 없을때),(i의 종성이없을때)

                joong = res1[i + 1][1]

                if res1[i+2][2] != -1 and res1[i+2][0]==-1:#종성이 있을때(i+2)
                    jong = res1[i + 2][2]
                    #res1[i][3] = 1
                    res1[i + 2][1]=-1###  (ㅖ와 받침ㅆ이 중복이기에 중복제거)
                else:                                 #종성이 없을때(i+2)
                    jong=0
            else :                          ##중성이 있을때(i),(i+1의 초성이 있을때?),(i+1의 중성이 있을때?)
                if res1[i-1][0] !=-1  and res1[i-1][2] ==-1 and res1[i][2]!=-1:
                    cho=res1[i-1][0]

                    res3.pop()

                joong=res1[i][1]

                if res1[i][2]!=-1 :  ##종성이 있을떄(i)
                    jong = res1[i][2]
                    res1[i][3]=1
                elif  res1[i+1][0]==-1 and res1[i+1][1]==-1 and res1[i+1][2]!=-1 :##종성이 없고(i),(i+1)이 있을때
                    jong = res1[i+1][2]
                    #res1[i][3] = 1
                else:
                    jong = 0


        if  res1[i-1][0] == -1 and res1[i][1] != -1 and res1[i][0] == -1 and res1[i][3]!=1 : ##초성이 ㅇ으로 없는경우(글자초반)
            cho=11
            joong = res1[i][1]
            if res1[i+1][2] != -1 and res1[i + 1][0] == -1:  # 종성이 있을때(i+1)
                jong =res1[i+1][2]
                res1[i + 1][3]=1
            else:  # 종성이 없을때(i)
                jong = 0
                res1[i][3]=1

        if res1[i-1][3]==1 and res1[i][0] == -1 and res1[i][1] != -1: ##초성이 ㅇ으로 없는경우(글자중간)
            cho = 11
            joong = res1[i][1]
            if res1[i+1][2] != -1 and res1[i + 1][0] == -1:  # 종성이 있을때(i+1)
                jong =res1[i+1][2]
                res1[i + 1][3]=1
            else:  # 종성이 없을때(i)
                jong = 0
                res1[i][3]=1

        if i>0 and i<len(data1):                ## 숫자일경우
            if data1[i-1]==0 and data1[i+1]==0 :
                n=number(data1,i)
                if n !=0:
                    res3.append(n)
                #print(n)
                    continue

        if res1[i][0] == -1 and cho!=11:        ##초성이 없고 ㅇ(11)이 아닌경우 제외
            continue


        res2 =chr(cho * 588 + joong * 28 + jong + 44032)

        res3.append(res2)
        #print(res2,"data:",data1[i],"번호:",i)
        cho=0

    #print("res3:",res3)

    text_representation = ''.join(map(str, res3))

    print(text_representation)
    tts = gTTS(text=text_representation, lang='ko', slow=False)

    # 변환된 음성을 파일로 저장
    tts.save("output.mp3")

    # 저장된 음성 파일을 재생
    #os.system("start output.mp3")
    return res3


#res4=trans_data(data12)
# text_representation = ''.join(map(str, res4))
#
# print(text_representation)
# tts = gTTS(text=text_representation, lang='ko', slow=False)
#
# # 변환된 음성을 파일로 저장
# tts.save("output.mp3")
#
# # 저장된 음성 파일을 재생
# os.system("start output.mp3")
#print(res4)










'''

    if code_3rd.all_list[3][1] == 0:  # 그래서
        32,26
    if code_3rd.all_list[3][1] == 1:  # 그러나
        32,48
    if code_3rd.all_list[3][1] == 2:  # 그러면
        32,12
    if code_3rd.all_list[3][1] == 3:  # 그러므로
        32,9
    if code_3rd.all_list[3][1] == 4:  # 그런데
        32,54
    if code_3rd.all_list[3][1] == 5:  # 그리고
        32,35
    if code_3rd.all_list[3][1] == 6:  # 그리하여
        32,37

def sign():
    if code_3rd.all_list[2][1] == 32:  # ' ' 공백
        0

    if code_3rd.all_list[2][1] == 33:  # ' ! '
        14
    if code_3rd.all_list[2][1] == 40:  # ' ( '
        11,2
    if code_3rd.all_list[2][1] == 41:  # ' ) '
       1,7

    if code_3rd.all_list[2][1] == 44:  # ' , '
       4

    if code_3rd.all_list[2][1] == 46:  # ' . '
       13
    if code_3rd.all_list[2][1] == 47:  # ' / '
        21,18

    if code_3rd.all_list[2][1] == 58:  # ' : '
        4,8

    if code_3rd.all_list[2][1] == 59:  # ' ; '
        5,10

    if code_3rd.all_list[2][1] == 63:  # ' ? '
        11

    if code_3rd.all_list[2][1] == 126:  # ' ~ '
        8

    if code_3rd.all_list[2][1] == 34:  # ' " '
        8

    if code_3rd.all_list[2][1] == 39:  # ' ' '
        8
'''
