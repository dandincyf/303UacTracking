# -*- coding: UTF-8 -*-
import math
import matplotlib.pyplot as plt
f = open("B:\Document\Code\\586\KCFTest\\testVideo\kcfwithactiveframe.txt")
kcfline = f.readline()             # 调用文件的 readline()方法
kcflinelist=[]
kcfcorex=[]
kcfcorey=[]
while kcfline:
    kcfline=kcfline.replace("\n","")
    kcfline=kcfline.replace("[","")
    kcfline=kcfline.replace("]","")
    kcfline=kcfline.replace(" ","")
    kcfline = kcfline.split(",")
    # print(line)
    kcflinelist.append(kcfline)
    kcfline = f.readline()
# print(len(linelist))
for i in range(len(kcflinelist)):
    if(kcflinelist[i]!=""):
        # print(i)
        # print(int(linelist[i][0]))
        CoreX=int(kcflinelist[i][0])+int(kcflinelist[i][2])/2
        CoreY=int(kcflinelist[i][1])+int(kcflinelist[i][3])/2
        # print(CoreY)
        # print(CoreY)
        kcfcorex.append(CoreX)
        kcfcorey.append(CoreY)
f.close()
# for i in range(len(kcfcore)):
#     print(kcfcore[i])
# print(len(kcfcore))
'''
'''
f = open("B:\Document\Code\\586\KCFTest\\testVideo\domp4_groundtruth_rect.txt")
dogline = f.readline()             # 调用文件的 readline()方法
doglinelist=[]
dogcorex=[]
dogcorey=[]
while dogline:
    dogline=dogline.replace("\n","")
    # kcfline=kcfline.replace("[","")
    # kcfline=kcfline.replace("]","")
    dogline=dogline.replace("	",",")
    dogline = dogline.split(",")
    # print(dogline)
    doglinelist.append(dogline)
    dogline = f.readline()
print(len(doglinelist))
for i in range(len(doglinelist)):
    if(doglinelist[i]!=""):
        # print(i)
        # print(int(linelist[i][0]))
        CoreX=int(doglinelist[i][0])+int(doglinelist[i][2])/2
        CoreY=int(doglinelist[i][1])+int(doglinelist[i][3])/2
        # print(CoreY)
        # print(CoreY)
        dogcorex.append(CoreX)
        dogcorey.append(CoreY)
f.close()
f = open("B:\Document\Code\\586\KCFTest\\testVideo\ohjesus.txt")
kalmanline = f.readline()             # 调用文件的 readline()方法
kalmanlinelist=[]
kalmancorex=[]
kalmancorey=[]
while kalmanline:
    kalmanline=kalmanline.replace("\n","")
    kalmanline=kalmanline.replace("[","")
    kalmanline=kalmanline.replace("]","")
    kalmanline=kalmanline.replace("'","")
    # kalmanline=kalmanline.replace("	",",")
    kalmanline = kalmanline.split(",")
    print(kalmanline)
    kalmanlinelist.append(kalmanline)
    kalmanline = f.readline()
# print(len(kalmanlinelist))
for i in range(len(kalmanlinelist)):
    if(kalmanlinelist[i]!=""):
        if(kalmanlinelist[i][0]=="0."):
            kalmanlinelist[i][0]=0
        if(kalmanlinelist[i][1] == "0."):
            kalmanlinelist[i][1]=0
        kalmancorex.append(kalmanlinelist[i][0])
        kalmancorey.append(kalmanlinelist[i][1])
f.close()




# for i in range(len(dogcore)):
    # print(dogcore[i])
# print(len(dogcore))

calclisty=[]
calclistx=[]
anverage=0
for i in range(20,len(kcfcorex)):
#     # print(kcfcore[i*2])
    divide=math.sqrt(math.pow(float(kcfcorex[i])-dogcorex[i],2)+math.pow(float(kcfcorey[i])-dogcorey[i],2))
    anverage+=divide
    # print(i)
    # print(len(kcfcorex))
    print(divide)
    calclistx.append(i)
    calclisty.append(divide)
anverage/=len(kcfcorex)
anverage=round(anverage,2)
print("hsdfhdghfig")
print(anverage)
titleStr="KCF Record:Central Location Error/Average value:"+str(anverage)
plt.title(titleStr)
plt.plot(calclistx, calclisty)
plt.xlabel('frames')
plt.ylabel('CLE')
plt.legend()
plt.show()
plt.savefig("ohmygod")


# for i in range(20,len(kalmancorex)-5):
# #     # print(kcfcore[i*2])
#     divide=math.sqrt(math.pow(float(kalmancorex[i])-dogcorex[i+8],2)+math.pow(float(kalmancorey[i])-dogcorey[i+8],2))
#     anverage+=divide
#     # print(i)
#     # print(len(kcfcorex))
#     print(divide)
#     calclistx.append(i)
#     calclisty.append(divide)
# anverage/=len(kalmancorex)
# anverage=round(anverage,2)
# print("hsdfhdghfig")
# print(anverage)
# titleStr="Kalman Predice:Central Location Error/Average value:"+str(anverage)
# plt.title(titleStr)
# plt.plot(calclistx, calclisty)
# plt.xlabel('frames')
# plt.ylabel('CLE')
# plt.legend()
# plt.savefig("ohmygod")

# file_write_obj = open("B:\Document\Code\\586\KCFTest\\testVideo\Bird1\\OTB100_bird_groundtruth_rect_core.txt", 'w')
# for i in range(len(Strok)):
#     file_write_obj.writelines(Strok[i])
#     file_write_obj.write('\n')
#     print(Strok[i])






