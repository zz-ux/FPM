import cv2 as cv
import numpy as np
import time
'''将Lena频谱图按照N*N的圆心定位截取，并记录图像Lena在不同子孔径下的图像，并按照FPM算法复原图像'''

def Create_Low_Images (Image,N,Bias,Radius) :
    '''模拟FPM过程，生成低频图像'''
    '''
    Image:输入的高频图像(np格式，二维数组)
    N：平板上一列（行）点光源的数量(int)
    Bias:频谱上子孔径圆心的偏移尺度(float)
    Radius：频谱上子孔径半径大小(float)
    '''
    # 首先子孔径判断有没有超出图像边长范围
    if(((Bias*(N-1))+2*Radius)>len(Image[0])  or  ((Bias*(N-1))+2*Radius)>len(Image) ):
        return "输入的子孔径超过图像边缘，重新输入"

    # 对输入图像归一化
    Image = np.array(Image ,dtype=np.float64)
    Image = (Image-np.min(Image))/(np.max(Image)-np.min(Image))
    Center_Image=0.5*np.array([float(len(Image)),float(len(Image[0]))])
    # 计算输入图像的傅里叶变换
    FFT_Image = np.fft.fftshift(np.fft.fft2(Image ))
    # 计算子孔径的中心坐标矩阵
    Center = np.zeros((N, N, 2))
    for row in range(len(Center)):
        for col in range(len(Center[0])):
            # Center[row][col][:]=np.array([ (256)-(4-row)*bias, (256)-(4-col)*bias ])
            Center[row][col][:] = np.array([ Center_Image[0]-( 0.5*(N-1)- row)*Bias ,
                                             Center_Image[1]-( 0.5*(N-1)- col)*Bias])
    del row,col,Center_Image
    Center=np.reshape(Center,(-1,2))
    # 进入傅里叶变换部分
    Inv_Images=[]  # 相应频域子孔径对应的空间域图像
    Template_total = np.zeros(np.shape(Image)) # 为求FPM理论能实现的最好图像设置
    for i in range(len(Center)):
        '''针对对应圆心制作子孔径模板'''
        Circle_center = Center[i]
        Template = np.zeros(np.shape(Image))
        horizen_vector = (np.arange(len(Template[0])) - Circle_center[1]) ** 2
        vertical_vector = np.reshape((np.arange(len(Template)) - Circle_center[0]) ** 2, (-1, 1))
        Template = horizen_vector + vertical_vector
        Template[Template <= (Radius ** 2)] = 1.0
        Template[Template > (Radius ** 2)] = 0.0
        Template_total = Template_total + Template
        ''' 根据模板计算Lena图像加窗限制的傅里叶变换 '''
        Template_Image = FFT_Image * Template
        Inv_Image = np.abs(np.fft.ifft2(np.fft.fftshift(Template_Image)))
        Inv_Images.append(np.abs(Inv_Image ))
    '''计算理想的复原图像'''
    Template_total[Template_total > 1] = 1.0
    Ideal_Image = np.abs(np.fft.ifft2(np.fft.fftshift(FFT_Image* Template_total)))
    del Circle_center, FFT_Image, Inv_Image, Template, Template_Image, horizen_vector, vertical_vector , i , Template_total
    Inv_Images = np.array(Inv_Images)
    # '''至此只有：
    # 子孔径中心位置——Center,对应子孔径形成的图像序列——Center_Inv_Lena,根据模板形成的理想复原像——Ideal_Image三个参数存在
    # 通过这3个参数复原高分辨率图像'''
    return (Center,Inv_Images,Ideal_Image )

if __name__=="__main__":
    Lena=np.array(cv.imread("C:\\Users\\Administrator\\Desktop\\Image super-resolution\\program\\Lena.jpg",0),
              dtype=np.float64)
    (Center,Inv_Images,Ideal_Image )=Create_Low_Images( Image=Lena,N=5,Bias=30,Radius=40)
    cv.imshow("Ideal_Image",Ideal_Image)
    for i in range(len(Inv_Images)):
        print(i,100*((np.linalg.norm(Inv_Images[i,:,:]))/np.linalg.norm(Ideal_Image)))
        cv.imshow("Inv_Images",np.sqrt(Inv_Images[i,:,:]))
        cv.waitKey(200)
    # cv.imshow("Inv_Images", np.sqrt(Inv_Images[12, :, :]))
    print('done')
    cv.waitKey(0)
