README:

-> Prerequisites: 
1. OpenCV
2. NVCC (Cuda framework 12.1+)

-> How to use:
1. Make sure you have opencv_world490.dll file in the folder you are running
2. Save a image as image.png (this is the image you want to blur) in the folder
3. Run blur.exe

NOTE: if any changes are made to main.cpp you will have to rebuild in x64 Native Tools for Developers terminal
  ->  nvcc main.cpp gaussian.cu -o blur -I"(OpenCV_LOCATION\opencv\build\include" -L"D:\OpenCV_Location\build\x64\vc16\lib" -lopencv_world490
<br>
![image](https://github.com/user-attachments/assets/01653147-4702-41e2-8438-a4cb0d0c427c)

Blurred output when sigma = 1.0f
<br>
<img width="287" height="176" alt="blurred" src="https://github.com/user-attachments/assets/3bb2455e-2a9e-46b9-ae39-5501558963ce" />
