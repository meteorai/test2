# Requirements

- cuda > 7.5
- cudnn
- opencv <= 3.4.0

    
# Install

You can download the code:
```
    git clone https://github.com/4Dager/4Ddetect.git
```
 
Pre-trained model can be downloaded from:https://pan.baidu.com/s/1dFjtA57ZUg4a5HNT_VOSAA
    
# Usage


- **When you compile the code in Windows**:

Open build\darknet\darknet.sln, set x64 and Release, then build with Microsoft Visual Studio 2015. Then open build\darknet\x64 and perform the following in cmd:
```
    4D_detect data\car.jpg
```

Then you can get the predictions of the data\car.jpg.


- **When you compile the code in Linux**:

Open the directory in the terminal, make the code and compile with the following step:
```
    make
    ./4D_detect data/car.jpg
```
Then you can get the predictions of the data/car.jpg.
