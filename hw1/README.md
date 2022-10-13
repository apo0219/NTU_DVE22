# \[DVE\] project #1: High Dynamic Range Imaging

### Dependency

Our project is implemt in Python3, with package opencv-python 4.5.5.64, numpy.

```
$ pip install opencv-python
$ pip install numpy
```

### Usage

1. First you need to make a directory named `raw_image` (under the same directory with our python code) and put your .ppm image under it.

```
$ mkdir raw_image
```

2. Run the hdr_reconstruction.py

```
$ python3 hdr_reconstruction.py 
# Please put the file under the folder "raw_image" !
# Give the list of file name separated with "," (ex: fileA.ppm, fileB.ppm):
A.ppm, B.ppm, C.ppm, D.ppm
# Give the list of shutter speed separated with "," (ex: 1, 0.5, 0.25, 0.125):
1/2, 1/4, 1/8, 1/16
```

(Bonus)

3. Run the globle-tone-mapping.py

```
$ python3 globle-tone-mapping.py
# Give the file name: 
image.hdr
```



### Some note

1. Our implement of MTB algorithm have a process with 7 rounds, each round it will compress both height and width of image to 1/2. So you have to make sure that the size of image have to greater than $2^7 \cdot 2^7$ , or it might occur some error. Ofcourse you can also change the number in MTB.py line 78 and line 84 to a lower depth.
2. Our raw image needs the file format of .ppm, you can use the packages likes [dcraw](https://github.com/ncruces/dcraw) to change the orignal photo to this format.

# Report
## Data preprocess
* **Raw data**
    * camara : Canon PowerShot SX50 HS
    * file tpye : CR2
    * ISO : ISO-80
    * Aperture : f/8
    * size : 4000 $\times$ 2664
    * shutter speed : 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256
* **CR2 to PPM**
    we use DCRAW ( https://github.com/ncruces/dcraw ) to covert cr2 to ppm.
    `./dcraw -4 -w file_path.cr2`
    * -4
It generates a linear 16-bit file instead of an 8-bit gamma corrected file which is the default.
    * -w
    If DCRAW manages to find it, it will use the white balance that was adjusted in the camera at shooting.
* below is the demonstration of photos by jpg transfromed from CR2.
  we use online transformer, iloveimg, here. ( https://reurl.cc/nEGk16 )
  Barrel distortion has been fix by iloveimg automatically.
![](https://i.imgur.com/wxGwyBb.jpg)

## MTB
To deal with the image alignment issue, we implement the MTP algorithm with python.

We compress the image to 7 size, and run the process to estimate the best offset 7 times from the smallest image to the biggest one. In each process we chosse the best offset by the method below:
1. We first change the image to grayscale with function : $Y = (54R+183G+19B)/256$
($Y$ represent the light intensity of the pixel, and the red, green, blue of each pixels is denoted as $R,G,B$)
2. Then, we calculate $\bar{Y}$ (the average of $Y$). And turn the image to binary by comparison with $\bar{Y}$ (If greater than $\bar{Y}$ we give it 0, else we give it 1).
3. Also we compute the mask to filter out the pixel with intenseti in $\bar{Y}$ +- 5%.
4. Then we evaluate the score of each offset by the function $\sum_{x,y} (\text{p1}[x,y] \oplus \text{p2_offset}[x,y] \& \text{mask}[x,y])$, and choose offset with the min vlue of score.

After each rounds, the offset will change to 2 times and be add as the initialize offset of the next round.

![](https://i.imgur.com/WQtIHnX.jpg)

![](https://i.imgur.com/ScRQFtp.jpg)

![](https://i.imgur.com/dEXDH6u.jpg)

## HDR
Recover HDR image by method taught on class. Minimize the square error between the best fit line and points.
![](https://i.imgur.com/6wrWYKX.png)
$$
arg\ \min_{E_i} Err(E_i) = \sum^p_{j=1}(X_{ij}-E_i \times \Delta t_j)^2\\
0 = \frac{\partial Err}{\partial E_i} = \sum^p_{j=1}-2 \Delta t_j(X_{ij}-E_i \times \Delta t_j)\\
\sum^p_{j=1} \Delta t_j X_{ij} = E_i\sum^p_{j=1} (\Delta t_j)^2\\
E_i = \frac{\sum^p_{j=1} \Delta t_j X_{ij}}{\sum^p_{j=1} (\Delta t_j)^2}
$$
## tone mapping
* **default global operator**
$$
\begin{align}
&(x, y) : \text{ the posision of a pixel}\\
&c : \text{ the color channel } (r, g, b)\\
&\bar{L_w}(c)& &=& &exp\left(\frac{1}{N} \sum_{x,y} log(\delta+L_w(x,y, c))\right)\\
&L_m(x,y,c)& &=& &\frac{\alpha}{\bar{L_w}} L_w(x,y,c)\\
&L_d(x,y,c)&&=& &\frac{L_m(x,y,c)}{l+L_m(x,y,c)}
\end{align}
$$
picture of $\delta = 10^{-6}, \alpha = 0.09$
![](https://imgur.com/WFIx1Kc.png)
* **barrel distortion**
    To fix the barrel distortion, we take the method on **stackoverflow**(https://reurl.cc/dXZ5qg).
    Use cv2.undistort().
![](https://imgur.com/JUu8KPs.png)
* $L_{white}$
    Add $L_{white}$ in $L_d$, making picture brighter and cut off the light brighter than $L_{white}$.
    $$
    L_d(x, y, c) = \frac{L_m(x,y, c)\left(1+\frac{L_m(x,y,c)}{L^2_{white}}\right)}{1+L_m(x, y, c)}
    $$
picture of $L_{white} = 1.5,\ \delta = 10^{-6},\ \alpha = 0.09$
![](https://imgur.com/oN6wPeQ.png)
* $color$
    To fix the color imbalance, we give a Coefficient to each color.
    $$
    \begin{align}
    &L_d'(x. y, g) &=& &0.75 \times L_d(x,y,g)\\
    &L_d'(x, y, r) &=& &1.1\times L_d(x,y,r)\\
    &L_d'(x, y, b) &=& &1 \times L_d(x,y,b)\\
    \end{align}
    $$
picture of modifing the color
![](https://imgur.com/Sf41r07.png)
* $beta$
    The Luminous intensity contrast is still too big that we lose lots of detail. Therefore, we modify the$L_m$ by adding the $\beta$:
    $$
    \begin{align}
    L_m(x, y, c) &= \frac{\alpha}{\bar{L}_w} L_w(x,y,c) + \beta\\
    &= \left(\frac{\alpha}{\bar{L}_w} + \frac{\beta}{L_w(x,y,c)}\right) L_w(x,y,c)\\
    &= coif \times L_w(x,y,c)
    \end{align}\\
    \text{The smaller } L_w(x,y,c) \text{ comes the bigger } coif\\
    $$
picture of $beta = 0.3,\ L_{white} = 1.5,\ \delta = 10^{-6},\ \alpha = 0.09$
![](https://imgur.com/2PyE2Vz.png)
* **Final global operator**
$$
\begin{align}
&\bar{L_w}(c)& &=& &exp\left(\frac{1}{N} \sum_{x,y} log(\delta+L_w(x,y, c))\right)\\
&L_m(x,y,c)& &=& &\frac{\alpha}{\bar{L_w}} L_w(x,y,c) + \beta\\
&L_d(x,y,g)&&=& &0.75\times\frac{L_m(x,y,g)}{l+L_m(x,y,g)}\\
&L_d(x,y,r)&&=& &1.1\times\frac{L_m(x,y,r)}{l+L_m(x,y,r)}\\
&L_d(x,y,b)&&=& &1\times\frac{L_m(x,y,b)}{l+L_m(x,y,b)}\\
\end{align}
$$
* **Compare to cv2.createTonemapDurand()**
Our tone mapping (left), cv2.creatTonemapDurand() (right)
![](https://i.imgur.com/8FevsZI.jpg)

    Compare to the cv2, we still have a lot to do on detail. The roof, word on the wall, and the shadow of doors on our picture lose most of their color and light. 
    Because we modify the color manually, the colors may still have some implanced. cv2 doesn.t have to do that so it's more rubust. 
