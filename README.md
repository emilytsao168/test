# 1112 PDL Project:看圖算數學
---
本專案利用簡單的Convolutional Neural Network來實作辨識圖片中的數字及運算符號，並且根據辨識出的數字及運算符號進行運算。
---
## 輸入資料格式
### 圖片
給定⼀張圖⼤⼩是200\*60像素，辨識圖中的數字(0~9)及運算⼦(加+/減-/乘\*)，共9個字元，輸出數學運算結果(整數)。
      
![p0](https://github.com/emilytsao168/test/assets/117272534/1ba33342-8f46-4319-8dba-818b411a2798)
### 標籤     
由「train_data01.csv」中匯入每張圖片的標籤，其中的格式為「0	6 0 4 5 4 - 5 + 8 =	60457」
---
## 降噪
透過dealImg(path)進行圖片的降噪。      

![下載](https://github.com/emilytsao168/test/assets/117272534/dbaa8a33-6630-4dae-bcd7-58fd2af85026)      

```python     
def dealImg(path):
    # 獲取圖片img物件
    img = getImg(path)
    # 將彩色圖片轉換為灰度圖片
    grayImg = toGrayImg(img)
    # 將灰度圖片二值化
    binImg = toBinImg(grayImg)
    noiseImg = binImg
    # 對圖片進行降噪
    noiseImg = noiseReduction(noiseImg)
    return noiseImg
``` 

