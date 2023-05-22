# 1112 PDL Project:看圖算數學
---
本專案利用簡單的Convolutional Neural Network來實作辨識圖片中的數字及運算符號，並且根據辨識出的數字及運算符號進行運算。
---
## 輸入資料格式
### 圖片
給定⼀張圖⼤⼩是200\*60像素，辨識圖中的數字(0~9)及運算⼦(加+/減-/乘\*)，共9個字元，輸出數學運算結果(整數)。

### 標籤     

由「train_data01.csv」中匯入每張圖片的標籤，其中的格式為「0	6 0 4 5 4 - 5 + 8 =	60457」

---
## 資料預處理
### 圖片降噪
透過dealImg(path)進行圖片的降噪。      
   

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
### 標籤處理
原本從csv檔中讀到的str標籤「0	6 0 4 5 4 - 5 + 8 =	60457」     
將標籤中的運算符號 (加+/減-/乘\*) 分別轉換為 (10, 11, 12)     
並將所有標籤儲存成[6, 0, 4, 5, 4, 11, 5, 10, 8]格式     
最後將標籤轉換為 one-hot 編碼      

### 將圖片和標籤打包
將圖片和標籤打包成一個元組，並且儲存為「data_50000_v1.pkl」     
資料預處理完成！

---
## 讀入資料並建立模型
### 讀入資料並進行處理
將前面經過預處理的圖片及標籤讀入，並且將資料以8:2的比例切割為訓練資料及測試資料(共50000筆資料)。     
因為我們的模型是多輸出的結構，為了符合之後模型的輸入格式，需要對於標籤形狀進行修改。     
     
原本的標籤格式：     
```
[[第一張第1個數字,...,第一張第9個數字], [第二張第1個數字,...,第二張第9個數字], ... , [...]]
```
修改後的標籤格式：     
```
[[第一張第1個數字,...,最後一張第1個數字], [第一張第2個數字,...,最後一張第2個數字], ... , [...]]
```
經過處理後輸入模型的資料形狀：     
```
訓練資料維度：(40000, 60, 200, 1)
訓練標籤維度：(9, 40000, 13)
```
---
## 建立模型並訓練     
模型的建立：
```
#圖片大小
imgshape = (60, 200, 1)
#字元類別數(0-9以及+-\*)
nchar = 13

def createmodel():
    img = layers.Input(shape=imgshape)
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)
    conv3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(mp2)
    mp3 = layers.MaxPooling2D(padding='same')(conv3)
    conv4 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(mp3)
    bn = layers.BatchNormalization()(conv4)
    mp3 = layers.MaxPooling2D(padding='same')(bn)
    
    flat = layers.Flatten()(mp3)

    outs = []
    for _ in range(9):
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.2)(dens1)
        res = layers.Dense(nchar, activation='softmax')(drop)

        outs.append(res)
    
    model = Model(img, outs)
    model.compile(loss='categorical_crossentropy', optimizer='adamax',metrics=["accuracy"])
    return model
```
最後來看看model的summary輸出長甚麼樣子：
```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 60, 200, 1)  0           []                               
                                ]                                                                 
                                                                                                  
 conv2d (Conv2D)                (None, 60, 200, 16)  160         ['input_1[0][0]']                
                                                                                                  
 max_pooling2d (MaxPooling2D)   (None, 30, 100, 16)  0           ['conv2d[0][0]']                 
                                                                                                  
 conv2d_1 (Conv2D)              (None, 30, 100, 32)  4640        ['max_pooling2d[0][0]']          
                                                                                                  
 max_pooling2d_1 (MaxPooling2D)  (None, 15, 50, 32)  0           ['conv2d_1[0][0]']               
                                                                                                  
 conv2d_2 (Conv2D)              (None, 15, 50, 64)   18496       ['max_pooling2d_1[0][0]']        
                                                                                                  
 max_pooling2d_2 (MaxPooling2D)  (None, 8, 25, 64)   0           ['conv2d_2[0][0]']               
                                                                                                  
 conv2d_3 (Conv2D)              (None, 8, 25, 128)   73856       ['max_pooling2d_2[0][0]']        
                                                                                                  
 batch_normalization (BatchNorm  (None, 8, 25, 128)  512         ['conv2d_3[0][0]']               
 alization)                                                                                       
                                                                                                  
 max_pooling2d_3 (MaxPooling2D)  (None, 4, 13, 128)  0           ['batch_normalization[0][0]']    
                                                                                                  
 flatten (Flatten)              (None, 6656)         0           ['max_pooling2d_3[0][0]']        
                                                                                                  
 dense (Dense)                  (None, 64)           426048      ['flatten[0][0]']                
                                                                                                  
 dense_2 (Dense)                (None, 64)           426048      ['flatten[0][0]']                
                                                                                                  
 dense_4 (Dense)                (None, 64)           426048      ['flatten[0][0]']                
                                                                                                  
 dense_6 (Dense)                (None, 64)           426048      ['flatten[0][0]']                
                                                                                                  
 dense_8 (Dense)                (None, 64)           426048      ['flatten[0][0]']                
                                                                                                  
 dense_10 (Dense)               (None, 64)           426048      ['flatten[0][0]']                
                                                                                                  
 dense_12 (Dense)               (None, 64)           426048      ['flatten[0][0]']                
                                                                                                  
 dense_14 (Dense)               (None, 64)           426048      ['flatten[0][0]']                
                                                                                                  
 dense_16 (Dense)               (None, 64)           426048      ['flatten[0][0]']                
                                                                                                  
 dropout (Dropout)              (None, 64)           0           ['dense[0][0]']                  
                                                                                                  
 dropout_1 (Dropout)            (None, 64)           0           ['dense_2[0][0]']                
                                                                                                  
 dropout_2 (Dropout)            (None, 64)           0           ['dense_4[0][0]']                
                                                                                                  
 dropout_3 (Dropout)            (None, 64)           0           ['dense_6[0][0]']                
                                                                                                  
 dropout_4 (Dropout)            (None, 64)           0           ['dense_8[0][0]']                
                                                                                                  
 dropout_5 (Dropout)            (None, 64)           0           ['dense_10[0][0]']               
                                                                                                  
 dropout_6 (Dropout)            (None, 64)           0           ['dense_12[0][0]']               
                                                                                                  
 dropout_7 (Dropout)            (None, 64)           0           ['dense_14[0][0]']               
                                                                                                  
 dropout_8 (Dropout)            (None, 64)           0           ['dense_16[0][0]']               
                                                                                                  
 dense_1 (Dense)                (None, 13)           845         ['dropout[0][0]']                
                                                                                                  
 dense_3 (Dense)                (None, 13)           845         ['dropout_1[0][0]']              
                                                                                                  
 dense_5 (Dense)                (None, 13)           845         ['dropout_2[0][0]']              
                                                                                                  
 dense_7 (Dense)                (None, 13)           845         ['dropout_3[0][0]']              
                                                                                                  
 dense_9 (Dense)                (None, 13)           845         ['dropout_4[0][0]']              
                                                                                                  
 dense_11 (Dense)               (None, 13)           845         ['dropout_5[0][0]']              
                                                                                                  
 dense_13 (Dense)               (None, 13)           845         ['dropout_6[0][0]']              
                                                                                                  
 dense_15 (Dense)               (None, 13)           845         ['dropout_7[0][0]']              
                                                                                                  
 dense_17 (Dense)               (None, 13)           845         ['dropout_8[0][0]']              
                                                                                                  
==================================================================================================
Total params: 3,939,701
Trainable params: 3,939,445
Non-trainable params: 256
__________________________________________________________________________________________________

```
訓練模型：
```python
hist = model.fit(train_feature, train_label, batch_size=128, epochs=100, verbose=2, validation_data=(test_feature, test_label), validation_split=0.2)
```
模型訓練準確度：
```python
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.plot(hist.history['dense_3_accuracy'], label='accuracy')
plt.plot(hist.history['val_dense_3_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
```

---
## 預測數字並進行計算     
載入先前訓練好的模型並且進行預測。     
透過calculate_expression(nums,count)進行結果的運算。     
最後將運算結果存入csv檔中。     
```python
import sys
# 設置整數轉換的最大位數
sys.set_int_max_str_digits(5000000) 

def calculate_expression(nums,count):
    try:
      expression = ""
      for num in nums:
          if num == 10:
              expression += "+"
          elif num == 11:
              expression += "-"
          elif num == 12:
              expression += "*"
          else:
              expression += str(num)
      
      result = eval(expression)
      return result

    except Exception as e:
      #print(f'讀取檔案時發生錯誤: {str(e)}')
      print("#",count,nums)
      return 0
```


