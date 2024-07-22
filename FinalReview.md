# Module1
## Knowledge Check
1. 人們在"1970-1990"對AI的發展感到悲觀
- In "1970-1990" people are "pessimistic感到悲觀" about the future of artificial intelligence
2. 目標識別任務是："輸入圖像, 輸出分類"
- an object recognition task: "input an image and output a classification". supervised learning.
3. 最常見的ML是："監督學習"
- most common category of machine learning: "Supervised Learning"
4. 非監督學習是：學習"數據集的空間關係"
- unsupervised learning: "Learning spatial空間的 relations of a dataset"
5. 適合由專家編寫而非使用ML： "最短路徑規劃"
- expert programs a specific solution: "Shortest Path Planning"
6. ***爬山算法是屬於："迭代優化算法"
- hill-climbing method: "An iterative-optimization algorithm"
7. 兩樣本中之間的關係在邊換維度後：距離可能有所不同 
- the distance between two samples in a reduced dimension space on the manifold as compared to the original space: 
- "The distances will likely differ"
8. 監督學習："透過示範"學習 
- teach a model in supervised learning: "By demonstration"
9. 量子計算的特點："計算容量指數成長"，"零時通信"
- key features of quantum computing: "computing capacity grow exponentially指數成長"; "zero communication time"
10. ***需要最少量子比特的數據嵌入方法："振幅编码"
- data embedding method requires the fewest number of qubits: "Amplitude encoding"
11. ontology本體論用於："表示知識"
- "representing knowledge"
12. 沒有閉合解可以使用："迭代优化"
- "Iterative optimization"
13. 學習率不夠大會導致："收斂慢"
- "Slow convergence"
14. 梯度下降中的步長依照："學習率"縮放
- "The learning rate"
15. 貧瘠高原是："傳統ML和量子ML"的問題
16. 強化學習的基礎是："操作条件反射"(條件制約)
- "Operant conditioning"
17. 學習率不夠小會造成："超過最小值"
- "Overshooting minima"
18. 機器學習的主要目標："實現高度泛化"
- "Achieving high generalization"
19. 強化學習的重點是："自主學習的智能體"
- "An agent leaning autonomously"
20. 量子比特的最終輸出是："0或1"
21. 監督學習數據集包含："輸入和預期輸出"
- "Inputs and desired outputs"
22. 強化學習的學習方式是："獎勵和懲罰"
- "Reward and punishment"
23. 梯度下降更新參數的方式是："朝最陡下降方向取步長"
- "By taking a step in the direction of the steepest descent"
24. 無監督學習的示例："聚類"
- "Clustering"
25. 監督學習數據的標籤是由："人類專家準備"
- "Labels are prepared by a human expert"
## Quiz
1. ML和AL涵蓋主題："基本相同"
- "mostly the same topics"
2. 理性思考和向人類一樣思考是衝突的AI要求：正確
3. 使用梯度下降法："總是朝著最抖的方向前進"
- "Gradient descent will always head in the direction of steepest descent"
- 此題目包含梯度下降的其他敘述
4. ***了解數據的基礎分佈後可以："由一個特徵預測另一個特徵的概率"
- "Predict the probability of one feature, given another"
5. 使用線性回歸建模需要："斜率和y軸截距"
- "Slope and y-intercept"
6. 強化學習的主要挑戰是："設計一個恰當的獎勵函數"
- "Engineering a good reward function"
7. 輸出為離散輸出的監督學習任務："分類"
- "Classification"
8. 無監督學習："只有輸入數據沒有輸出標籤"
- "only have input data but not output label data"
9. Ansatz架構意味著：設計神經網路的結構和層次
- "designing the structure and layers of the neural network"
10. ***最高時間複雜性的數據鑲嵌方法："振幅編碼"
- "Amplitude encoding"


# Module2
## Knowledge Check
1. 人工神經網路最早提出於："1940"
2. 人類大腦的神經元有約："860億個"
- "86 billion"
3. 1986誕生某總算法使人們重新對人工神經網路感興趣："反向傳播"
- "Backpropagation"
4. 線性感知器和：單個神經元相似
- "A single neuron"
5. 哪個元素提供非線性功能：激活函數
- "The activation function"
6. 激活函數的輸出是："連續的"
- "The output is continuous"
7. 典型神經網路的要求是："網路中的神經元連接都有權重參數"
- "There is a weight parameter for every neuron connection in the network"
8. 輸入向量通過模型到產出輸出的過程稱作："正向傳遞"
- "Forward pass"
9. 課程中機器人的神經網路輸出是："預測是否碰撞"
- "A prediction on whether or not a collision will occur"
10. 課程中機器人的訓練數據是收集於："隨機探索環境並紀錄"
- By randomly exploring the environment and recording sensor readings
- 稱為"wandering漫遊"
11. 反向傳播的作用是："確定參數更新以提高擬合度"
- "To determine parameter updates to increase a model’s fit"
12. 反向傳播計算梯度需要："輸入和目標輸出數據集"
- "A dataset of inputs and target outputs"
13. 神經網路的最後一層的激活函數輸出等於："網路的輸出"
- "The output of the network"
14. 訓練過程中正向傳播需要在："反向傳播之前"
- "Before every backward propagation"
15. 防止特定特徵出現壓倒性需要："對特徵進行規範化"
- "Normalize the features"
16. 訓練數據集的擬合改善："並不能對測試數據集進行任何保證"
- "This does not guarantee anything for the testing dataset"
17. pytorch自定義神經網路使用："torch.nn.Module"
18. torch.nn.Linear是："一個與其他完全連接的層"
- "Any fully connected layer to another"
19. pytorch中的優化器是用於："計算梯度確定如何更新參數"
- "The optimizer determines how to update our parameters based on the calculated gradient"
20. pytorch中反向傳播使用："loss.backward()"
## Quiz
1. 人工神經網路的主要挑戰是："難以分析和調試"
- "Artificial neural networks are difficult to analyze and debug"
2. 線性感知器的輸出等於："輸入和權重的點積(內積)"
- "The dot product of the inputs with the connection weights"
3. 生物神經元的閾值對應線性感知器的："篇置"
- "The bias"
4. 最接近生物神經元的激活方法是："閾值激活"
- "Threshold activation"
5. 使用非線性激活函數的目的是因為："沒有非線性激活，任意線性組合等效於單個線性組合"
- "Without nonlinear activations, any number of linear combination layers can be performed as a single linear combination"
6. 神經網路的特性是："一般每個神經元都有一個篇置"
- "There is a bias parameter for every neuron"
7. 正向傳遞是由："線性函數和激活函數組成"
- "A forward pass consists of basic linear algebra and activation functions"
8. 收集到的數據應劃分為："較大的訓練數據和較小的測試數據"
- "The collected data should be split into a large training dataset and a smaller testing dataset"
9. 衡量學習模型的質量時："在具代表性分佈的未知數聚衡量損失很重要"
- " It is important to measure the loss on a representative distribution of unseen data"
10. 反向傳播用來計算嵌套函數的篇導數微積分公式是："鏈式法則"
- "Chain rule"
11. 如果神經網路的每一層是一個函數，整個網路是："每一層函數的嵌套函數"
- "A nested function of the layer functions"
12. 更新模型時損失函數的更新是根據："網路中的所有函數"
- "All parameters in the network"
13. 計算損失函數對於網路中任何參數的變化會因為："參數的深度"
- "The depth of the parameter"
14. 訓練數據和測試數據的話分："80:20"
15. 防止過擬合的方法有："早停法"


# Module3
## Knowledge Check
1. 有輸入和輸出數據集的建模："監督學習"
2. 提取無標籤的動物生理數據集的隱藏結構可以使用："層次聚類"
- "Hierarchical Clustering"
3. 強化學習給於獎勵應在："整個過程中"
- "Rewards should be given throughout"
4. 每個樣本都進行更新的梯度下降法："隨機梯度下降法"
- "Stochastic Gradient Descent"
5. 神經元權重參數為(1,-1)，學習率0.01，梯度(-1,0.09)：更新後的權重為(權重參數)-(梯度*學習率)
- (1.01,-1.0009)
6. 計算激活函數的輸入：((輸入向量*權重)+偏置)
7. 如果不對輸入數據規範化，會導致："學習質量下降"
- "Quality of learning decreases"
8. softmax計算：搜尋"softmax calculator"
9. 客戶分組可以使用："無監督學習"
10. 機器人學習行走可以使用："強化學習"
## Quiz
1. 動量隨機梯度下降法："可以防止過早收斂於局部最優"
- "It prevents premature convergence"
2. 避免直接使用差作為誤差是：因為"正負會抵消"
- "Because positive and negative values will cancel out"
3. 圖像分類的最後一層通常是："softmax層"
4. softmax計算：搜尋"softmax calculator"
5. 反向傳播的運作是："計算輸出層的損失，鏈式法則有助於傳播到所有層"
- "The loss is calculated for the output layer only. The chain rule helps to backpropogate the loss through all layers"
6. 假設一個隨機變量概率分佈，現在有一個新的觀察值，更新分佈可以使用："貝葉斯法則"
- "Bayes’ Rule"
7. MNIST數據集訓練，驗證有無過度擬合可以使用"K折交叉驗證"
- "Use k-fold cross-validation"
8. 假如神經網路沒有激活函數："每一層的輸出會是正負無限"
9. 偏差是用來：控制激活域值
- "Bias controls the activation threshold"
10. Adam的特徵是："考慮了曲率"
- "It takes the curvature into account"


# Module4
## Knowledge Check
1. 給定圖像建模使用："CNN"
2. 訓練模型時權重沒有變化，可能是因為："梯度的消失"
- "Vanishing Gradients"
3. 為避免梯度指數下降(梯度消失)，可以使用："ReLU激活函數"
- "Use a ReLu activation function"
4. 會影響LSTM單元狀態的門有："遺忘門和輸入門"
- "Forget and Input gate"
5. LSTM的門控單元添加於："層中的每個神經元"
- "To every neuron in the layer"
6. 蒙特卡羅丟棄法的丟棄是在："推理時完成"
- "In Monte Carlo, dropout is also done at inference"
7. 卷積層會："在圖像得像素上滑動"
- "It runs kernels over pixels of the image"
8. 卷積計算
9. 有零填充的卷積計算
10. 最大池化計算
## Quiz
1. 翻譯RNN的梯度成指數成長，應使用："帶梯度剪裁的神經網路"
- "Use a Neural Network with gradient clipping"
2. 在LSTM中，遺忘門的保留使用："sigmoid激活"
- "By using Sigmoid activation"
3. 生成對抗網路的組成是："生成網路和鑑別網路"
- "Generator and Discriminator Network"
4. 丟棄可以使用於神經網路的："輸入層和隱藏層"
- "Dropout can be applied on the input and hidden layers of the neural network"
5. 最大池化計算
6. 有零填充的最大池化計算
7. 卷積計算
8. 有零填充的卷積計算
9. CNN中的扁平化是指："特徵的植被堆疊成一個向量，透過全聯接網路"
- "Each value within the feature maps is stacked into a vector to be passed through a fully connected network"
10. 圖像描述生成的目標是："生成文本性的圖像描述"


# Module5
## Knowledge Check
1. 世界座標包含："現實世界中固定的三維座標"
- "The world coordinate frame contains 3D coordinates fixed in the real world"
2. 投影的特徵包含："線投影仍是線"
- "Lines are projected to lines"
3. 立體相機可以："計算目標距離"
- "Calculate the distance to a target"
4. Sfm運動結構是："利用運動信號從二維圖像估計三維結構的技術"
- "SfM is a technique for estimating 3D structures from 2D images with motion signals"
5. 使用窗口匹配估計深度時，窗口越大："圖的視差更平滑(信息越多)，細節更少"
- "The maps will have smoother disparity and less detail"
6. 亮度恆定公式的特徵是："(假設)給定點在每一幀看起來都相同"
- "The projection of a given point looks the same in every frame"
7. SIFT尺度不變特徵變換會："估計兩圖像素間的相似度，並匹配對應特徵"
- "SIFT estimates the similarity of pixels among two images and matches the corresponding features"
8. 語義分割和實例分割之間最大的區別是："語義分割會為圖片中的每一個像素分配標籤"
- "Semantic segmentation assigns a categorical label to each pixel in an image"
9. 計算機視覺："在圖像，視頻，和自然語言等取得了重大發展"
- "Computer vision techniques have made great progress in various data types, like image, video, and natural languages"
10. 在感知任務中："相機校準允許用戶設置相機內部特徵"
- "Camera calibration allows users to set the camera's internal characteristics"
## Quiz
1. 消防員透過煙霧識別熱區可以使用："熱像儀"
2. 立體相機能夠："測量到目標的距離"
3. Sfm運動結構是："利用運動信號從二維圖像估計三維結構的技術"
4. 深度圖如果細節減少，可能是因為："增大窗口"
- "A larger window size"
5. 雙目立體深度估計："輸入必須來自已知點的兩個圖像"
- "The input to calibrate the binocular stereo must be two images from various known points"
6. SfM運動結構可以："從多個相機重建世界座標戲中的目標"
- "SfM restores and reconstructs targets from multiple cameras in the world coordinate system"
7. CNN："可以看作是一中基於特徵的方法"
- "CNN extracts the features from images directly and can be viewed as feature-based approach"
8. 預測目標位置可以使用："目標檢測"
- "Object Detection"
9. 圖像字幕生成結合了："計算機視覺和機器翻譯"
- "image captioning combines recent advances in computer vision and machine translation"
10. 目標分類使用了："CNN"


# Module6
## Knowledge Check
1. 貝葉斯網路用於："對不確定性建模"
- "It is used to model uncertainty"
2. 貝葉斯網路可以："使用簡單的局部分佈來描述複雜的聯合分佈"
- "It describes complex joint distributions using simple, local distributions"
3. 貝葉斯網路的因子分解是根據："其網路拓樸"
- "Bayesian networks factorize according to its network topology"
4. 貝葉斯網路中的枚舉推理："首先要在聯合概率分佈中，找度與證據相符的概率條目"
- "Inference by enumeration starts with finding the probability entries in the joint probability distribution that align with the evidence"
5. MCMC抽樣方法使用隨機遊走採樣，使得："MCMC比似然加權更好的代表了採樣分佈"
- "The MCMC method creates samples that are more representative of the sampled distribution than likelihood weighting"
6. 在似然加權中："每個樣本的權重是單獨計算的"
- "In likelihood weighting, the weights are computed independently for each sample"
7. DBN深度信念網路中："粒子過濾是一種近似推理方法"
- "Particle filter is an approximate inference method for DBNs"
8. 確定獨立性問題：參照筆記
9. 已確定是件的概率為1
10. 
## Quiz
1. 確定獨立性問題：參照筆記
2. 確定獨立性問題：參照筆記
3. 機器人移動問題
    1. 已知X1=(1,1), 求X2=(4,4)的機率
    2. 機器人在時間2移動到(4,4)只有可能是使用能力, 使用能力的機率為0.5
    3. 場上總共有10*10格, 傳送到其中一格的機率為0.01
    4. 機率為0.005
4. ***機器人移動問題
    1. 已知E1=1, E2=2, X1=(1,1), Xt=(E,c), 求X2=(2,1)
    2. 因為已知E2=2, X2會在第二行的機率為1
    3. 
5. 貝葉斯網路機率問題
    1. P(A)=0.2, P(B)=0.7 , P(C|A,B)=0.9
    2. 0.2*0.7\*0.9=0.126
6. ***貝葉斯機率問題
7. 在V型結構的貝葉斯網路中："觀察到相同結果，就會影響傳遞"
- " In a V structure, when the common effect is observed, the influence is passed"
8. 在貝葉斯網路的過濾中："重新採樣是指，用當前樣本重新創建一組有不同權重的新樣本"
- "In filtering, resampling takes the current samples and recreates a new set of samples that is the same but has different weights"
9. 在拒絕抽樣中："和證據不一致的樣本會被拒絕"
- "In rejection sampling, any samples that are inconsistent with the evidence are rejected"
10. 貝葉斯網路是："一種圖模型"
- "It is a type of graphical model"


# Module7
## Knowledge Check
1. 要達到目標必須經過的landmark特徵點
2. 如果要使用atomic原子表達，最小狀態數會是n^(p+d+t)
3. 7城市, 12車, 11包裹, 卡車可以在任意城市 包裹可以在任意城市或是卡車中, 使用因子表卡車和包裹位置需要23個變量
4. 同Q3
5. 馬爾可夫決策：附近可到達的位置，最優策略就是避免到達終止狀態
6. 馬爾可夫決策：附近可到達的位置，最優策略就是避免到達終止狀態，可能狀態值均為0
7. 馬爾可夫決策：附近可到達的位置，最優策略就是避免到達終止狀態，可能狀態值等於SUM(機率*折扣因子\*值)
8. 馬爾可夫決策：已知實際嘗試的決策，估計值為所有決策分數的平均
9. 同Q9
10. 同Q9
## Quiz
1. PDDL
2. PDDL
3. 同Knowledge Check，Q2
4. 同Knowledge Check，Q3
5. PDDL
6. 同Knowledge Check，Q5
7. 如果下一步獎勵均相同，可以計算再下一步
8.
9.
10.
