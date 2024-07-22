# Unit1
## Probability Theory
1. general situation
    1. observed variables: the state of the world that agent knew
    2. unobserved variable: the state that agent needs to reason about
    3. model: relation between observed variable and unobserved variable
    4. model is probabilistic
    5. for example:
        1. sensors detect a red object: observed variables
        2. where is the object ant what is it: unobserved variable
        3. some method to find out it is an apple: model
        4. there is no 100% guarantee it is an apple: usually model is probabilistic
2. Probability Space
    1. probability space is a triplet (Ω, B, P)
        1. Ω: all possible outcomes.
            1. e.g., tossing a coin: head and tail are sample space Ω
        2. B: σ-algebra/Borel field, a collection of events to consider
            1. event: a subset of Ω
            2. σ-algebra是指 Ω子集合的集合
        3. P: probability, a measure defined on B
            1. P(A) ≧ 0 for all A ∈ B
            2. P(Ω)
            3. if A1,A2, ... ∈ B are "pairwise disjoint兩兩不相交(沒有重疊)", "P(∪ Ai) = ∑P(Ai) A的"union並集" 等於A的概率之和"
### 為了得到有用的模型, 我們會需要轉換許多不同的機率分佈, 以下為不同轉換的介紹
3. Conditional Probability
    1. H∈B, P(H)>0
    2. P(A|H) = "P(A ∩ H)/P(H) 機率A和H的交集除以機率H(在H的機率中發生A的機率)"
    3. P(A|H) is the "conditional probability of B, given H 給定/發生H的情況下, A發生的條件機率/機率"
4. The Total Probability Rule
    1. {Hj} are "pairwise disjoint event in B是B中多個兩兩不相交的事件"
    2. "P(A) = ∑P(A ∩ H) A的機率是A和H並集機率的和"
    3. "P(A) = ∑P(A|H)P(H) A的機率是發生H的情況下, A發生的機率乘上H的機率"
5. The Product Rule
    1. conditional distributions to joint distributions
    2. P(x ∩ y) = P(x, y)
    3. P(y)P(x|y) = P(x ∩ y)
    4. P(x|y) = P(x ∩ y)/P(y)
6. The Chain Rule
    1. base on The Product Rule
    2. P(x1, x2, x3) = P(x1)P(x2|x1)P(x3|x1,x2)
7. Bayes' Theorem
    1. the most equation in AI, it is the foundation of many systems
    2. it lets us "build one conditional from its reverse 建構反向的機率條件"
    3. P(A|B) = P(B|A)P(A)/P(B)
8. Independence of Events
    1. if P(A,B) = P(A)P(B), A and B are independent
    2. it lets us Splits the joint distribution into a product of two simple ones
    3. usually variables are not independent but we can use it to simplify modelling assumption
    4. Example:
        1. P(toothache, "cavity蛀牙", "catch檢測到蛀牙")
        2. P(+catch| +toothache, +cavity) = P(+catch| -cavity)
            1. 在蛀牙的條件下檢測到蛀牙的機率
            2. 以上假設我檢測到蛀牙的機率和牙痛無關, 只和我有無蛀牙有關
        3. P(+catch| +toothache, -cavity) = P(+catch| -cavity)
            1. 在沒蛀牙的條件下檢測到蛀牙的機率
            2. 以上假設我檢測到蛀牙的機率和牙痛無關, 只和我有無蛀牙有關
        4. Catch is conditionally independent of Toothache given cavity
            1. P(catch|toothache, cavity) = P(catch|cavity)
            2. 可以說給定"蛀牙"時"檢測到蛀牙"的機率有條件獨立於"牙痛"


# Unit2
## Bayesian Network
1. graphical models
    1. provide an "intuitive直觀的" way to visualize the relationship
    2. relationship does not imply "causality因果關係"
2. Bayesian Network
    1. a "directed acyclic graph有向無環圖"
    2. node represent random variables
        1. assigned(observed) variables
        2. unassigned(unobserved) variable
    3. "directed edges有向邊" represent "immediate dependence直接依賴關係"
    4. independence
        1. x2 <- x1 -> x3
        2. P(x1,x2,x3)=P(x1)P(x2|x1)P(x3|x1)
        3. in BN, each variable x is independent of all its "non-descendants非後代" given its parents
3. "D-separation有向分離"
    1. lets us analyze complex cases in terms of simple structures
    2. Active/ inactive path
        1. a path is active if each triple on the path is active
        2. Causal Chains: X -> Z -> Y, not given Z, path is not blocked, active path,  not guaranteed X independent of Y
        3. Common Cause: x <- Z -> Y, not given Z, path is not blocked, active path,  not guaranteed X independent of Y
        4. Common Effect: x -> Z <- Y, given Z, path is not blocked, active path,  not guaranteed X independent of Y


# Unit3
## "Inference推理" in Bayesian Network