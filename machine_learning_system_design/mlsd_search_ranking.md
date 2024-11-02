# Search Ranking System 

### 1. Problem Formulation
* Clarifying questions
    - Is it a generalized search engine (like google) or specialized (like amazon product)?
      - We will assume that you are working towards finding relevant results like amazon product.
    - What is the primary (business) objective of the search system?
      - Provides the most relevant results for a search query by ranking them in order of relevance.
    - What are the specific use cases and scenarios where it will be applied?
      - People search products through search bar by entering query.
    - What are the system requirements (such as response time, accuracy, scalability, and integration with existing systems or platforms)?
      - We will assume that you have billions of documents to search from, and the search engine is getting around 10K queries per second (QPS).
    - What is the expected scale of the system in terms of data and user interactions?
      - We will assume that you have billions of documents to search from, and the search engine is getting around 10K queries per second (QPS).
    - Is their any data available? What format? 
      - Yep, we have search history.
    - Personalized?
      - You will assume that the user is logged in, and you have access to their profile as well as their historical search data.
    - How many languages needs to be supported?
      - So far only English.
    - What types of items (products) are available on the platform, and what attributes are associated with them?
    - What are the common user search behaviors and patterns? Do users frequently use filters, sort options, or advanced search features?
    - Are there specific search-related challenges unique to the use case (e-commerce)? such as handling product availability, pricing, and customer reviews?

    
* Use case(s) and business goal
  * Use case: user enters text query into search box, system shows the most relevant items (products) 
  * business goal: increase CTR, conversion rate, Successful session rate, etc  
* Requirements
  * response time, accuracy, scalability (50M DAU)
* Constraints
  * budget limitations, hardware limitations, or legal and privacy constraints
* Data: sources and availability
  * Sources:
    * Search history query and relevant products order
* ML formulation: 
  * ML Objective: retrieve items that are most relevant to a text query  
    * we can define relevance as weighted summary of click, successful session, conversion, etc. 
  * ML I/O: I: text query from a user, O: ranked list of most relevant items on an e-commerce platform  
  * ML category: MM input search system -> retrieval and ranking 
    * retrieval: Query input -> document selection (similarity, binary classification) 
    * ranking: MM input -> multi-label classification (click, success, convert, add to cart, save for later, etc.)
    * we can use a multitask classification
   
### 2. Metrics  
- Offline
  - retrieval:
    - recall only care about FN error, don't care about FP error
  - stage one rank:
    - AUC-ROC: classification metric
    - recall@k: top-k TP / all positive
    - mAP: overall quality of recommendations or search results across a diverse set of queries. mAP works well for binary relevance
    - precision, precision@k: top-k TP / top-k TP + top-k FP. Apply if only one stage rank
    - all above metric not consider output order. If relevant output model A is place 15, model B is place 50, both recall@10 are 0, but model A is better than model B.
  - stage two rank:
    - precision@k: same as above
    - MRR: only care about first relevant product place. MRR is often used in search and recommendation systems to assess how quickly users find relevant content. It's particularly useful when there is only one correct answer or when the order of results matters.
    - NDCG: nDCG is beneficial when relevance is not binary (i.e., there are degrees of relevance), and you want to account for the diminishing importance of items lower in the ranking. Best for stage two. One caveat is nDCG does not penalize irrelevant search results, assign negative relevance score to the irrelevant document.
- Online
  - CTR: problem: doesn't track relevancy, clickbaits, For example, this might include short clicks where the searcher only looked at the resultant document and clicked back immediately. You could solve this issue by filtering your data to only successful clicks, i.e., to only consider clicks that have a long dwell time.
  - success session rate: dwell time > T or add to cart / number of total sessions (means click after seen)
  - total dwell time 
  - conversion rate

### 3. Architectural Components
* Multi-layer architecture 
  * Query Rewriting -> Query Understanding -> Candidate generation (document selection, retrieval) -> stage 1 Ranker -> stage 2 Ranker -> Blender(reranker) -> Filter -> SERP (search engine result page)
* Query Rewriting
  * spell checker: fix basic spelling mistakes
  * query normalization: 
  * query expansion (e.g. add alternative word like food to restaurant) / relaxation (e.g. remove "good")
* Query understanding
  * Intent/Domain classification
* User provide information
  * search criteria (tags, category)
* Candidate generation (document selection, retrieval)
  * focus on recall, millions/billions into 10k/1k, care about speed instead of accuracy/relevant
* Ranking 
  * ML/NN based
  * multi-stage ranker:
    * if more than 10k items to select from or QPS > 10k  
      * 100k items: stage 1 (liner model, pointwise) -> 1000/500 items: stage 2 (DNN model, pairwise) -> 500 items
    * no more than 10k items
      * 1k/500 items: stage 1 (DNN model, pairwise) -> 500 items
* Blender:
  * outputs a SERP (search engine result page)
  * blends results from multiple sources e.g. textual (inverted index, semantic) search, visual search, etc. may not suit for e-commence search. 

#### Retrieval
* from 100 B to 100k, need fast no accuracy
* IR: compares query text with document text 
* Document types: 
  * item (product) title 
  * item description 
  * item reviews 
  * item category 
* No ML method:
  * fast and have third part service, problem is no text sematic match
  * inverted index (e.g. elastic search): 
    * index DS, mapping from words into their locations in a set of documents (e.g. ABC -> documents 1, 7)
  * after query expansion (e.g. black pants into black and pants or suit-pants or trousers etc.), do a search in inverted index db and find relevant items with relevance score 
  * relevance score 
    * weighted linear combination of: 
      * terms match (e.g. TF-IDF score)(e.g. w = 0.5), 
      * item popularity (e.g. no of reviews, or bought) (e.g. w=0.125), 
      * intent match score (e.g. 0.125), 
      * domain match score (e.g. 0.125)
      * personalization score (e.g. age, gender, location, interests, e.g. 0.125) 
* ML method:
  * solve text sematic match issue, but cost lot time and money
  * DSSM Two tower embedding architecture (item(product) description and text_query encoders)
  * train:
    * one text query with n documents (1 positive and n-1 negative), apply softmax
  * inference:
    * input text query -> embedding -> ANN system -> top 100k document

#### Ranking: 
* see the model sections.

### 4. Data Collection and Preparation
* Data sources: 
  - Users 
  - Queries 
  - Items (products)
  - Context 
  - User - Items (products)
  - Queries - Items (products)

* Labeling: 
  - use online user engagement data to generate positive, negative labels and assign score for order

* Data storage (engineer part)
  - User, queries and items information: BQ, cloud storage bucket, elastic search db etc
  - Interaction information: history same as above, recently use cache

* Data generation pipeline (engineer part)
  - Data ingestion
  - Feature generation including user products information and interaction information.
  - Feature transformation like standardized some features or embedding some features.
  - Label generation, create two stages label.

### 5. Feature Engineering
* Feature selection and preprocessing
  * User: 
    * ID, username -> embedding in 5-10 dimension
    * Demographics:
      * age -> bucketize into categorical then one hot
      * gender -> one hot
      * language, city, country -> embedding
    * User interaction history
      * click rate, purchase rate   -> keep original
      * avg purchase product rating -> keep original
      * avg purchase product price  -> keep original
      * avg visited product after purchase -> scale
    * User interests
      * recent top three categories -> one hot and concat
    * User purchase history
      * avg all products features in below item features
    * User save as favourite history
      * avg all products features in below item features
    * User add to cart history
      * avg all products features in below item features
    * User history search query and product
      * calculate query with each attribute (title, tags, description) TF-IDF score and avg all scores in past fix interval
  * Context: 
    * device -> one hot
    * time of the day -> one hot
    * previous queries intent / domain
      * categories -> one hot
  * Query features: 
    * query historical engagement
      * top frequency category like sports, from other users -> one hot
    * query intent / domain
      * categories -> one hot
    * query embeddings (can be use in later as similarity score)
      * short key word, use Word2Vec, since no context, need fast and light method
  * Item (product) features 
    * Title embeddings (can be use in later as similarity score)
      * use BERT here, title are in sentence usually, context aware helps, and BERT first token contains sentence meaning
    * Description embeddings (can be use in later as similarity score)
      * use BERT here, title are in phrases usually, context aware helps, and BERT first token contains sentence meaning
    * Reviews data (can be use in later as similarity score)
      * avg reviews number -> scale
      * no of reviews -> one hot
      * review textual data embeddings -> same as description, then aggregate all reviews embedding into one embedding
    * category -> one hot
    * engagement radius, product shipping radius, local or global -> one hot
    * rating point -> keep original
    * avg shipping day -> keep original
    * number of sales in past month -> scale
    * how many days this product sales -> scale
    * price
  * User-Item(product) features 
    * distance for shipment -> scale
    * product historical engagement by the user
      * product purchase before or not -> one hot
      * product click before or not -> one hot
      * product add to cart before or not -> one hot
      * product save for favorite before or not -> one hot
      * how many days from last time purchase/visited -> scale
  * Query-Item(product) features
    * text match or not
      * title -> one hot
      * description -> one hot
      * category -> one hot
    * uni-gram or bi-gram search
      * title -> TF-IDF score
      * description -> TF-IDF score
      * tags -> TF-IDF score
    * embedding similarity score between query and title, description, reviews
      * BERT embedding dot product or cosim similarity
    * historical engagement
      * click rate of Item for that query -> keep original
      * purchase rate of Item for that query -> keep original

### 6. Model Development and Offline Evaluation
#### Ranking 

* Training Dataset
  * Pointwise approach 
    * positive samples: user engaged (e.g. click, spent time > T, add to cart, purchased), this depends on what we predict if it's CTR then click means engage, if's success session rate, then dwell > T means engage.
    * negative samples: no engagement by the user + random negative samples e.g. from pages 10 and beyond
    * 5 million Q/day -> one positive one negative sample from each query -> 10 million samples a day 
    * use a whole week's data at least to capture daily patterns
      * capturing and dealing with seasonal and holiday data
      * weekend and weekday user behavior may different which need whole week
    * train-valid/test split: 70/30 (of 70 million)
    * temporal affect: e.g. use 3 weeks data: first 2/3 of weeks: train, last week valid / test 
  * Pairwise approach:
    * ranks items according to their relative order, which is closer to the nature of ranking 
    * predict product scores in a way that minimizes number of inversions in the final ranked result
    * one query form different product pairs
    * Two options for train data generation for pointwise approach
      * human raters: each human rates 10 results per 100K queries * 10 humans = 10M examples
        * expensive, doesn't scale 
      * online engagement data 
        * assign scores to each engagement type e.g. 
          * impression with no click -> label/score 0 
          * click only -> score 1 
          * spent time after click > T : score 2 
          * add to cart : score 3 
          * purchase: score 4

* Model Selection  
  * Two options:
    * Pointwise LTR model: <user, item> -> relevance score 
      * approximate it as a binary classification problem probability (relevant)
      * fast but not consider items order when training, will output top 10k relevant items.
    * Pairwise LTR model: <user, item1, item2> -> item1 score > item2 score ?
      * loss function if the predicted order is correct 
      * more natural to ranking, but more complicated and cost time and money more
  * Multi - Stage ranking
    * 100k items (focus on recall) -> 500 items (focus on precision) -> 500 items in correct order   
    * Stage 1: We use a pointwise LTR -> binary classifier
      * latency: microseconds 
      * suggestion: 
        * basemodel: 
          * LR:
            * hyperparameters: L1/L2 regularization penalty lambda, n_iterations, learning rate, etc.
            * loss function: CE
            * regularization: L2/L1 to constraint too many features
            * pros: fast and easy to explain
            * cons: too simple can't get nonlinear feature
        * advance:
          * LGB + LR:
            * hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, learning_rate
            * loss function: CE
            * regularization: tree + L2/L1 to constraint too many features
            * pros: interpretable，non-linearity, help select feature first
            * cons: inefficient for continual training, can't train embedding layers 
          * Deep and cross network (DCN)
            * finds feature interactions automatically 
            * two parallel networks: deep network (learns complex features) and cross network (learns high order features interaction)
            * pros: can learn high order features combination, continuing learning, parallel training, automatic do feature interactions
            * cons: cross network only models certain feature interactions, which may negatively affect the performance of the cross network model. don't have low oder feature interaction which also been approved important.
          * Deep factorization machine (DeepFM)
            * combines a NN (for complex high order features) and an FM (for pairwise interactions low order)
            * pros: can learn both low and high order features combination, continuing learning, parallel training, automatic do feature interactions
            * cons: too complex, cost lot time and money especially in inference stage if we have too many input.
      * metric: ROC AUC, recall for two stages framework, precision for one stage framework 
    * Stage 2: Pairwise LTR model (learn relative order after first stage we have high relevant product output list)
      * Two options (choose based on train data availability and capacity):
        * basemodel:
          * GBDT/LGB + pairwise loss
          * inefficient for continual training, can't train embedding layers, too sparse features performance not good.
        * advance:
          * LambdaMART: a variation of MART, obj fcn changed to improve pairwise ranking
            * Tree-based algorithms, generalize effectively using a moderate set of training data. Few million examples training size will be the best choice to use in pairwise ranking in the second stage.
            * 适用于排序场景：不是传统的通过分类或者回归的方法求解排序问题，而是直接求解 
            * 损失函数可导：通过损失函数的转换，将类似于NDCG这种无法求导的IR评价指标转换成可以求导的函数，并且赋予了梯度的实际物理意义，数学解释非常漂亮 
            * 增量学习：由于每次训练可以在已有的模型上继续训练，因此适合于增量学习 
            * 组合特征：因为采用树模型，因此可以学到不同特征组合情况 
            * 特征选择：因为是基于MART模型，因此也具有MART的优势，可以学到每个特征的重要性，可以做特征选择 
            * 适用于正负样本比例失衡的数据：因为模型的训练对象具有不同label的文档pair，而不是预测每个文档的label，因此对正负样本比例失衡不敏感
          * LambdaRank: NN based model, pairwise loss (minimize inversions in ranking)
            * NN model, need large training dataset and training time, latency when inference time.
            * high order features combination, continuing learning, directly use embedding features to train, incorporating multi-modal data, automatically feature combination
  * Metric: NDCG

### 7. Prediction Service
- Data Prepare pipeline
  - static features (e.g. product features, user features) -> batch feature compute (daily, weekly) -> feature store in cloud
  - dynamic features: number of ad impressions, clicks.
- Prediction pipeline 
  - two stage (funnel) architecture 
    - candidate generation (document selection)
      - use no ML method, inverted index (e.g. elastic search)
      - use ML method, query embedding -> ANN -> top 100k products
    - ranking
      - stage one: features -> model -> click prob. -> sort top 500/10k products
      - stage two: features -> model -> relative order -> top 500 products
      - re-ranking: see below
- Re-ranking pipeline
  - business level logic and policies
    - filtering inappropriate items 
    - diversity (exploration/exploitation)
    - add ads products
  - Two ways: 
    - rule based filters and aggregators 
    - ML model 
      - Binary Classification (P(inappropriate))
      - Data sources: human raters, user feedback (report, review)
      - Features: same as product features in ranker
      - Models: LR, MART, or DNN (depending on data size, capacity, experiments)
      - More details on harmful content classification
- Continual learning pipeline
  - keep update dynamic features weekly/daily
  - fine tune on new data, eval, and deploy if improves metrics
  - research on add new features and do offline experiment at same time

### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release (engineer part)
  - deploy platform depends on company ML infrastructure
  - deploy size and cost

### 9. Scaling, Monitoring, and Updates
* Scaling (SW and ML systems)
  - SW
    - train data partition storage.
    - cache text, image, user information features by weekly or daily, in prediction service.
    - sharding data generation pipeline component like in BQ, pyspark or beam.
  - ML
    - feature generation pipeline parallel
    - cache frequent user features
* Monitoring 
  - System gate metrics
    - each component running time and cost
    - system reaction time, latency time
  - ML gate metrics
    - model training time and cost
    - batch prediction time and cost
    - model offline metrics like train and test performance ROC-AUC, recall, save in dashboard or cloud performance tracker
    - model online metrics save on post-process pipeline or prediction system
  - Data gate metrics
    - features distribution like categorical feature unique value cardinality, dense numerical value mean, variance, kl-diverge compare last time training data
    - features correlation
    - each class training sample size ratio, training size after retrival component, this will cause two stages ranker or one stage ranker.
* Updates 
  - model update
    - train from scratch depends on how seriously user information change, this change should be defined early and trigger retrain automatically like alarm.
    - frequency, same as above depends on how seriously user information or interaction change like user's interested topic change. Weekly or monthly.
    - continual learning pipeline will update model depends on how metric been improved
  - pipeline update
    - depends on monitor, cost tracker or human sense.

### 10. Other talking points
* Positional bias