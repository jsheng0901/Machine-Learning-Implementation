# Ads Click Prediction 

### 1. Problem Formulation
* Clarifying questions
  - What is the primary business objective of the click prediction system?
    - maximize revenue
  - What types of ads are we predicting clicks for (e.g., display ads, video ads, sponsored content)?
  - Are there specific user segments or contexts we should consider (e.g., user demographics, browsing history)?
  - How will we define and measure the success of click predictions (e.g., click-through rate, conversion rate)?
  - Do we have negative feedback features (such as hide ad, block, etc.)? 
    - Yes, assume users can hide an ad they don’t like.
  - Do we have fatigue period (where ad is no longer shown to the users where there is no interest, for X days)?
    - yes, and we may use this to label negative case
  - What type of user-ad interaction data do we have access to can we use it for training our models? 
  - Do we need continual training? 
    - usually yes, since experiments have shown that even a 5-minute delay in updating models can damage performance.
  - How do we collect negative samples? (not clicked, negative feedback). 
  
* Use case(s) and business goal
  - use case: predict which ads a user is likely to click on when presented with multiple ad options.
  - business objective: maximize ad revenue by delivering more relevant ads to users, improving click-through rates, and maximizing the value of ad inventory.

* Requirements
  - Real-time prediction capabilities to serve ads dynamically.
  - Scalability to handle a large number of ad impressions.
  - Integration with ad serving platforms and data sources.
  - Continuous model training and updating.

* Constraints:
  - Privacy and compliance with data protection regulations.
  - Latency requirements for real-time ad serving.
  - Limited user attention, as users may quickly decide whether to click on an ad.

* Data: Sources and Availability:
  - Data sources include user interaction logs, ad content data, user profiles, and contextual information.
  - Historical click and impression data for model training and evaluation.
  - Availability of labeled data for supervised learning.

* Assumptions:
  - Users' click behavior is influenced by factors that can be learned from historical data.
  - Ad content and relevance play a significant role in click predictions.
  - The click behavior can be modeled as a classification problem.
  
* ML Formulation:
  - ML Category
    - Ad click prediction is a ranking problem, point-wise LTR to binary classification click or not.

### 2. Metrics
* Offline metrics 
  - CE in binary classification 
  - NCE (normalized over baseline)

* Online metrics 
  - CTR (#clicks/#impressions)
  - Conversion rate (#conversion/#impression)
  - Revenue lift (increase in revenue over time)
  - Hide rate (#hidden ads/#impression)

### 3. Architectural Components  
* High level architecture
  - prepare data
  - generate train/eval/test data
  - training
    - We can use point-wise learning to rank (LTR)
      - The binary classification task, where the goal is to predict whether a user will click (1) or not click (0) on a given ad impression -> given a pair of <user, ad> as input -> click or no click 
      - Features can include user demographics, ad characteristics, context (e.g., device, location), and historical behavior.
      - Machine learning models, such as logistic regression, decision trees, gradient boosting, or deep neural networks, can be used for prediction.

### 4. Data Collection and Preparation
* Data Sources
  - Doc or query information
    - Users (query)
    - Ads (doc)
  - Doc interaction query information
    - User-ads interaction

* Labelling
  - Use ads interaction will click (1) or not click (0) on a given ad impression time range, time range can as hyperparameter.
  - Note: positive class may be fewer than negative, so label may be unbalanced.

### 5. Feature Engineering
* Feature selection 
  - Ads: 
    - IDs 
    - categories 
    - Image/videos
    - Number of impressions / clicks (ad, adv, campaign)

[//]: # (    - | Ad ID | Advertiser ID | Ad group ID | Campaign ID | Category  | Subcategory | Images or Videos |)

[//]: # (      |-------|---------------|-------------|-------------|-----------|-------------|------------------|)

[//]: # (      | 1     | 1             | 4           | 7           | travel    | hotel       | u1.jpg           |)

[//]: # (      | 2     | 7             | 2           | 9           | insurance | car         | t5.jpg           |)
  - User: 
    - ID, username
    - Demographics (Age, gender, location, timezone)
    - Context (device, time of day, etc)
    - Interaction history (e.g. user ad click rate, total clicks, etc)

[//]: # (    - | ID | Username | Age | Gender | City | Country | Language | timezone |)

[//]: # (      |----|----------|-----|--------|------|---------|----------|----------|)

[//]: # (      | 1  | A        | 15  | male   | NYC  | US      | English  | EST      |)

[//]: # (      | 2  | B        | 25  | female | LOS  | US      | English  | PST      |)
  - User-Ad interaction: 
    - IDs(user, Ad), interaction type, time, location, dwell time

* Feature representation / preparation
  - Numerical feature
      - Sparse features
        - IDs: embedding layer (each ID type its own embedding layer, like in nlp id in here is word)
        - City/Country/Language: embedding don't need too many dimension like 5 is ok.

      - Dense features:
        - Engagement feats: Number of clicks, impressions, etc 
          - use directly, not scale the range which means normalized numerical data
  
      - Discrete numerical: 
        - Age: 
          - bucketize into categorical then one hot
        - Time of day
          - bucketize into categorical then one hot

  - Image / Video: 
    - preprocess first
    - use pre-trained models e.g. SimCLR to convert -> feature vector

  - Category feature
    - Textual data 
      - normalization, tokenization, encoding (lookup table, hash table), convert text to unique id and then convert to embedding
      - use pre-trained LM model like BERT or easy model word2vec convert -> feature vector, word2vec easy since it's static word in here. 
    - Gender
      - one hot
    - Device
      - bucketize into categorical then one hot

### 6. Model Development and Offline Evaluation
* Model selection 
  - Basemodel
    - LR:
      - fast
      - easy to explain
  - Advanced model:
    - Feature crossing + LR 
      - hyperparameters: L1/L2 regularization penalty lambda, n_iterations, learning rate, etc.
      - loss: CE loss
        - feature crossing: combine 2/more features into new feats (e.g. sum, product)
        - pros: capture non-linearity interactions between feats 
        - cons: more sparsity, manual process, cannot capture complex interactions, and domain knowledge needed
    - GBDT 
      - hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, learning_rate
      - pros: interpretable，non-linearity
      - cons: inefficient for continual training, can't train embedding layers 
    - GBDT + LR 
      - GBDT for feature selection and/or extraction, LR for classified
      - pros: interpretable，non-linearity, help select feature first
      - cons: inefficient for continual training, can't train embedding layers 
    - NN
      - Two options: single network, two tower network (user tower, ad tower)
      - Cons for ads prediction: 
        - sparsity of features, huge number of them 
        - hard to capture pairwise interactions (large no of them)
      - Not a good choice here. 
    - Deep and cross network (DCN)
      - finds feature interactions automatically 
      - two parallel networks: deep network (learns complex features) and cross network (learns interactions)
      - two types: stacked, and parallel 
    - Factorization Machine 
      - embedding based model, improves LR by automatically learning feature interactions (by learning embeddings for features) 
      - w0  + \sum (w_i.x_i) + \sum\sum <v_i, v_j> x_i.x_j
      - cons: can't learn higher order interactions from features unlike NN
      - pros: fast, use less parameters to effectively capture pairwise interactions between features
    - Deep factorization machine (DFM)
      - combines a NN (for complex features) and an FM (for pairwise interactions)
    - start with LR to form a baseline, then experiment with DCN & DeepFM 
   
* Model Training 
  - Loss function: 
    - binary classification: CE 
    - Dataset 
      - labels: positive: user clicks the ad < t seconds after ad is shown, negative: no click within t secs  
  - Model eval and HP tuning 
  - Iterations 
  
### 7. Prediction Service
* Data Prepare pipeline
  - static features (e.g. ad img, category) -> batch feature compute (daily, weekly) -> feature store in cloud
  - dynamic features: number of ad impressions, clicks. 
* Prediction pipeline 
  - two stage (funnel) architecture 
    - candidate generation 
      - use ad targeting criteria by advertiser (age, gender, location, etc.)
    - ranking 
      - features -> model -> click prob. -> sort top k ads
      - re-ranking: business logic (e.g. diversity)
* Continual learning pipeline 
  - fine tune on new data, eval, and deploy if improves metrics  
  
### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release (engineer part)
  - deploy platform depends on company ML infrastructure
  - deploy size and cost 

### 9. Scaling, Monitoring, and Updates 
* Scaling (SW and ML systems)
* Monitoring 
* Updates 

### 10. Other topics  
* calibration: 
  - fine-tuning predicted probabilities to align them with actual click probabilities 
* data leakage: 
  - info from the test or eval dataset influences the training process
  - target leakage, data contamination (from test to train set)
  - train, val, test split by time, don't use old data train or test data from old history
* catastrophic forgetting 
  - model trained on new data loses its ability to perform well on previously learned tasks, usually happens when new data has big change which should be monitored by data gate metrics