# Harmful content detection on social media

### 1. Problem Formulation
* Clarifying questions
  - What types of harmful content are we aiming to detect? (e.g., hate speech, explicit images, cyberbullying)?
  - What are the potential sources of harmful content? (e.g., social media, user-generated content platforms)
  - Are there specific legal or ethical considerations for content moderation
  - What is the expected volume of content to be analyzed daily?
  - What are supported languages? 
  - Are there human annotators available for labeling? 
  - Is there a feature for users to report harmful content? (click, text, etc). 
  - Is explainablity important here? 
  
* Integrity deals with: 
  - Harmful content (focus here)
  - Harmful act/actors  
* Goal: monitor posts, detect harmful content, and demote/remove 
* Examples harmful content categories: violence, nudity, hate speech 
* ML objective: predict if a post is harmful 
  - Input: Post (MM: text, image, video) 
  - Output:  P(harmful) or P(violent), P(nude), P(hate), etc
* ML Category: Multimodal (Multi-label) classification 
* Data: 500M posts / day (about 10K annotated)
* Latency: can vary for different categories 
* Able to explain the reason to the users (category) 
* Support different languages? Yes 

### 2. Metrics  
- Offline 
  - F1 score, PR-AUC, ROC-AUC 
- Online 
  - prevalence (percentage of harmful posts didn't prevent over all posts), harmful impressions, percentage of valid (reversed) appeals, proactive rate (ratio of system detected over system + user detected) 

### 3. Architectural Components  
* Multimodal input (text, image, video, etc): 
  - Multimodal fusion techniques 
    - Early Fusion: modalities combined first, then make a single prediction
      - pons:
        - It is unnecessary to collect training data separately for each modality. Since there is a single model to train, we only need to collect training data for that model.
        - The model considers all the modalities, so if each modality is benign, but their combination is harmful, then the model can potentially capture this in the unified feature vector.
    - Late Fusion: process modalities independently, fuse predictions
      - cons:
        - Separate training data for modalities, comb of individually safe content might be harmful
        - Time-consuming and expensive
* Multi-Label/Multi-Task classification 
  - Single binary classifier (P(harmful))
    - easy, not explainable for reason
  - One binary classifier per harm category (p(violence), p(nude), p(hate))
    - multiple models, trained and maintained separately, expensive 
  - Single multi-label classifier 
    - complicated task to learn 
  - Multi-task classifier: learn multi tasks simultaneously 
    - single shared layers (learns similarities between tasks) -> transformed features 
    - task specific layers: classification heads 
    - pros: single model, shared layers prevent redundancy, train data for each task can be used for others as well (limited data), prevent overfitting.
    - one post maybe belong to multi harmful label

### 4. Data Collection and Preparation

* Main actors for which data is available: 
  - Users 
    - user_id, age, gender, location, contact
  - Items(Posts) 
    - post_id, author_id, text context, images, videos, links, timestamp
  - User-post interactions 
    - user_id, post_id, interaction_type, value, timestamp


### 5. Feature Engineering
Feature preprocessing / preparation: 
Post Content (text, image, video) + Post Interactions (text + structured) + Author info + Context

- missing feature check
- feature distribution check

* Posts 
  - Text:  
    - Preprocessing (normalization + tokenization) 
    - Encoding (Vectorization): 
      - Statistical (BoW, TF-IDF)
      - ML based encoders (BERT)
    - We chose pre-trained ML based encoders (need semantics of the text)
    - We chose Multilingual Distilled (smaller, faster) version of BERT (need context), DistilmBERT 
  - Images/ Videos:   
    - Preprocessing: decoding, resize, scaling, normalization
    - Feature extraction: pre-trained feature extractors 
      - Images: 
        - CLIP's visual encoder 
        - SImCLR 
      - Videos: 
        - VideoMoCo
* Post interactions: 
  - Number of likes, comments, shares, reports -> scale the range to speed up convergence during model training.
  - Comments (text): similar to the post text (aggregate embeddings over comments like average comments vector)
* Users: 
  - Only use post author's info
    - demographics (age, gender, location)
    - account features (Number of followers /following, account age, no scale)
    - violation history (Number of violations, Number of user reports, profane words rate, no scale since as high as worse)
* Context: 
  - Time of day, device (one-hot encoding)

### 6. Model Development and Offline Evaluation
* Model selection 
  - NN: we use NN as it's commonly used for multi-task learning 
* HP tuning: 
  - Number of hidden layers, neurons in layers, activation function, learning rate, etc
  - grid search commonly used, but cost more time and resource
* Dataset: 
  - Natural labeling (user reports) - speed 
  - Hand labeling (human contractors) - accuracy 
  - we use natural labeling for train set (speed) and manual for eval set (accuracy)
* loss function: 
  - L = L1 + L2 + L3 ... for each task with w_i for each task
  - each task is a binary classified so e.g. CE for each task  
* Challenge for MM training: 
  - over-fitting (when one modality e.g. image dominates training)
    - gradient blending and focal loss 

### 7. Prediction Service
* 3 main components: 
  - Harmful content detection service 
  - Demoting service (prob of harm with low confidence)
    - Temporarily demotes the post in order to decrease the chance of it spreading among users.
    - The post is stored in storage for manual review by humans.
    - Use these labeled posts in future training iterations to improve the model
  - violation service (prob of harm with high confidence)
    - Immediately takes down a post. It also notifies the user why the post was removed.

### 8. Online Testing and Deployment  

### 9. Scaling, Monitoring, and Updates

### 10. Other topics 
* biases by human labeling 
* quickly tackling trending harmful content
  - Entailment as Few-Shot Learner, convert the class label into a natural language sentence which can be used to describe the label, and determine if the example entails the label description. Use CLS output as binary classification problem with zero, few, or small sample.
* use temporal information (e.g. sequence of actions)
* detect fake accounts
* architecture improvement: linear transformers
  - save train and inference time in long text transformer but may not have good performance when long sentence with small k (parameter in paper).