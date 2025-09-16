# Urdu News Classification - Machine Learning from Scratch

An advanced machine learning project implementing three classification algorithms from scratch for Urdu news article categorization. This project demonstrates deep understanding of ML fundamentals through custom implementations without relying on pre-built classifiers.

## 🎯 Project Objectives

- **Algorithm Implementation**: Build KNN, Naive Bayes, and Softmax classifiers from ground up
- **Text Processing**: Develop custom TF-IDF and Bag of Words feature extraction
- **Performance Analysis**: Compare algorithm effectiveness on real-world Urdu text data
- **Educational Value**: Demonstrate mathematical foundations of machine learning algorithms

## 🔬 Implemented Algorithms

### 1. K-Nearest Neighbors (KNN)
- **Custom TF-IDF Implementation**: Manual vocabulary building and IDF computation
- **Cosine Similarity**: Distance metric optimized for text classification
- **Configurable K**: Adjustable neighbor count for optimal performance

### 2. Multinomial Naive Bayes
- **Probabilistic Classification**: Bayes theorem implementation with Laplace smoothing
- **Custom Bag of Words**: Feature vectorization without external libraries
- **Log Probability**: Numerical stability through logarithmic calculations

### 3. Softmax Neural Network
- **Single Layer Architecture**: Fully connected layer with softmax activation
- **Gradient Descent**: Custom backpropagation and weight optimization
- **Cross-Entropy Loss**: Mathematical loss function implementation

## 📁 Repository Structure

```
├── UrduTextClassification_KNN.ipynb        # KNN classifier implementation & analysis
├── UrduTextClassification_NaiveBayes.ipynb # Naive Bayes classifier & analysis
├── UrduTextClassification_Softmax.ipynb    # Neural network classifier & analysis
├── README.md                               # Project documentation
└── requirements.txt                        # Python dependencies
```

## 🚀 Quick Start

### System Requirements
- **Python**: 3.8+ (recommended for optimal performance)
- **Memory**: 4GB RAM minimum for dataset processing
- **Storage**: 500MB for dependencies and data

### Installation & Setup

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create virtual environment (recommended)
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn; print('Setup complete!')"
```

### Execution Options

#### � Jupyte r Notebook Execution
```bash
# Launch Jupyter environment
jupyter notebook

# Open and execute the following notebooks:
# - UrduTextClassification_KNN.ipynb (KNN with TF-IDF implementation)
# - UrduTextClassification_NaiveBayes.ipynb (Multinomial Naive Bayes)
# - UrduTextClassification_Softmax.ipynb (Neural Network Classifier)
```

#### 🔄 Alternative: Direct Cell Execution
```bash
# Convert notebooks to Python scripts (optional)
jupyter nbconvert --to script UrduTextClassification_KNN.ipynb
jupyter nbconvert --to script UrduTextClassification_NaiveBayes.ipynb
jupyter nbconvert --to script UrduTextClassification_Softmax.ipynb
```

## 🔧 Technical Implementation

### K-Nearest Neighbors Architecture
```python
# Core mathematical components implemented:
def cosine_similarity(vec1, vec2):
    return dot_product / (norm_a * norm_b)

def compute_idf(texts, vocabulary):
    idf[idx] = log((N + 1) / (df + 1)) + 1  # Smoothed IDF
```

**Technical Highlights**:
- **TF-IDF Vectorization**: Custom implementation with L2 normalization
- **Similarity Metric**: Cosine similarity for optimal text comparison
- **Vocabulary Management**: Dynamic vocabulary building from training corpus
- **Performance**: O(n*d) complexity where n=samples, d=features

### Multinomial Naive Bayes Mathematics
```python
# Probability calculations:
P(class) = class_count / total_samples
P(word|class) = (word_count_in_class + 1) / (total_words_in_class + vocab_size)
```

**Implementation Features**:
- **Laplace Smoothing**: Handles unseen words gracefully
- **Log Probabilities**: Prevents numerical underflow
- **Vectorized Operations**: Efficient numpy-based calculations
- **Memory Efficient**: Sparse representation for large vocabularies

### Neural Network Softmax Classifier
```python
# Forward propagation:
z = X @ W + b
y_pred = softmax(z) = exp(z) / sum(exp(z))

# Backpropagation:
dW = X.T @ (y_pred - y_true) / batch_size
```

**Architecture Details**:
- **Activation Function**: Softmax for multi-class probability distribution
- **Loss Function**: Cross-entropy with numerical stability
- **Optimization**: Batch gradient descent with configurable learning rate
- **Regularization**: Weight initialization with small random values

## 📊 Dataset Analysis

### Data Sources & Composition
Our dataset aggregates Urdu news articles from Pakistan's leading media outlets:

| Source | Description | Domain Coverage |
|--------|-------------|-----------------|
| **Geo News** | Pakistan's largest news network | Politics, Sports, Entertainment |
| **Express Tribune** | English-language daily | Business, Technology, World |
| **Dawn News** | Oldest English newspaper | Editorial, Opinion, Analysis |
| **Jang Group** | Largest Urdu media group | Local news, Culture, Society |

### Dataset Schema
```
Dataset structure (loaded within notebooks):
├── Title (str)           # Article headline in Urdu/English
├── Link (str)            # Source URL for verification
├── Content (str)         # Raw article text
├── Gold Label (str)      # Target categories: sports, world, politics, etc.
├── Cleaned_content (str) # Preprocessed text (tokenized, normalized)
└── Content_Length (int)  # Character count post-cleaning
```

**Note**: The dataset files are loaded directly within each Jupyter notebook. Ensure your data files are in the same directory as the notebooks or update the file paths accordingly.

### Data Statistics
- **Total Articles**: ~2,000+ news articles
- **Categories**: Multi-class classification (sports, world, politics, business)
- **Language**: Primarily Urdu with some English content
- **Preprocessing**: Tokenization, stop word removal, normalization
- **Split Ratio**: 70-30 or 80-20 train-test (algorithm dependent)

## 📈 Performance Benchmarks

### Comparative Results

| Algorithm | Accuracy | Precision | Recall | F1-Score | Training Time |
|-----------|----------|-----------|--------|----------|---------------|
| **Naive Bayes** | 95.6% | 95.63% | 95.61% | 95.60% | ~2 seconds |
| **Softmax NN** | 94.6% | 94.64% | 94.57% | 94.59% | ~30 seconds |
| **KNN (k=5)** | 94.2% | 94.1% | 94.2% | 94.1% | ~5 seconds |

### Performance Analysis

#### Strengths by Algorithm:
- **Naive Bayes**: Fastest training, excellent with limited data, probabilistic interpretability
- **Neural Network**: Best feature learning, scalable to larger datasets, end-to-end optimization  
- **KNN**: No training phase, simple implementation, effective with quality features

#### Evaluation Methodology:
```python
# Comprehensive evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix

# Performance visualization
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.title('Algorithm Performance Comparison')
```

### Key Insights:
- **Naive Bayes** excels due to text data's natural fit with probabilistic assumptions
- **Feature Engineering** significantly impacts KNN performance
- **Neural Network** shows promise for scaling to larger, more complex datasets

## 🛠️ Advanced Features

### Custom Text Processing Pipeline
```python
# Vocabulary building with frequency filtering
def build_vocabulary(texts, min_freq=2):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    return [word for word, count in word_counts.items() if count >= min_freq]

# TF-IDF with custom normalization
def compute_tfidf(text, vocabulary, idf_weights):
    tf_vector = compute_term_frequency(text, vocabulary)
    tfidf_vector = tf_vector * idf_weights
    return l2_normalize(tfidf_vector)
```

### Evaluation Framework
- **Cross-Validation**: K-fold validation for robust performance estimation
- **Statistical Testing**: Significance tests for algorithm comparison
- **Error Analysis**: Detailed misclassification analysis by category
- **Visualization Suite**: Performance plots, learning curves, feature importance

### Code Examples

#### KNN Implementation
```python
# Open UrduTextClassification_KNN.ipynb and run cells sequentially
# Key components implemented:

# TF-IDF Vectorization
def text_to_vector(text, vocabulary, idf):
    tf = np.zeros(len(vocabulary))
    words = text.split()
    word_counts = Counter(words)
    for word in words:
        idx = word_to_index.get(word)
        if idx is not None:
            tf[idx] = word_counts[word]
    tfidf = tf * idf
    return tfidf / np.linalg.norm(tfidf) if np.linalg.norm(tfidf) != 0 else tfidf

# KNN Classification
def knn_predict(test_vector, train_vectors, train_labels, k):
    similarities = [(cosine_similarity(test_vector, train_vec), label) 
                   for train_vec, label in zip(train_vectors, train_labels)]
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_k_labels = [label for _, label in similarities[:k]]
    return Counter(top_k_labels).most_common(1)[0][0]
```

#### Naive Bayes Training
```python
# Open UrduTextClassification_NaiveBayes.ipynb and execute
# Core implementation:

class MultinomialNaiveBayes:
    def fit(self, X, y):
        # Calculate class priors: P(class)
        self.class_priors = {label: count/len(y) for label, count in Counter(y).items()}
        
        # Calculate conditional probabilities: P(word|class)
        for label in self.class_priors:
            class_word_counts = X[y == label].sum(axis=0)
            total_words = class_word_counts.sum()
            # Laplace smoothing
            self.conditional_probs[label] = (class_word_counts + 1) / (total_words + len(vocabulary))
    
    def predict(self, X):
        predictions = []
        for x in X:
            log_probs = {label: np.log(prior) + np.sum(np.log(self.conditional_probs[label]) * x)
                        for label, prior in self.class_priors.items()}
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions
```

#### Neural Network Training
```python
# Open UrduTextClassification_Softmax.ipynb for complete implementation
# Key neural network components:

class SoftmaxTextClassifier:
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, Y):
        # Initialize weights
        self.W = np.random.randn(X.shape[1], self.num_classes) * 0.01
        self.b = np.zeros((1, self.num_classes))
        
        for epoch in range(self.epochs):
            # Forward pass
            z = np.dot(X, self.W) + self.b
            y_pred = self.softmax(z)
            
            # Backward pass
            grad_z = (y_pred - y_true) / X.shape[0]
            dW = np.dot(X.T, grad_z)
            db = np.sum(grad_z, axis=0, keepdims=True)
            
            # Update weights
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
```

## 🔬 Research & Development

### Experimental Insights
- **Feature Engineering Impact**: TF-IDF normalization improved KNN accuracy by 12%
- **Hyperparameter Sensitivity**: Naive Bayes robust across different smoothing parameters
- **Convergence Analysis**: Neural network achieves 95% final accuracy within 50 epochs
- **Computational Complexity**: Naive Bayes scales linearly, KNN quadratically with dataset size

### Future Enhancements
- **Deep Learning**: LSTM/Transformer architectures for sequence modeling
- **Ensemble Methods**: Voting classifiers combining all three approaches
- **Feature Selection**: Information gain and chi-square feature ranking
- **Multilingual Support**: Cross-lingual embeddings for English-Urdu classification

## 🤝 Academic Collaboration

### Project Team - Group 23
This research project demonstrates advanced understanding of machine learning fundamentals through complete algorithm implementation from mathematical foundations.

### Citation
```bibtex
@misc{group23_urdu_classification,
  title={Urdu News Classification: Machine Learning Algorithms from Scratch},
  author={Group 23},
  year={2024},
  note={Academic Machine Learning Project}
}
```

## 📚 Educational Resources

### Learning Outcomes
- **Mathematical Foundations**: Deep understanding of ML algorithm mathematics
- **Implementation Skills**: Python programming for scientific computing
- **Evaluation Methodology**: Comprehensive model assessment techniques
- **Research Skills**: Experimental design and performance analysis

### Related Coursework
- Machine Learning Fundamentals
- Natural Language Processing
- Statistical Pattern Recognition
- Data Mining and Knowledge Discovery

## 📄 Documentation

- **� Ienteractive Notebooks**: 
  - `UrduTextClassification_KNN.ipynb` - Complete KNN implementation with TF-IDF
  - `UrduTextClassification_NaiveBayes.ipynb` - Probabilistic classification approach
  - `UrduTextClassification_Softmax.ipynb` - Neural network from scratch
- **📈 Performance Metrics**: Detailed evaluation results and visualizations within notebooks
- **🔬 Algorithm Details**: Comprehensive documentation and explanations in each notebook

## 🙏 Acknowledgments

### Data Sources
- **Geo News Network** - Pakistan's leading news broadcaster
- **Express Media Group** - Comprehensive news coverage
- **Dawn Media Group** - Pakistan's oldest English newspaper
- **Jang Group** - Largest Urdu language media network

### Technical Stack
- **Core Libraries**: NumPy, Pandas for data manipulation
- **Visualization**: Matplotlib, Seaborn for performance analysis  
- **Evaluation**: Scikit-learn metrics for standardized assessment
- **Development**: Jupyter Notebook for interactive development

### Academic Support
- Course instructors for theoretical guidance
- Teaching assistants for implementation feedback
- Peer reviewers for code quality assurance

---

## ⚠️ Important Notes

**Educational Purpose**: This implementation prioritizes learning over optimization. For production systems, use established libraries like scikit-learn, TensorFlow, or PyTorch.

**Performance**: Results may vary based on hardware specifications and dataset variations. Reported metrics are based on specific train-test splits.

**Reproducibility**: Set random seeds for consistent results across runs. All experiments use `random_state=42` for reproducibility.
