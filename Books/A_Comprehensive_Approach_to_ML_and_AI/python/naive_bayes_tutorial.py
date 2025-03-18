import numpy as np
import matplotlib.pyplot as plt

# --- Data Generation ---
np.random.seed(1)
X = np.vstack([np.random.randn(50, 2), np.random.randn(50, 2) + 3])
y = np.hstack([np.ones(50), 2*np.ones(50)])  # Classes 1 and 2

# --- Naive Bayes Model Functions ---
def my_naive_bayes(trainData, trainLabels):
    classes = np.unique(trainLabels)
    numClasses = len(classes)
    N, d = trainData.shape
    priors = np.zeros(numClasses)
    means = {}
    variances = {}
    for i, cls in enumerate(classes):
        idx = (trainLabels == cls)
        priors[i] = np.sum(idx) / N
        means[cls] = np.mean(trainData[idx, :], axis=0)
        variances[cls] = np.var(trainData[idx, :], ddof=0)  # 1/N normalization
    model = {'classes': classes, 'priors': priors, 'means': means, 'variances': variances}
    return model

def classify_naive_bayes(model, x):
    numClasses = len(model['classes'])
    scores = np.zeros(numClasses)
    for i, cls in enumerate(model['classes']):
        mu = model['means'][cls]
        sigma2 = model['variances'][cls]
        # Compute Gaussian likelihood for each feature (assuming independence)
        likelihood = np.prod((1/np.sqrt(2*np.pi*sigma2)) * np.exp(-((x - mu)**2) / (2*sigma2)))
        scores[i] = np.log(model['priors'][i]) + np.log(likelihood + 1e-8)
    predictedClass = model['classes'][np.argmax(scores)]
    return predictedClass

# Train model
model = my_naive_bayes(X, y)

# Classify a new point
new_point = np.array([0, 0])
predicted_class = classify_naive_bayes(model, new_point)
print("Predicted class for point [0, 0]:", predicted_class)

# Create grid for decision boundary
x1range = np.linspace(np.min(X[:,0])-1, np.max(X[:,0])+1, 100)
x2range = np.linspace(np.min(X[:,1])-1, np.max(X[:,1])+1, 100)
X1, X2 = np.meshgrid(x1range, x2range)
gridPoints = np.c_[X1.ravel(), X2.ravel()]

predictions = np.array([classify_naive_bayes(model, pt) for pt in gridPoints])
Z = predictions.reshape(X1.shape)

# Plot original data and decision boundaries
plt.figure()
plt.scatter(X[y==1, 0], X[y==1, 1], s=50, c='r', label='Class 1')
plt.scatter(X[y==2, 0], X[y==2, 1], s=50, c='b', label='Class 2')
plt.title("Naive Bayes Classification")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.legend(loc='best')
plt.show()

plt.figure()
plt.scatter(X[y==1, 0], X[y==1, 1], s=50, c='r', label='Class 1')
plt.scatter(X[y==2, 0], X[y==2, 1], s=50, c='b', label='Class 2')
plt.contourf(X1, X2, Z, levels=np.unique(Z), alpha=0.3, cmap='coolwarm')
plt.title("Naive Bayes Classification - Decision Boundaries")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.legend(loc='best')
plt.show()
