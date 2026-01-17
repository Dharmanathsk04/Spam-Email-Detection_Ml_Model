from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X = ["free money", "hello friend", "win prize", "how are you"]
y = [1, 0, 1, 0]  

cv = CountVectorizer()
X_vec = cv.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

print(model.predict(cv.transform(["free prize"])))
