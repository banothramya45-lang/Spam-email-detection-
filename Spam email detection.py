from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "Congratulations! You won a lottery",
    "Meeting at 10 AM tomorrow",
    "Claim your free gift now",
    "Project submission deadline is today"
]

labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

test_email = ["Free gift available now"]
test_vector = vectorizer.transform(test_email)

prediction = model.predict(test_vector)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")
