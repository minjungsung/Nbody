import matplotlib.pyplot as plt

# 마지막 배치 가져오기
X_test, y_test = dataset[-1]
with torch.no_grad():
    y_pred = model(X_test.unsqueeze(0)).numpy()[0]

plt.figure(figsize=(10,5))
plt.plot(y_test.numpy()[:6], label="True positions (x1,y1,x2)")
plt.plot(y_pred[:6], "--", label="Predicted")
plt.legend()
plt.show()