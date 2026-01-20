import matplotlib.pyplot as plt

accuracy = [0.82, 0.85, 0.88, 0.89, 0.91]
epochs = [1, 2, 3, 4, 5]

plt.plot(epochs, accuracy, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Epochs")
plt.ylim(0, 1)
plt.grid(True)

# Save image in static folder
plt.savefig("static/accuracy.png")
plt.close()

print("Graph saved in static/accuracy.png")
