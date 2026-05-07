import matplotlib.pyplot as plt

# Data
modalities = [
    "106-channel polarization averaged reconstruction",
    "4-channel spectrum averaged reconstruction",
    "424-channel spectral polarimetric reconstruction",
    "4-channel Scatterogram",
]
channels = [106, 4, 424, 4]
textiles = [88.86, 89.47, 86.97, 95.70]
camouflage = [70.67, 76.67, 69.33, 86.67]

labels = [f"{m} ({c} ch)" for m, c in zip(modalities, channels)]


def sorted_for_plot(values, labels):
    pairs = sorted(zip(values, labels), key=lambda x: x[0])  # lowest -> highest
    sorted_values = [p[0] for p in pairs]
    sorted_labels = [p[1] for p in pairs]
    return sorted_values, sorted_labels


text_vals, text_labels = sorted_for_plot(textiles, labels)
camo_vals, camo_labels = sorted_for_plot(camouflage, labels)

fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

# Textiles chart
axes[0].barh(text_labels, text_vals, color="#4C78A8")
axes[0].set_title("Textiles Accuracy")
axes[0].set_xlabel("Accuracy (%)")
axes[0].set_xlim(0, 100)
axes[0].invert_yaxis()  # lowest at top, highest at bottom
for y, v in enumerate(text_vals):
    axes[0].text(v + 0.8, y, f"{v:.2f}%", va="center")

# Camouflage chart
axes[1].barh(camo_labels, camo_vals, color="#59A14F")
axes[1].set_title("Camouflage Accuracy")
axes[1].set_xlabel("Accuracy (%)")
axes[1].set_xlim(0, 100)
axes[1].invert_yaxis()  # lowest at top, highest at bottom
for y, v in enumerate(camo_vals):
    axes[1].text(v + 0.8, y, f"{v:.2f}%", va="center")

plt.show()