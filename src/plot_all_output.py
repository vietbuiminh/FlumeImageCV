import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

output_images_folder = 'data/output_images'
image_files = [f for f in os.listdir(output_images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    if i < len(image_files):
        image_path = os.path.join(output_images_folder, image_files[i])
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.set_title(image_files[i], fontsize=8)
        ax.axis('off')  # Hide axes
    else:
        ax.axis('off')  # Hide unused subplots

# Adjust layout
plt.tight_layout()
plt.show()