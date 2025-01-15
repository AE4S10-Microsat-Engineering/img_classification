import os
import torch
import kornia
import kornia.augmentation as K
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import random as rnd

# Load an image using PIL and convert it to a PyTorch tensor
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = kornia.image_to_tensor(np.array(img)).float() / 255.0  # Normalize to [0, 1]
    return img_tensor

# Display the original and transformed images
def display_images(original, transformed, titles=["Original", "Transformed"]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, title in zip(axes, [original, transformed], titles):
        img_np = kornia.tensor_to_image(img)
        ax.imshow(img_np)
        ax.set_title(title)
        ax.axis("off")
    plt.show()

def display_images_grid(original_image, transformed_images, title_original="Original", titles_transformed="Transformed", figure_title="Sample Images"):
    """
    Displays the original image in the top-left corner and a grid of transformed images.

    Parameters:
    - original_image: The original image (PyTorch tensor).
    - transformed_images: List of transformed images (PyTorch tensors).
    - title_original: Title for the original image (default is "Original").
    - title_transformed: Title for the transformed images (default is "Transformed").

    Returns:
    - None: Displays the images in a grid.
    """
    if titles_transformed == str:
        titles_transformed = [titles_transformed]*len(transformed_images)
    else:
        for i,item in enumerate(transformed_images):
            titles_transformed[i] = f"{titles_transformed[i]}"
    print(f"{len(transformed_images)}, {len(titles_transformed)}")
    print(titles_transformed)

    num_transformed = len(transformed_images)
    grid_size = int(np.ceil(np.sqrt(num_transformed + 1)))  # Calculate grid size to fit all images

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    fig.suptitle(figure_title)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Plot the original image in the top-left corner
    axes[0].imshow(kornia.tensor_to_image(original_image))
    axes[0].set_title(title_original)
    axes[0].axis("off")

    # Plot transformed images
    for i, transformed in enumerate(transformed_images, start=1):
        axes[i].imshow(kornia.tensor_to_image(transformed))
        axes[i].set_title(f"trans:{i}, {titles_transformed[i-1]} ")
        axes[i].axis("off")

    # Turn off unused subplots
    for j in range(len(transformed_images) + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

# Example transformation pipeline
def apply_rotation(image_tensor, angles={-45,45}, probability=1):
    """
    Apply a random rotation. If single value is given it is [-degrees,degrees] Otherwise range can be specified, p is the probability to rotate at all
    """ 
    transform = torch.nn.Sequential(
        K.RandomRotation(degrees=angles,p=probability)
    )
    transformed_image = transform(image_tensor.unsqueeze(0))  # Add batch dimension
    return transformed_image.squeeze(0)  # Remove batch dimension

# Example transformation pipeline
def apply_horizontalflip(image_tensor, probability=1):
    """
    p is the probability to flip
    """ 
    transform = torch.nn.Sequential(
        K.RandomHorizontalFlip(p=probability)
    )
    transformed_image = transform(image_tensor.unsqueeze(0))  # Add batch dimension
    return transformed_image.squeeze(0)  # Remove batch dimension

# Example transformation pipeline
def apply_colorjitter(image_tensor, probability=1):
    # Define a sequence of transformations
    transform = torch.nn.Sequential(
        K.ColorJitter(brightness=[2,2], contrast=0.2, saturation=0.2, hue=0.1, p=probability)
    )
    # Apply transformations
    transformed_image = transform(image_tensor.unsqueeze(0))  # Add batch dimension
    return transformed_image.squeeze(0)  # Remove batch dimension

# Example transformation pipeline
def apply_geotransformations(image_tensor, rotAngles=[-45,45], translate=[0.25,0.25], scale=[0.75,1.25], shearAngles=[-45,45], probability=1, padding="zeros"):
    """
    rotAngles are angles for rotation, either single value to specifiy range, or give the complete range
    translate specifices the max x and y translation allowed, the number signifies the percentage of movement allowed
    scale scales the image between min and max, the number is the percentage of scaling allowed
    shearAngles are the angles for shearing of the image
    
    p is the probability to flip

    padding defines the padding mode, options are: "zeros" for black, "border" extends pixels to border, "reflection" this will reflect the image
    """ 
    if type(rotAngles) == int or type(rotAngles) == float:
        rotAngles = [-rotAngles,rotAngles]
    if type(translate) == int or type(translate) == float:
        translate = [translate,translate]
    if type(scale) == int or type(scale) == float:
        scale = [scale,scale]
    if type(shearAngles) == int or type(shearAngles) == float:
        shearAngles = [-shearAngles,shearAngles]
    transform = torch.nn.Sequential(
        K.RandomAffine(degrees=rotAngles, translate=translate, scale=scale, shear=shearAngles, p=probability, padding_mode=padding)
    )
    # Apply transformations
    transformed_image = transform(image_tensor.unsqueeze(0))  # Add batch dimension
    return transformed_image.squeeze(0)  # Remove batch dimension



def shift_image(image, num_rows, start_row):
    """
    Removes a specified number of rows starting from a given row (start_row) 
    and adds an equal number of black rows to the top.
    This simulates velocity variations

    Parameters:
    - image: Input image (PyTorch tensor of shape CxHxW, normalized to [0, 1]).
    - num_rows: Number of rows to remove from the image.
    - start_row: Row index from which to start removing rows.

    Returns:
    - Transformed image as a PyTorch tensor.
    """

    c, h, w = image.shape

    # Validate inputs
    if start_row < 0 or start_row >= h:
        raise ValueError(f"The starting row ({start_row}) is outside of the image height ({h}).")
    if start_row + num_rows > h:
        raise ValueError(f"Removing {num_rows} rows from row {start_row} exceeds image height ({h}).")

    # Create a black padding tensor
    black_rows = torch.zeros((c, num_rows, w), dtype=image.dtype, device=image.device)

    # Remove rows starting from start_row
    top_part = image[:, :start_row, :]  # Rows before start_row
    bottom_part = image[:, start_row + num_rows:, :]  # Rows after start_row + num_rows

    # Combine the remaining parts and add black rows at the top
    cropped_image = torch.cat([top_part, bottom_part], dim=1)
    transformed_image = torch.cat([cropped_image, black_rows], dim=1)

    return transformed_image

def get_random_dataset():
    setname_list = ["AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"]
    index = rnd.randrange(0,len(setname_list))
    setname = setname_list[index]
    dataset_path = f"{setname}"
    return dataset_path

# Main script
if __name__ == "__main__":
    # for n in range(10):
    #     print(n)
    #     image_path = os.path.join(os.getcwd(),f"./training_data/EuroSAT_RGB/AnnualCrop/AnnualCrop_{n+1}.jpg") # Replace with your image path
    #     original_image = load_image(image_path)
    #     transformed_images = []
    #     rots = []
    #     trans = []
    #     scales = []
    #     shears = []
    #     labels = []
    #     for n in range(15):
    #         rot = rnd.randrange(-50,50,step=1)/10
    #         rots.append(rot)
    #         tran = rnd.randrange(0,10,step=1)/100
    #         trans.append(tran)
    #         scale = rnd.randrange(80,120,step=1)/100
    #         scales.append(scale)
    #         shear = rnd.randrange(-50,50,step=1)/10
    #         shears.append(shear)
    #         labels.append(f"r:{rot},t:{tran},s:{scale},sh:{shear}")
    #         transformed_image = apply_geotransformations(original_image, rotAngles=[rot,rot], translate=[tran,tran], scale=[scale,scale], shearAngles=[shear,shear], probability=1, padding="zeros")
    #         transformed_images.append(transformed_image)
    #     display_images_grid(original_image, transformed_images, titles_transformed=labels)
    
    # for n in range(10):
    #     setname = get_random_dataset()
    #     image_path = os.path.join(os.getcwd(),f"./training_data/EuroSAT_RGB/{setname}/{setname}_{n+1}.jpg") # Replace with your image path
    #     original_image = load_image(image_path)
    #     transformed_images = []
    #     starts = []
    #     nums = []
    #     labels = []
    #     for n in range(15):
    #         num = rnd.randrange(0,15)
    #         nums.append(num)
    #         start = rnd.randrange(0,63-num)
    #         starts.append(start)
    #         labels.append(f"s:{start},n:{num}")
    #         transformed_image = shift_image(original_image, num_rows=num, start_row=start)
    #         transformed_images.append(transformed_image)
    #     display_images_grid(original_image, transformed_images, titles_transformed=labels, figure_title=setname)


    for n in range(10):
        setname = get_random_dataset()
        n = rnd.randrange(0,1000)
        image_path = os.path.join(os.getcwd(),f"./training_data/EuroSAT_RGB/{setname}/{setname}_{n+1}.jpg") # Replace with your image path
        original_image = load_image(image_path)
        transformed_images = []
        rots = []
        trans = []
        scales = []
        shears = []
        starts = []
        nums = []
        labels = []
        for n in range(15):
            rot = rnd.randrange(-50,50,step=1)/10
            rots.append(rot)
            tran = rnd.randrange(0,10,step=1)/100
            trans.append(tran)
            scale = rnd.randrange(80,120,step=1)/100
            scales.append(scale)
            shear = rnd.randrange(-50,50,step=1)/10
            shears.append(shear)
            if rnd.randrange(0,99) <= 50: #adds a probability factor
                num = rnd.randrange(0,15)
                nums.append(num)
                start = rnd.randrange(0,63-num)
                starts.append(start)
                transformed_image = shift_image(original_image, num_rows=num, start_row=start)
            else:
                start = -1
                num = -1
                transformed_image = original_image
            labels.append(f"r:{rot},t:{tran},s:{scale},sh:{shear},c:{start},n:{num}")
            transformed_image = apply_geotransformations(transformed_image, rotAngles=[rot,rot], translate=[tran,tran], scale=[scale,scale], shearAngles=[shear,shear], probability=1, padding="zeros")
            transformed_images.append(transformed_image)
        display_images_grid(original_image, transformed_images, titles_transformed=labels, figure_title=setname)