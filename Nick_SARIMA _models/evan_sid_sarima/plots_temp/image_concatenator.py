from PIL import Image

def concat_images_vertically(image1_path, image2_path, output_path):
    # Open the images
    im1 = Image.open(image1_path)
    im2 = Image.open(image2_path)
    
    # Get sizes of the images
    w1, h1 = im1.size
    w2, h2 = im2.size

    # If widths are different, resize image2 to match image1's width
    if w1 != w2:
        new_height = int(h2 * (w1 / w2))
        im2 = im2.resize((w1, new_height))
        w2, h2 = im2.size

    # Create a new image with the same width and combined height
    new_image = Image.new('RGB', (w1, h1 + h2))
    
    # Paste image1 at the top and image2 below it
    new_image.paste(im1, (0, 0))
    new_image.paste(im2, (0, h1))
    
    # Save the concatenated image
    new_image.save(output_path)
    print(f"Output image saved as {output_path}")

# File names are hardcoded here.
image1 = 'WG_10-30m_temp_train_test_forecast.png'
image2 = '1030temp.png'
output = 'LSTM_SARIMA_Temp_plot.png'

concat_images_vertically(image1, image2, output)
