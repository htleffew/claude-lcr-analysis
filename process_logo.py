import sys
from PIL import Image

def process_logo(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    
    # The original image has white corners which get turned into black corners.
    # To fix this, let's just crop out the outer 15% of the image before processing.
    width, height = img.size
    crop_x = int(width * 0.15)
    crop_y = int(height * 0.15)
    img = img.crop((crop_x, crop_y, width - crop_x, height - crop_y))
    
    data = img.getdata()

    new_data = []
    for r, g, b, a in data:
        # Calculate brightness (0 to 255)
        brightness = (r + g + b) / 3.0
        
        # New image: solid black, but alpha is equal to the original brightness.
        # This perfectly inverts the white symbol to black, and makes the black background transparent!
        new_data.append((0, 0, 0, int(brightness)))

    img.putdata(new_data)
    img.save(output_path, "PNG")

if __name__ == "__main__":
    process_logo(sys.argv[1], sys.argv[2])
