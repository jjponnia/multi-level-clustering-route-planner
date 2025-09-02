import re
from PIL import Image
import os

def natural_sort_key(s):
    """Generate a key for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def create_gif_from_pngs(png_folder, output_gif, duration=500):
    """
    Convert a sequence of PNG files into a GIF.

    Args:
        png_folder (str): Path to the folder containing PNG files.
        output_gif (str): Path to save the output GIF file.
        duration (int): Duration for each frame in milliseconds.

    Returns:
        None
    """
    # Get all PNG files in the folder, sorted by name
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')], key=natural_sort_key)
    for file in png_files:
        print(f"Found PNG file: {file}")

    # Load images
    images = [Image.open(os.path.join(png_folder, file)) for file in png_files]

    # Save as GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

# Example usage
create_gif_from_pngs('/home/jjponnia/Documents/repos/jetstream_macwC/agents-visit-targets/agent_and_target_clustering/plots7', 'output7.gif', duration=200)