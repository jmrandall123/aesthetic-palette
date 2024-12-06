from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import numpy as np
from typing import Tuple, List
import os

class ImagePaletteGenerator:
    def __init__(self, target_aspect_ratio: float = 2.35, output_width: int = 750,
                 num_colors: int = 10, padding: int = 5, palette_height_ratio: float = 0.35):
        """
        Initialize the ImagePaletteGenerator.
        
        Args:
            target_aspect_ratio: Desired aspect ratio for the main image (width/height)
            output_width: Width of the output image in pixels
            num_colors: Number of color swatches in the palette
            padding: Padding between elements in pixels
            palette_height_ratio: Height of palette relative to main image height (e.g., 0.25 = 25% of image height)
        """
        self.target_aspect_ratio = target_aspect_ratio
        self.output_width = output_width
        self.num_colors = num_colors
        self.padding = padding
        self.output_height = int(output_width / target_aspect_ratio)
        self.palette_height_ratio = palette_height_ratio

    def load_image(self, image_path: str) -> Image.Image:
        """Load and validate image file."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            raise Exception(f"Error loading image: {e}")

    def crop_to_aspect_ratio(self, img: Image.Image) -> Image.Image:
        """Crop image to target aspect ratio from center."""
        original_width, original_height = img.size
        original_aspect_ratio = original_width / original_height

        if abs(original_aspect_ratio - self.target_aspect_ratio) < 0.01:
            return img

        if original_aspect_ratio > self.target_aspect_ratio:
            # Wider than target, crop sides
            new_width = int(original_height * self.target_aspect_ratio)
            left = (original_width - new_width) // 2
            return img.crop((left, 0, left + new_width, original_height))
        else:
            # Taller than target, crop top/bottom
            new_height = int(original_width / self.target_aspect_ratio)
            top = (original_height - new_height) // 2
            return img.crop((0, top, original_width, top + new_height))

    def extract_colors(self, img: Image.Image) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering."""
        # Resize for faster processing while maintaining color distribution
        img_small = img.resize((150, 150), Image.Resampling.LANCZOS)
        pixels = np.float32(img_small).reshape(-1, 3)
        
        # Use k-means++ initialization for better color selection
        kmeans = KMeans(n_clusters=self.num_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        # Sort colors by brightness for aesthetic ordering
        colors = kmeans.cluster_centers_
        colors = sorted(colors, key=lambda x: sum(x))  # Sort by brightness
        return [tuple(map(int, color)) for color in colors]

    def create_palette(self, colors: List[Tuple[int, int, int]]) -> Image.Image:
        """Create color palette image with exact width match to main image."""
        palette_height = int(self.output_height * self.palette_height_ratio)
        
        # Calculate swatch width to exactly match image width
        total_padding = self.padding * (self.num_colors - 1)  # Only internal padding
        swatch_width = (self.output_width - total_padding) // self.num_colors
        
        # Adjust last swatch width to account for any rounding
        remaining_width = self.output_width - ((swatch_width * (self.num_colors - 1)) + total_padding)
        
        palette = Image.new("RGB", (self.output_width, palette_height), "white")
        draw = ImageDraw.Draw(palette)
        
        current_x = 0
        for i, color in enumerate(colors):
            # Use remaining width for the last swatch
            if i == self.num_colors - 1:
                width = remaining_width
            else:
                width = swatch_width
            
            draw.rectangle(
                [current_x, 0, current_x + width, palette_height],
                fill=color
            )
            
            current_x += width + (self.padding if i < self.num_colors - 1 else 0)
        
        return palette

    def combine_image_and_palette(self, img: Image.Image, palette: Image.Image) -> Image.Image:
        """Combine main image and color palette."""
        spacing = self.padding
        canvas_width = self.output_width + 2 * self.padding
        canvas_height = (self.output_height + palette.height + spacing + 
                        2 * self.padding)
        
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        canvas.paste(img, (self.padding, self.padding))
        canvas.paste(palette, (self.padding, self.output_height + spacing + self.padding))
        
        return canvas

    def process_image(self, image_path: str, output_path: str = None) -> Image.Image:
        """Process image and generate palette."""
        try:
            # Load and prepare image
            img = self.load_image(image_path)
            img = self.crop_to_aspect_ratio(img)
            img = img.resize((self.output_width, self.output_height), Image.Resampling.LANCZOS)
            
            # Extract colors and create palette
            colors = self.extract_colors(img)
            palette = self.create_palette(colors)
            
            # Combine and save
            final_image = self.combine_image_and_palette(img, palette)
            
            if output_path:
                final_image.save(output_path, "JPEG", quality=95, dpi=(300, 300))
                print(f"Saved final image as '{output_path}'")
            
            return final_image
            
        except Exception as e:
            print(f"Error processing image: {e}")
            raise

def main():
    # Example usage
    generator = ImagePaletteGenerator()
    image_path = input("Enter the path to your image file: ").strip('"')
    output_path = "final_image_cinematic_with_palette.jpg"
    
    try:
        final_image = generator.process_image(image_path, output_path)
        final_image.show()
    except Exception as e:
        print(f"Failed to process image: {e}")

if __name__ == "__main__":
    main()