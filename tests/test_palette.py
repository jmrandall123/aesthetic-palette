import importlib.util
from pathlib import Path
from PIL import Image


def load_module():
    module_path = Path(__file__).resolve().parents[1] / 'color selector.py'
    spec = importlib.util.spec_from_file_location('color_selector', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_and_create_palette(tmp_path):
    module = load_module()
    generator = module.ImagePaletteGenerator(output_width=50, num_colors=3)

    img_path = tmp_path / 'test_image.jpg'
    img = Image.new('RGB', (20, 20), (255, 0, 0))
    img.putpixel((0, 0), (0, 255, 0))
    img.save(img_path)

    loaded = generator.load_image(str(img_path))
    colors = generator.extract_colors(loaded)
    assert len(colors) == generator.num_colors

    palette = generator.create_palette(colors)
    assert palette.width == generator.output_width
