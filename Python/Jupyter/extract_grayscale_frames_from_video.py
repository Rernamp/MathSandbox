import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from PIL import Image, ImageChops
    import imageio.v3 as iio
    import io
    import numpy as np
    import cv2

    return Image, ImageChops, io, mo, np


@app.cell
def _(mo):
    image_file = mo.ui.file(filetypes=[".png"])
    return (image_file,)


@app.cell
def _(image_file, mo):
    _md = mo.md(f"Upload file {image_file}")
    mo.stop(len(image_file.value) == 0, _md)
    image = image_file.value[0]
    _md
    return (image,)


@app.cell
def _(Image, image, io):
    img = Image.open(io.BytesIO(image.contents)).convert("RGB")
    return (img,)


@app.function
def to_grayscale_same_format(img):
    
    original_format = img.format
    
    gray_img = img.convert("L").convert("RGB")
    
    return gray_img


@app.cell
def _(img):
    gray_scale_img = to_grayscale_same_format(img)
    return (gray_scale_img,)


@app.cell
def _(ImageChops, gray_scale_img, img):
    diff = ImageChops.subtract(img, gray_scale_img)
    return (diff,)


@app.cell
def _(diff):
    gray_diff = diff.convert("L")
    return (gray_diff,)


@app.cell
def _(diff, gray_diff, gray_scale_img, img, mo):
    image_to_type = {
        "original": img,
        "grayscale_original": gray_scale_img,
        "diff":diff,
        "diff_grayscale":gray_diff
    }
    dropdown = mo.ui.dropdown(options=image_to_type.keys(), value=next(iter(image_to_type)))
    return dropdown, image_to_type


@app.cell
def _(dropdown, image_to_type, mo):
    viewed_img = image_to_type[dropdown.value]
    mo.vstack([dropdown, viewed_img])
    return


@app.cell
def _(gray_diff, np):
    gray_diff_array = np.array(gray_diff)

    zero_pixel_count = np.count_nonzero(gray_diff_array == 0)
    pixel_count = gray_diff_array.size
    grey_scale_percent = zero_pixel_count / pixel_count
    is_gray_scale_image = grey_scale_percent > 0.9
    return (
        grey_scale_percent,
        is_gray_scale_image,
        pixel_count,
        zero_pixel_count,
    )


@app.cell(hide_code=True)
def _(
    grey_scale_percent,
    is_gray_scale_image,
    mo,
    pixel_count,
    zero_pixel_count,
):
    mo.md(rf"""
    Total pixel count: {pixel_count}

    Black pixel count: {zero_pixel_count}

    Black / total: {grey_scale_percent}

    Is gray scale image: {is_gray_scale_image}
    """)
    return


if __name__ == "__main__":
    app.run()
