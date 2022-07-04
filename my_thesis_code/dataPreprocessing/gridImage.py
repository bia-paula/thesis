from PIL import Image, ImageDraw

if __name__ == '__main__':
    x_step_count = 16
    y_step_count = 10
    height = 320
    width = 512
    image = Image.open('/Volumes/DropSave/Tese/dataset/resized_images/bottle/000000001455.jpg')

    # Draw some lines
    draw = ImageDraw.Draw(image)
    draw.rectangle([274, 5, 274 + 32, 5 + 74], outline="black")
    y_start = 0
    y_end = image.height
    x_step_size = int(image.width / x_step_count)
    y_step_size = int(image.height / y_step_count)

    for x in range(0, image.width, x_step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, y_step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)

    del draw

    image.show()