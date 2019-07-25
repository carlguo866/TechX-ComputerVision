from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (130, 35), color = "white")

draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/Library/Fonts/Georgia.ttf', 15)
draw.text((10,10), "Hello World",font=font, fill=0)
img.save("test1.png")
img.show()
