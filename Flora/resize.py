from PIL import Image
import os
import sys
import glob


def resize_all_in_fodler():
    folders = glob.glob('train_images/*')
    for folder in folders:
        for file_name in os.listdir(folder):
            try:
                image = Image.open(os.path.join(folder, file_name))
                output_file_name = os.path.join(folder, file_name)
                x, y = image.size
                if (x, y) == (224, 224):
                    continue
                print("Processing %s %s" % (folder, file_name))
                new_dimensions = (224, 224)
                output_file_name = os.path.join(folder, file_name)
                output = image.resize(new_dimensions, Image.ANTIALIAS)
                output.save(output_file_name, "JPEG", quality=100)
            except OSError:
                os.system('rm ' + output_file_name)
                print('Removed: ', output_file_name)
    print("All done")


image = Image.open('image_test.jpg')
x, y = image.size
new_dimensions = (224, 224)
output = image.resize(new_dimensions, Image.ANTIALIAS)
output.save('image_test.jpg', "JPEG", quality=100)
