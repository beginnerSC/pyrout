# sudo apt-get install poppler-utils
# pip install pdf2image

# ctrl-F5 to execute

# Kmad2023

from pdf2image import convert_from_path

if __name__ == '__main__':
    images = convert_from_path('/workspace/pyrout/pdf2image_test/Ecology.pdf')
 
    for i in range(len(images)):
        images[i].save('page'+ str(i) +'.jpg', 'JPEG')