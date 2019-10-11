# load libraries
from comet_ml import Experiment
from file_manager import FileManager
from os.path import join
import torch


def main():
    # experiment = Experiment(api_key="a604AfX0S9Bmt6HdpMHxg9MCI",
    #                        project_name="style-transfer", workspace="polmonroig")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Currently using cuda device.")

    device = torch.device('cuda')
    content_images_dir = "content_images/"
    style_images_dir = "style_images/"
    output_images_dir = "output_images/"
    content_image_name = "content_00.jpg"
    style_image_name = "style_00.jpg"

    output_image_path = join(output_images_dir,
                             content_image_name.split('.')[0] + "_" + style_image_name.split('.')[0] + ".jpg")
    content_image_path = join(content_images_dir, content_image_name)
    style_image_path = join(style_images_dir, style_image_name)

    fileManager = FileManager(content_image_path, style_image_path,
                              device, (800,800))

    print("Reading images...")
    content_image, style_image = fileManager.read_images()
    print("Done")
    print("Saving output")
    fileManager.save_image(style_image, output_image_path)
    print("Done")



if __name__ == "__main__":
    main()
