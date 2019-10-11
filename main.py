# load libraries
from comet_ml import Experiment
from file_manager import FileManager
from os.path import join
from art_net import ArtNet
import torch


def main():

    hyper_params = {
        "learning_rate": 1,
        "style_weight": 1000000,
        "content_weight": 1,
        "n_steps": 300
    }

    experiment = Experiment(api_key="a604AfX0S9Bmt6HdpMHxg9MCI",
                            project_name="style-transfer", workspace="polmonroig")

    experiment.log_parameters(hyper_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Currently using cuda device.")

    # define file paths and directories
    content_images_dir = "content_images/"
    style_images_dir = "style_images/"
    output_images_dir = "output_images/"
    content_image_name = "content_00.jpg"
    style_image_name = "style_00.jpg"
    output_image_path = join(output_images_dir,
                             content_image_name.split('.')[0] + "_" + style_image_name.split('.')[0] + ".jpg")
    content_image_path = join(content_images_dir, content_image_name)
    style_image_path = join(style_images_dir, style_image_name)

    # define image file manager
    max_shape = (800, 800)
    fileManager = FileManager(content_image_path, style_image_path,
                              device, max_shape)

    # read images
    content_image, style_image = fileManager.read_images()
    input_image = content_image.clone()
    model = ArtNet(device=device)
    output_image = model.train(hyper_params['content_weight'], hyper_params['style_weight'],
                               hyper_params['n_steps'], content_image, style_image, input_image,
                               hyper_params['learning_rate'], experiment)

    fileManager.save_image(output_image, output_image_path)

    experiment.log_image(output_image_path, content_image_name.split('.')[0] + "_" + style_image_name.split('.')[0])







if __name__ == "__main__":
    main()
