import os
from datetime import datetime


def handle_folder_creation(result_path: str, retrieve_text_file=True):
    """
    Handle the creation of a folder and return a file in which it is possible to write in that folder, moreover
    it also returns the path to the result path with current datetime appended
    :param result_path: basic folder where to store results
    :param retrieve_text_file: whether to retrieve a text file or not
    :return (descriptor to a file opened in write mode within result_path folder. The name of the file
    is result.txt, folder path with the date of the experiment). The file descriptor will be None
    retrieve_text_file is set to False
    """
    date_string = datetime.now().strftime('%b%d_%H-%M-%S/')
    output_folder_path = result_path + date_string
    output_file_name = output_folder_path + "results.txt"
    try:
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
    except FileNotFoundError as e:
        os.makedirs(output_folder_path)

    fd = open(output_file_name, "w") if retrieve_text_file else None

    return fd, output_folder_path
