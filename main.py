import os
import PySimpleGUI as Sg
import utils
import ux


class ImageReadError(BaseException):
    """Custom exception to catch an OpenCV failure type."""
    pass


# Start of the program...
window = ux.window
if __name__ == "__main__":
    while True:
        event, values = window.read()

        if event == Sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == "About Face Cropper":
            ux.about_page()

        if event == "Submit":
            source, destination = values["SOURCE"], values["DESTINATION"]
            width, height = values["WIDTH"], values["HEIGHT"]
            padding, confidence, face, gamma = values["PADDING"], values["CONF"], values["FACE"], values["GAMMA"]
            file_list = os.listdir(source)
            # Appending radio values to list
            radio_values = []
            for i in range(len(utils.radio_choices)):
                radio_values.append(values["CONVERT_" + str(i)])

            v = 1
            for image in file_list:
                # ignore non-images
                ext = os.path.splitext(image)[1]
                if ext.lower() not in utils.COMBINED_FILETYPES:
                    continue
                utils.process(image, source, destination, padding, width, height, confidence, face, gamma, radio_values)
                v += 1
                val = int(100 * (v / len(file_list)))
                window['progbar'].update_bar(val)

            ux.finished()
