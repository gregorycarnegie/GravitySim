import PySimpleGUI as Sg
import utils


def finished():
    Sg.popup("Your images have been cropped!\nCheck the reject folder for failures.")


def about_page():
    Sg.popup(
        "", "Face Cropper was created by:", "Gregory Carnegie",
        "Based on François (leblancfg) Leblanc's autocrop library",
        "and Bleed AI's 'Effective Face Detection' article.", "Version: 1.1.0"
    )


class ScrollBar:
    def __init__(self, text, scroll_range, orientate, default_value, key):
        self.text, self.scroll_range, self.orientate = text, scroll_range, orientate
        self.default_value, self.key = default_value, key
        self.size = (60, 20)

    def scroll(self):
        return [
            Sg.Text(self.text), Sg.Slider(range=self.scroll_range, orientation=self.orientate, size=self.size,
                                          default_value=self.default_value, key=self.key)
        ]


class Browser:
    def __init__(self, loc, key):
        self.loc, self.key = loc, key
        self.size = (49, 1)

    def browse(self):
        return [[Sg.Text(self.loc)], [Sg.Input(key=self.key, size=self.size)], [Sg.FolderBrowse(target=self.key)]]


picture_frame = [[Sg.Image("logo.png")]]

instructions = [
    [Sg.Text("Please check that your images\n"
             "are fully supported by the\n"
             "Pillow Python image Library\n"
             "(e.g. JPG, PNG etc...)")]
]

frame0, frame1 = Browser("Source Folder", "SOURCE").browse(), Browser("Destination Folder", "DESTINATION").browse()

wid = ScrollBar("Width (px)", (200, 500), 'h', 300, "WIDTH")
hei = ScrollBar("Height (px)", (200, 500), 'h', 485, "HEIGHT")
pad = ScrollBar("Padding (px)", (0, 20), 'h', 0, "PADDING")
gam = ScrollBar("Gamma", (2000, 0), 'h', 1000, "GAMMA")
fac = ScrollBar("Face %", (1, 100), 'h', 62, "FACE")
con = ScrollBar("Confidence %", (1, 100), 'h', 50, "CONF")

radio = []
for n in range(len(utils.radio_choices)):
    if n == 0:
        radio.append(Sg.Radio(utils.radio_choices[n], 1, default=True, key=f"CONVERT_{str(n)}"))
    else:
        radio.append(Sg.Radio(utils.radio_choices[n], 1, key=f"CONVERT_{str(n)}"))

frame2 = [wid.scroll(), hei.scroll(), pad.scroll(), gam.scroll(), fac.scroll(), con.scroll(), radio]

button_width, button_height = 7, 1
frame3 = [
    [Sg.Submit(size=(button_width, button_height)), Sg.Exit(size=(button_width, button_height))],
    [Sg.Button("About Face Cropper", size=(2 * (button_width + 1), 1))]
]

layout0 = [[Sg.Frame("", frame0)], [Sg.Frame("", frame1)], [Sg.Frame("", instructions), Sg.Frame("", frame3)]]

layout = [
    [Sg.Column(picture_frame), Sg.VSeparator(), Sg.Column(layout0)],
    [Sg.Frame("SETTINGS", frame2, element_justification="right")],
    [Sg.ProgressBar(100, orientation="h", size=(wid.size[0] - 1, 20), border_width=4, key="progbar",
                    bar_color=("Red", "Green"))]
]
# Start of the program...
window = Sg.Window('Face Cropper', layout)
