#version.regex
version__= '1.0.3'

from kivy.app import App
from kivy.uix.label import Label


class SimpleApp(App):
    def build(self):
        return Label(text="Hello World")


if __name__=="__main__":
    SimpleApp().run()