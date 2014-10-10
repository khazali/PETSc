#!/opt/local/bin/python

from Tkinter import Tk, BOTH
from ttk import Frame, Button, Style, Combobox

class Example(Frame):

    value_of_combo = 'X'

    def __init__(self, parent):
        Frame.__init__(self, parent)#, background="white")
        self.parent = parent
        self.initUI()

    def initUI(self):

        self.parent.title("Quit button")
        self.style = Style()
        self.style.theme_use("default")
        self.style.configure("TFrame",background="white")
        self.pack(fill=BOTH, expand=1)

        quitButton = Button(self, text="Quit",
            command=self.quit)
        quitButton.place(x=50, y=50)
        self.centerWindow()

        self.box_value = StringVar()
        self.box = Combobox(self.parent, textvariable=self.box_value)
        self.box['values'] = ('X', 'Y', 'Z')
        self.box.current(0)
        self.box.grid(column=0, row=0)

    def centerWindow(self):
      
        w = 640
        h = 480

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        
        x = (sw - w)/2
        y = (sh - h)/2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))

def main():
  
    root = Tk()
    ex = Example(root)
    root.mainloop()  

if __name__ == '__main__':
    main() 
