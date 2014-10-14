#!/opt/local/bin/python

from Tkinter import *

#from Tkinter import Tk, BOTH
#from ttk import Frame, Button, Style, Combobox

class MainFrame(Frame):

    value_of_combo = 'X'

    def __init__(self, parent):
        Frame.__init__(self, parent)#, background='white')
        self.parent = parent
        self.initUI()

    def initUI(self):

        self.parent.title('TS Tester')
        self.pack(fill=BOTH, expand=1)

        #Buttons:
        quitButton = Button(self, text='Quit', foreground='red',command=self.quit)
        quitButton.place(x=560, y=450)

        tButton = Button(self, text='Test', foreground='black',command=self.say_hi)
        tButton.place(x=500, y=450)


        lb=Label(self, text="Example")
        lb.place(x=10,y=10)
        scrollbar = Scrollbar(self, orient=VERTICAL)
        scrollbar.pack( side = RIGHT, fill=Y )
        listExamples = Listbox(self,yscrollcommand = scrollbar.set )
        scrollbar.config( command = listExamples.yview )
        #listExamples.place(x=10,y=30)
        listExamples.pack( side = LEFT, fill = BOTH )
        #scrollbar.pack(side=RIGHT, fill=Y)
        
        listExamples.insert(END, "a list entry")
        for item in ["one", "two", "three", "four"]:
            listExamples.insert(END, item)


        #print tButton.bbox()

        #quitButton.grid(row=3, column=3)
        #tButton.grid(row=3, column=2)

        #Window Positioning
        self.centerWindow()

        #self.box_value = 'test'
        #self.box = Combobox(self.parent, textvariable=self.box_value)
        #self.box['values'] = ('X', 'Y', 'Z')
        #self.box.current(0)
        #self.box.grid(column=0, row=0)
    def yview(self, *args):
        apply(listExamples.yview, args)

    def centerWindow(self):

        w = 640
        h = 480

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()

        x = (sw - w)/2
        y = (sh - h)/2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))
    def say_hi(self):
        print 'hi there, everyone!'

def main():

    root = Tk()
    ex = MainFrame(root)
    root.mainloop()

if __name__ == '__main__':
    main()
