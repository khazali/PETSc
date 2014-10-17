#!/opt/local/bin/python

from Tkinter import *
import os

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
        Button_Quit = Button(self, text='Quit', foreground='red',command=self.quit)
        Button_Quit.place(x=730, y=565)

        Button_GenCmdLineOptions = Button(self, text='Build Command Line', foreground='black',command=self.BuildCmdLine)
        Button_GenCmdLineOptions.place(x=570, y=565)

        Button_Run = Button(self, text='Run', foreground='black',command=self.say_hi)
        Button_Run.place(x=510, y=565)

        Button_Convergence = Button(self, text='Convergence Test', foreground='black',command=self.conv_test)
        Button_Convergence.place(x=510, y=535)


        ExampleFrame = Frame(master = self, height=250, width=230, bd=1, relief=SUNKEN)
        ExampleFrame.place(x=10,y=10)

        lb=Label(ExampleFrame, text="Examples")
        lb.place(x=5,y=0)


        ExampleListFrame = Frame(master = ExampleFrame, height=150, width=190, bd=1, relief=SUNKEN)
        ExampleListFrame.place(x=5,y=60)

        scrollbar = Scrollbar(ExampleListFrame, orient=VERTICAL)
        scrollbar.pack( side = RIGHT, fill=Y )
        listExamples = Listbox(ExampleListFrame,yscrollcommand = scrollbar.set )
        scrollbar.config( command = listExamples.yview )
        #listExamples.place(x=10,y=30)

        #scrollbar.pack(side=RIGHT, fill=Y)
        import glob
        listEx=glob.glob("./ex*.c")
        for item in range(len(listEx)):
            listExamples.insert(END, listEx[item][2:-2])
        self.ExampleList=listExamples


        Entry_AddExample = Entry(ExampleFrame, width=12)
        Entry_AddExample.place(x=5,y=25)
        Entry_AddExample.delete(0, END)
        Entry_AddExample.insert(0, "exXXX")
        self.ExampleEntry=Entry_AddExample

        #Button_AddExample = Button(ExampleFrame, text='Add', height=4, width=4,command=self.AddToList(Entry=Entry_AddExample,List=listExamples))
        Button_AddExample = Button(ExampleFrame, text='Add', height=1, width=4,command=self.AddToExampleList)
        Button_AddExample.place(x=135,y=25)

        listExamples.pack( side = LEFT, fill = BOTH )

        Text_CmdBox = Text(master = self, height = 3, width = 50, relief = SUNKEN, borderwidth = 1, background = 'white')
        Text_CmdBox.place(x=10,y=535)
        Text_CmdBox.insert(END, "make  ")

        self.TextCmdLine=Text_CmdBox

        #Window Positioning
        self.centerWindow()

    def yview(self, *args):
        apply(listExamples.yview, args)

    def centerWindow(self):

        w = 800
        h = 600

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()

        x = (sw - w)/2
        y = (sh - h)/2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def say_hi(self):
        print 'hi there, everyone!'

    def BuildCmdLine(self):
        self.TextCmdLine.delete(1.0, END)
        self.TextCmdLine.insert(END,'make ' + self.ExampleList.get(ACTIVE) + ' && ./' + self.ExampleList.get(ACTIVE))

    def conv_test(self):
        str_run='python order_test.py -e '+ self.ExampleList.get(ACTIVE) + ' -p ''-ts_arkimex_type 2e '''
        self.TextCmdLine.delete(1.0, END)
        self.TextCmdLine.insert(END,str_run)
        os.system(str_run)

    def AddToExampleList(self):
        print 'Adding to list :'+ self.ExampleEntry.get()
        self.ExampleList.insert(END, self.ExampleEntry.get())

def main():

    root = Tk()
    ex = MainFrame(root)
    root.mainloop()

if __name__ == '__main__':
    main()
