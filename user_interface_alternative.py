from tkinter import *


class App:
    def __init__(self, master):
        frame = Frame(master)
        #master.geometry('500x500')
        master.title('Polymer Growth Interface')
        frame.grid()

        # ------- Weights ------------- #
        self.is_weight = IntVar()
        self.toggle_weight = Checkbutton(frame, text="Enable Weights", variable=self.is_weight)
        self.toggle_weight.grid(row=2, column=1)

        if(self.is_weight):
            self.w1 = StringVar()
            e1 = Entry(frame, textvariable=self.w1)
            e1.grid(row=2,column=3)

            self.w2 = StringVar()
            e2 = Entry(frame, textvariable=self.w2)
            e2.grid(row=2,column=5)

            self.w3 = StringVar()
            e3 = Entry(frame, textvariable=self.w3)
            e3.grid(row=2,column=7)

            self.w4 = StringVar()
            e4 = Entry(frame, textvariable=self.w4)
            e4.grid(row=2,column=9)

            self.w5 = StringVar()
            e5 = Entry(frame, textvariable=self.w5)
            e5.grid(row=2,column=11)

        # ---------  Simulation Parameters ------------ #
        self.label_parameters = Label(frame, text="Simulatuon Parameters")
        self.label_parameters.grid(row=5, column=1)

        self.parameters_listbox = Listbox(frame, selectmode=SINGLE)
        for item in ['Simulation time', 'Number of molecules', 'Monomer pool', 'Growth probability',
                     'Death probabiity', 'Dead polymer reaction probability', 'Living length exponent',
                     'Dead length exponent', 'Kill spawn flag']:
            self.parameters_listbox.insert(END, item)
        self.parameters_listbox.grid(row=7, column=1)

        self.parameter_value = StringVar()
        e1 = Entry(frame, textvariable=self.parameter_value)
        e1.grid(row=7, column=3)

        # ----------- Evolutionary Algorithm ---------- #
        self.label_ea = Label(frame, text="Evolutionary Algorithm")
        self.label_ea.grid(row=11, column=1)

        # ----------- Baysian Optimisation ------------ #
        self.label_bo = Label(frame, text="Baysian Optimisation")
        self.label_bo.grid(row=11, column=5)

        # ------------ Run Buttons --------------------- #
        self.button_run_ea = Button(frame, text='RUN EA', fg='green', command=self.run_ea)
        self.button_run_ea.grid(row=14, column=1)
        self.button_run_bo = Button(frame, text='RUN BO', fg='green', command=self.run_bo)
        self.button_run_bo.grid(row=14, column=5)

    def run_ea(self):
        return 0

    def run_bo(self):
        return 0


root = Tk()  # root widget
app = App(root)
root.mainloop()
