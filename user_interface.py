import tkinter as tk
import numpy as np

# CONSTANTS
EA = "evolutionary algorithm"
BO = "bayesian optimization"
MFC = "MedianFoldChange"

class Parameter:
    def __init__(self,name,  t, value):
        if not isinstance(value, t):
            raise ValueError("{} is not of type {}".format(value, t))
        self.name = name
        self.type = t
        self.value = value
        tk.IntVar()


    def determine_var(self, t):
        tk_vars = {int:tk.IntVar, float:tk.DoubleVar, str:tk.StringVar, bool:tk.BooleanVar}
        return tk_vars[t]

    def get_name(self):
        return self.name

    def get_type(self):
        return self.type

    def get_value(self):
        return self.var.get()

    def set_var(self, widget):
        self.var = self.determine_var(self.type)(widget, self.value, self.name)

    def get_var(self):
        return self.var


class App(tk.Frame):

    def __init__(self, master=None):
        super(App, self).__init__(master)
        self.pack()
        # param_boundaries = np.array([[900, 1100], [90000, 110000], [3000000, 32000000],
        #                              [0, 1], [0, 0.0001], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]])
        self.parameters  = [
            Parameter("time_sim", int, 1000),
            Parameter("number_of_molecules", int, 100000),
            Parameter("monomer_pool", int, 3100000),
            Parameter("p_growth", float, 0.5),
            Parameter("p_death", float, 0.00005),
            Parameter("p_dead_react", float, 0.5),
            Parameter("l_exponent", float, 0.5),
            Parameter("d_exponent", float, 0.5),
            Parameter("l_naked", float, 0.5),
            Parameter("kill_spawns_new", bool, True)
        ]

        # EA: #iterations, population size, fitness_function
        self.alg_parameters = {
            EA: [Parameter("#iterations",int, value=25),
                 Parameter("population_size", int, 10),
                 Parameter("fitness_function", str, MFC)],
            BO:[Parameter("#iterations", int, 10)]
        }
        [x.set_var(self) for x in self.parameters]
        self.register_validation_functions()
        self.build_alg_selector()
        self.build_widgets()

    def build_alg_selector(self):
        lf = tk.LabelFrame(self, text="algorithm")
        lf.grid(column=1, row=0, sticky=tk.N)
        self.alg_list = tk.Listbox(lf, height=2, exportselection=0)
        self.alg_list.insert(tk.END, EA)
        self.alg_list.insert(tk.END, BO)
        self.alg_list.bind('<<ListboxSelect>>', self._switch_alg_parameters)
        self.alg_list.grid()
        # Force the settings to be shown
        # build frames for each alg
        self.alg_frames = {}
        for alg in self.alg_parameters:
            f = tk.LabelFrame(self, text=alg)
            params = self.alg_parameters[alg]
            for param in params:
                param.set_var(self)
                param_type = param.get_type()
                lab = tk.Label(f, text=param.get_name())
                lab.pack()
                control = tk.Entry(f,
                                   validate='all',
                                   validatecommand=(self.validation[param_type], '%s', '%P', '%W'),
                                   name=param.get_name(),
                                   textvar=param.get_var())
                control.pack()
            self.alg_frames[alg] = f
        self.current_alg_frame = None
        self.alg_list.select_set(0)
        self._switch_alg_parameters(None)

    def _switch_alg_parameters(self, event):
        # For some reason this method is called when tab is pressed in other fields.... :(
        try:
            selected = self.alg_list.get(self.alg_list.curselection()[0])
        except IndexError:
            return
        if self.current_alg_frame is not None:
            self.current_alg_frame.grid_forget()
        frame = self.alg_frames[selected]
        frame.grid(column=1, row=1)
        self.current_alg_frame = frame

    def build_widgets(self):
        self.param_controls = {}
        self.parameter_frame = tk.LabelFrame(self, text="initial parameters")
        self.parameter_frame.grid(column=0, row=0, rowspan=2)

        for param in self.parameters:
            control = None
            param_type = param.get_type()
            if param_type in [int, str, float]:
                lab = tk.Label(self.parameter_frame, text=param.get_name())
                lab.pack()
                control = tk.Entry(self.parameter_frame,
                                   validate='all',
                                   validatecommand=(self.validation[param_type], '%s', '%P', '%W'),
                                   name=param.get_name(),
                                   textvar=param.get_var())
            elif param_type is bool:
                control = tk.Checkbutton(self.parameter_frame, text=param.get_name(), name=param.get_name())

            self.param_controls[str(control)] = control
            control.pack()
        self.run = tk.Button(self, text="Run algorithm", command=self.run)
        self.run.grid(column=0, columnspan=2)

    def run(self):
        selected_alg = self.alg_list.get(self.alg_list.curselection()[0])
        print(selected_alg)
        print([x.get_value() for x in self.alg_parameters[selected_alg]])
        print(np.array([x.get_value() for x in self.parameters]))

    def register_validation_functions(self):
        self.validation = {}
        self.validation[int] = self.register(self.is_int)
        self.validation[float] = self.register(self.is_float)
        self.validation[str] = self.register(self.is_str)
        self.validation[bool] = self.register(self.is_bool)


    def is_int(self, s_old, s_new, name):
        if not self._try_cast(s_new, int):
            self.bell()
            return False
        return True

    def is_str(self, s_old, s_new, name):
        if not self._try_cast(s_new, str):
            self.bell()
            return False
        return True

    def is_float(self, s_old, s_new, name):
        if not self._try_cast(s_new, float):
            self.bell()
            return False
        return True

    def is_bool(self, s_old, s_new, name):
        return s_new.lower() in ['true', 'false', '0' '1']

    def _try_cast(self, val, t):
        try:
            t(val)
            return True
        except ValueError:
            return False



if __name__ == '__main__':

    root = tk.Tk()
    app = App(root)
    root.mainloop()

