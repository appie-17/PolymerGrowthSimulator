import tkinter as tk
import numpy as np
import tkinter.ttk as ttk
import time
import imageio
from simulation import polymer
from distributionComparison import medianFoldNorm, minMaxNorm, translationInvariant
from evolutionaryAlgorithm import EvolutionaryAlgorithm
from bayesianOptimization import bayesianOptimisation



# CONSTANTS
EA = "evolutionary algorithm"
BO = "bayesian optimization"


class Parameter:
    def __init__(self,name,  t, value, widget=None):
        if not isinstance(value, t):
            raise ValueError("{} is not of type {}".format(value, t))
        self.name = name
        self.type = t
        self.value = value
        if widget is not None:
            self.set_var(widget)

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

validation = {}
def register_validation_functions(widget):
    validation[int] = widget.register(is_int)
    validation[float] = widget.register(is_float)
    validation[str] = widget.register(is_str)
    validation[bool] = widget.register(is_bool)

def is_int( s_old, s_new, name):
    if not _try_cast(s_new, int):
        return False
    return True

def is_str( s_old, s_new, name):
    if not _try_cast(s_new, str):
        return False
    return True

def is_float( s_old, s_new, name):
    if not _try_cast(s_new, float):
        return False
    return True

def is_bool( s_old, s_new, name):
    return s_new.lower() in ['true', 'false', '0' '1']

def _try_cast( val, t):
    try:
        t(val)
        return True
    except ValueError:
            return False

MFC = "MedianFoldChange"
MM = "MaxMinNorm"
TIMFC = "Translation invariant MFC"
class CostFunctionChooser:
    def __init__(self, parent):
        self.parent = parent
        lf = tk.LabelFrame(parent, text="difference function")
        file_parameter = Parameter("filename", str, "Data\polymer_20k.xlsx")
        file_parameter.set_var(parent)
        sigma_parameter = Parameter("weights", str, "1,1,1,1,1,1", parent)
        f_parameter = Parameter("transfac", float,2.0, parent)
        self.parameters = {
            MFC: [file_parameter, sigma_parameter],
            TIMFC: [file_parameter, sigma_parameter, f_parameter],
            MM: [file_parameter]
        }
        self.function_list = tk.Listbox(lf, height=len(self.parameters), exportselection=0)
        self.functions = [MM, MFC, TIMFC]
        [self.function_list.insert(tk.END, x) for x in self.functions]
        self.function_list.bind('<<ListboxSelect>>', self.switch)
        self.function_list.select_set(0)
        self.function_list.pack(side=tk.BOTTOM)
        lf.pack()

        param_frame = tk.LabelFrame(self.parent, text="parameters")
        param_frame.pack()
        self.function_frames = self.build_frames(self.parameters, param_frame)
        self.current_frame = None
        self.switch(None)


        self.parameter_frame = tk.LabelFrame(parent, text="parameters")
        self.parameter_frame.pack()

    def switch(self, event):
        if self.current_frame is not None:
            self.current_frame.pack_forget()

        index = self.function_list.curselection()[0]
        func = self.functions[index]
        new_frame = self.function_frames[func]
        new_frame.pack()
        self.current_frame = new_frame



    def build_frames(self, functions, parent_widget):
        frames = {}
        for function in functions:
            parent = tk.Frame(parent_widget)
            for param in self.parameters[function]:
                control = None
                param_type = param.get_type()
                if param_type in [int, str, float]:
                    lab = tk.Label(parent, text=param.get_name())
                    lab.pack()
                    control = tk.Entry(parent,
                                       validate='all',
                                       validatecommand=(validation[param_type], '%s', '%P', '%W'),
                                       name=param.get_name(),
                                       textvar=param.get_var())
                elif param_type is bool:
                    control = tk.Checkbutton(parent, text=param.get_name(), name=param.get_name())
                control.pack()
            frames[function] = parent
        return frames

    def get_function(self):
        index = self.function_list.curselection()[0]
        func = self.functions[index]
        return func, self.parameters[func]


class App(tk.Frame):
    def __init__(self, master=None):
        super(App, self).__init__(master)
        self.pack()

        self.fig = Figure()
        self.result_frame = tk.LabelFrame(self, text="result")
        self.plot_window = FigureCanvasTkAgg(self.fig, self.result_frame)
        self.plot_window.draw()
        self.plot_window.get_tk_widget().pack()
        self.result_frame.grid(column=2, row=0, rowspan=2)

        self.function_frame = tk.LabelFrame(self, text="cost function")
        self.function_chooser = CostFunctionChooser(self.function_frame)
        self.function_frame.grid(column=1, row=1)

        # param_boundaries = np.array([[900, 1100], [90000, 110000], [3000000, 32000000],
        #                              [0, 1], [0, 0.0001], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]])
        self.lower_parameters  = [
            Parameter("l_time_sim", int, 1000),
            Parameter("l_number_of_molecules", int, 100000),
            Parameter("l_monomer_pool", int, 3100000),
            Parameter("l_p_growth", float, 0.5),
            Parameter("l_p_death", float, 0.00005),
            Parameter("l_p_dead_react", float, 0.5),
            Parameter("l_l_exponent", float, 0.5),
            Parameter("l_d_exponent", float, 0.5),
            Parameter("l_l_naked", float, 0.5),
            Parameter("l_kill_spawns_new", bool, True)
        ]
        self.upper_parameters = [Parameter("u_" + x.get_name()[2:], x.get_type(), x.value) for x in self.lower_parameters]
        # EA: #iterations, population size, fitness_function
        self.alg_parameters = {
            EA: [Parameter("#iterations",int, value=25),
                 Parameter("population_size", int, 10)
                # ,Parameter("distribution file location", str, "Data\polymer_20k.xlsx")
                ],
            BO:[Parameter("#iterations", int, 10)]
        }
        [x.set_var(self) for x in self.lower_parameters]
        [x.set_var(self) for x in self.upper_parameters]
        self.build_alg_selector()
        self.build_widgets()

    def build_alg_selector(self):
        lf = tk.LabelFrame(self, text="algorithm")
        lf.grid(column=1, row=0, sticky=tk.N)
        self.alg_list = tk.Listbox(lf, height=2, exportselection=0)
        self.alg_list.insert(tk.END, EA)
        self.alg_list.insert(tk.END, BO)
        self.alg_list.bind('<<ListboxSelect>>', self._switch_alg_parameters)
        self.alg_list.pack()
        # Force the settings to be shown
        # build frames for each alg
        self.alg_frames = {}
        for alg in self.alg_parameters:
            f = tk.LabelFrame(lf, text=alg)
            params = self.alg_parameters[alg]
            for param in params:
                param.set_var(self)
                param_type = param.get_type()
                lab = tk.Label(f, text=param.get_name())
                lab.pack()
                control = tk.Entry(f,
                                   validate='all',
                                   validatecommand=(validation[param_type], '%s', '%P', '%W'),
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
            self.current_alg_frame.pack_forget()
        frame = self.alg_frames[selected]
        frame.pack()#column=1, row=1)
        self.current_alg_frame = frame

    def build_widgets(self):
        self.param_controls = {}
        self.parameter_frame = tk.LabelFrame(self, text="initial parameters")
        self.lower_frame = tk.LabelFrame(self.parameter_frame, text="lower bounds")
        self.upper_frame = tk.LabelFrame(self.parameter_frame, text="upper bounds")
        self.parameter_frame.grid(column=0, row=0, rowspan=2)

        self.lower_frame.grid(column=0, row=0)
        self.upper_frame.grid(column=1, row=0)
        self.create_parameter_controls(self.lower_parameters, self.lower_frame)
        self.create_parameter_controls(self.upper_parameters, self.upper_frame)

        self.run = tk.Button(self, text="Run algorithm", command=self.run)
        self.run.grid(column=0, columnspan=2)

    def create_parameter_controls(self, params, parent):
        for param in params:
            control = None
            param_type = param.get_type()
            if param_type in [int, str, float]:
                lab = tk.Label(parent, text=param.get_name())
                lab.pack()
                control = tk.Entry(parent,
                                   validate='all',
                                   validatecommand=(validation[param_type], '%s', '%P', '%W'),
                                   name=param.get_name(),
                                   textvar=param.get_var())
            elif param_type is bool:
                control = tk.Checkbutton(parent, text=param.get_name(), name=param.get_name())

            self.param_controls[str(control)] = control
            control.pack()

    def run(self):
        selected_alg = self.alg_list.get(self.alg_list.curselection()[0])
        alg_parameters = [x.get_value() for x in self.alg_parameters[selected_alg]]

        low = np.array([x.get_value() for x in self.lower_parameters])
        high = np.array([x.get_value() for x in self.upper_parameters])
        print(selected_alg)
        func, func_alg = self.function_chooser.get_function()

        if func == MFC:
            file_name = func_alg[0].get_value()
            sigma_str = func_alg[1].get_value()
            sigma = np.array([float(x) for x in sigma_str.split(",")])
            norm = medianFoldNorm(file_name, polymer, sigma, fig=self.fig)
        elif func == MM:
            file_name = func_alg[0].get_value()
            norm = minMaxNorm(file_name, polymer, fig=self.fig)
        elif func == TIMFC:
            file_name = func_alg[0].get_value()
            sigma_str = func_alg[1].get_value()
            sigma = np.array([float(x) for x in sigma_str.split(",")])
            transfac = func_alg[2].get_value()
            norm = translationInvariant(file_name, polymer, sigma, transfac, fig=self.fig)

        bound = np.stack((low,high),1)
        if selected_alg == EA:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]

            ea = EvolutionaryAlgorithm(bound, pop_size, norm.costFunction)
            for i in range(iterations):
                print("iteration", i)
                f = ea.run(1)
                self.plot_window.draw()
                result = ea.get_best_individual(f)
                self.update()
        elif selected_alg == BO:
            iterations = alg_parameters[0]
            result = bayesianOptimisation(iterations, norm.costFunction, bound, 10)


        # print([x.get_value() for x in self.alg_parameters[selected_alg]])
        x = tk.Text(self, height=2)
        x.delete(0.1, tk.END)
        x.insert(tk.END, str(result))

        x.grid(column=0, columnspan=2, row=3)




class SimulationPage(tk.Frame):

    def __init__(self, master=None):
        super(SimulationPage, self).__init__(master)
        self.parameter_frame = tk.LabelFrame(self, text="parameters")
        self.build_parameters()

        self.result_frame = tk.LabelFrame(self, text="result")
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.plot_window = FigureCanvasTkAgg(self.fig, self.result_frame)
        self.plot_window.draw()
        self.plot_window.get_tk_widget().pack()
        self.pack()

        self.run = tk.Button(self, text="Run algorithm", command=self.run)
        self.parameter_frame.grid(column=0, row=0)
        self.run.grid(column=0, row=1)
        self.result_frame.grid(column=1, row=0, columnspan=2, rowspan=2)



    def build_parameters(self):
        self.video = Parameter("video", bool, True)
        self.save_video = Parameter("save video", bool, True)
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
            Parameter("kill_spawns_new", bool, True),
            self.video,
            self.save_video
        ]

        [x.set_var(self) for x in self.parameters]
        for param in self.parameters:
            control = None
            param_type = param.get_type()
            if param_type in [int, str, float]:
                lab = tk.Label(self.parameter_frame, text=param.get_name())
                lab.pack()
                control = tk.Entry(self.parameter_frame,
                                   validate='all',
                                   validatecommand=(validation[param_type], '%s', '%P', '%W'),
                                   name=param.get_name(),
                                   textvar=param.get_var())
            elif param_type is bool:
                control = tk.Checkbutton(self.parameter_frame, text=param.get_name(), name=param.get_name())


            control.pack()



    def run(self):
        values = [x.get_value() for x in self.parameters if x.get_name() not in ["video", "save video"]]

        self.run["text"] = "running"
        self.update()
        if self.save_video.get_value() and self.video.get_value():
            name = str(round(time.time()))
            self.writer = imageio.get_writer(name + '.gif', mode='I', duration=0.3)
        if self.video.get_value():
            result = polymer(*values, UI_vid=self.make_hist)
        else:
            result = polymer(*values)
            self.make_hist(result)

        if self.save_video.get_value() and self.video.get_value():
            self.writer.close()

        self.run["text"] = "run"


    def make_hist(self, results, state=None, coloured=1):
        self.ax.clear()
        living, dead,coupled = results
        if state is not None:
            current_monomer, initial_monomer, time = state
            conversion = 1 - current_monomer / initial_monomer
        d = np.hstack((living, dead, coupled))
        DPn = np.mean(d)
        DPw = np.sum(np.square(d)) / (DPn * d.shape[0])
        PDI = DPw / DPn
        # dlmwrite('polymerOutput.txt',[time, conversion, DPn, DPw, PDI], '-append');
        if coloured == 0:
            self.ax.hist(d, bins=int(np.max(d) - np.min(d)), facecolor='b')
        else:
            step = np.ceil((np.max(d) - np.min(d)) / 1000)
            binEdges = np.arange(np.min(d) - 0.5, np.max(d) + 0.5, step)
            midbins = binEdges[0:-1] + (binEdges[1:] - binEdges[0:-1]) / 2
            if coupled.size == 0:
                c,b,e = self.ax.hist([dead, living], bins=midbins, histtype='barstacked', stacked=False, label=['Dead', 'Living'])
                # e[0]["color"] = "blue"
                # e[1]["color"] = "orange"
                matplotlib.pyplot.setp(e[0], color="blue")
                matplotlib.pyplot.setp(e[1], color="orange")
                # setp(e[0], color='blue')
                # setp(e[1], color='orange')


            else:
                self.ax.hist([coupled, dead, living], bins=midbins, histtype='bar', stacked=True,
                             label=['Terminated', 'Dead', 'Living'])

        self.ax.set_xlabel('Length in units')
        self.ax.set_ylabel('Frequency')
        digits = 3
        if state is not None:
            title = "conversion={}, t={}, DPI={}, DPn={}, DPw={}".format(
                round(conversion, digits), time, round(PDI, digits),round(DPn, digits),round(DPw, digits)
            )
            self.ax.set_title(title)

        self.ax.legend()
        self.plot_window.draw()

        if self.save_video.get_value():
            width, height = self.plot_window.get_width_height()
            image = np.fromstring(self.plot_window.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
            self.writer.append_data(image)

class ApplicationNotebook(ttk.Notebook):

    def __init__(self, master=None):
        super(ApplicationNotebook, self).__init__(master)
        register_validation_functions(self)
        self.add(App(self), text="optimization")
        self.add(SimulationPage(self), text="Simulation")
        self.pack()


if __name__ == '__main__':
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
    from matplotlib.figure import Figure
    root = tk.Tk()
    # app = App(root)
    root.title("Optimization")
    app = ApplicationNotebook(root)
    root.mainloop()


