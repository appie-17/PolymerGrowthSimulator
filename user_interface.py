import tkinter as tk
import numpy as np
import tkinter.ttk as ttk
from simulation import polymer
from distributionComparison import medianFoldNorm
from evolutionaryAlgorithm import EvolutionaryAlgorithm
from bayesianOptimization import bayesianOptimisation


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
                 Parameter("distribution file location", str, "Data\polymer_20k.xlsx")],
            BO:[Parameter("#iterations", int, 10)]
        }
        [x.set_var(self) for x in self.parameters]
        register_validation_functions(self)
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
                                   validatecommand=(validation[param_type], '%s', '%P', '%W'),
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
        alg_parameters = [x.get_value() for x in self.alg_parameters[selected_alg]]
        file_name = [x.get_value() for x in self.alg_parameters[selected_alg] if x.get_name() == "distribution file location"][0]
        bounds = np.array([x.get_value() for x in self.parameters])
        print(selected_alg)
        print(file_name)
        norm = medianFoldNorm(file_name, polymer, fig=self.fig)
        if selected_alg == EA:
            low = bounds - bounds/10
            high = bounds - bounds/10
            bound = np.stack((low,high),1)
            pop_size = alg_parameters[1]
            iterations = alg_parameters[0]

            ea = EvolutionaryAlgorithm(bound, pop_size, norm.costFunction)
            for i in range(iterations):
                print("iteration", i)
                f = ea.run(1)
                self.plot_window.draw()
                ind = ea.get_best_individual(f)


        print([x.get_value() for x in self.alg_parameters[selected_alg]])
        print()
        x = tk.Text(self, text=str(ind))
        x.grid(column=0, columnspan=2)


class SimulationPage(tk.Frame):

    def __init__(self, master=None):
        super(SimulationPage, self).__init__(master)
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        # self.ax.plot([1,2,3,4,5], [5,4,3,7,3])
        self.parameter_frame = tk.LabelFrame(self, text="parameters")
        self.build_parameters()
        self.result_frame = tk.LabelFrame(self, text="result")
        self.plot_window = FigureCanvasTkAgg(self.fig, self.result_frame)
        self.plot_window.draw()
        self.plot_window.get_tk_widget().pack()
        self.pack()
        self.result_frame.grid(column=1, row=0)
        self.parameter_frame.grid(column=0, row=0)


    def build_parameters(self):
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
        self.run = tk.Button(self, text="Run algorithm", command=self.run)
        self.run.grid(row=1, column=0, columnspan=2)

    def run(self):
        values = [x.get_value() for x in self.parameters]
        self.run["text"] = "running"
        result = polymer(*values)
        self.make_hist(result)
        self.run["text"] = "run"
        print(result)


    def make_hist(self, results, coloured=1):
        self.ax.clear()
        living, dead,coupled = results
        d = np.hstack((living, dead, coupled))
        DPn = np.mean(d)
        DPw = np.sum(np.square(d)) / (DPn * d.shape[0])
        PDI = DPw / DPn
        #conversion = 1 - current_monomer / initial_monomer
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
        # ax.set_title(['conversion=', conversion, 'time=', time, 'PDI=', PDI, 'DPn=', DPn, 'DPw=', DPw])
        self.ax.legend()
        self.plot_window.draw()

class ApplicationNotebook(ttk.Notebook):

    def __init__(self, master=None):
        super(ApplicationNotebook, self).__init__(master)
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
    app = ApplicationNotebook(root)
    root.mainloop()


