import numpy as np
import json
import matplotlib.pyplot as plt

class wave:
    """A class which can be used to determine a nozzle geometry

    Parameters
    ----------
    input_vars : string or dict , optional
        Must be a .json file or python dictionary of the format

    Raises
    ------
    TypeError
        If the input_vars type is not a dictionary or the file path to 
        a .json file
    """

    
    def __init__(self,input_vars={}):

        # # get info or raise error
        # self._get_input_vars(input_vars)

        # retrieve info
        self._retrieve_info()

        # propogate
        self.propogate()

        # plot
        self.plot()


    def _get_input_vars(self,input_vars):
        # get info or raise error

        # determine if the input_vars is a file or a dictionary
        if isinstance(input_vars,dict):
            self.input_dict = input_vars
        
        # json file
        elif isinstance(input_vars,str) and input_vars.split(".")[-1] =="json":
            self.input_dict = self._get_json(input_vars)

        # raise error
        else:
            raise TypeError("input_vars must be json file path, or " + \
                "dictionary, not {0}".format(input_vars_type))
    

    def _get_json(self,file_path):
        # import json file from file path
        json_string = open(file_path).read()

        # save to vals dictionary
        input_dict = json.loads(json_string)
        
        return input_dict


    def _retrieve_info(self):
        """A function which retrieves the information and stores it globally.
        """
        
        # store variables from file input dictionary

        # create vars


        # define variables
        self.speed = 0.3 # m/s
        self.timestep = 1.0 # s
        self.step = self.speed * self.timestep
        self.t_final = 10.0

        self.count = 100

        self.start_point = np.array([1.0,1.0])

        self.shape = np.array([ [0.0,0.0],
                                [0.0,2.0],
                                [2.0,2.0],
                                [2.0,0.0],
                                [0.0,0.0]])
        

    def propogate(self):
        """Method which
        """

        # determine number of time steps
        self.n_steps = int(self.t_final / self.timestep)

        # initialize wave array
        wave = np.zeros((2,self.count,self.n_steps))

        # create initial droplet
        t = np.linspace(0.0,2. * np.pi,num=self.count)
        wave[0,:,0] = self.step / 100. * np.cos(t) + self.start_point[0]
        wave[1,:,0] = self.step / 100. * np.sin(t) + self.start_point[1]

        # run through each time step and determine new wave location
        for i in range(1,self.n_steps):

            # determine new wave location
            wave[0,:,i] = self.step * i * np.cos(t) + self.start_point[0]
            wave[1,:,i] = self.step * i * np.sin(t) + self.start_point[1]


        # make wave global
        self.wave = wave


    def plot(self):
        """Method
        """

        # plot wave propogation
        plt.plot(self.shape[:,0],self.shape[:,1],"k")

        # plot each wave propogation
        for i in range(self.n_steps):
            plt.plot(self.wave[0,:,i],self.wave[1,:,i],"b")
        
        # show plot
        plt.axis("equal")
        plt.show()




waver = wave()