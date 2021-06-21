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
        self._propogate()

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
        self.t_final = 5.0

        self.size = 10

        self.start_point = np.array([1.0,1.0])

        self.shape = np.array([ [0.0,0.0],
                                [2.0,0.0],
                                [2.0,2.0],
                                [0.0,2.0],
                                [0.0,0.0]])
        
        # determine shape normals
        self.normals = np.array([   [ 0.0,-1.0, 0.0],
                                    [ 1.0, 0.0, 0.0],
                                    [ 0.0, 1.0, 0.0],
                                    [-1.0, 0.0, 0.0]])
        
        # determine C values
        self.normals[:,2] = self.normals[:,0] * self.shape[:-1,0] + \
        self.normals[:,1] * self.shape[:-1,1]


    def _to_impact(self,particle,direc):
        """Method which
        """
        # flip and negate
        direction = 1. * np.flip(direc)
        direction[1] *= -1.0

        # calculate C value
        C = np.sum( direction * particle )

        # run through each "line" and determine whether it crosses
        segment = []
        crossings = []
        for i in range(self.normals.shape[0]):

            # calculate determinant
            det = direction[0]*self.normals[i,1] - \
                self.normals[i,0]*direction[1]
            if det != 0.0:
                x = (self.normals[i,1]*C - direction[1]*self.normals[i,2])/det
                y = (direction[0]*self.normals[i,2] - self.normals[i,0]*C)/det

                n = (x - particle[0]) / direction[0]
                print(det,particle,direction,self.normals[i],n)
                if n >= 0.0:
                    segment.append(i)
                    crossings.append(np.array([x,y]))
        
        # determine closest crossing
        crossings = np.array(crossings)
        dists = ((crossings[:,0]-particle[0])**2. + \
            (crossings[:,1]-particle[1])**2. )**0.5
        index = np.argwhere(dists == np.min(dists))[0,0]

        
        # determine correct crossings (in direction)
        for i in range(len(crossings)):
            plt.plot(crossings[i][0],crossings[i][1],"ro")
        
        plt.plot(crossings[index][0],crossings[index][1],"ko")


        # find segemnt which will hit
        # find lengths ti ll hit
        # subtract 1 each iteration
        # change direction, determine new point



    def _propogate(self):
        """Method which propogates the wave
        """

        # determine number of time steps
        self.n_steps = int(self.t_final / self.timestep)

        # initialize wave array
        wave = np.zeros((self.n_steps,self.size,2))

        # create initial droplet
        t = np.linspace(0.0,0.75 * np.pi,num=self.size)
        wave[0,:,0] = self.step / 100. * np.cos(t) + self.start_point[0]
        wave[0,:,1] = self.step / 100. * np.sin(t) + self.start_point[1]

        # define vector array
        # where col 0 = dx, col 1 = dy
        self.dir = np.zeros((self.size,2))
        self.dir[:,0] = np.cos(t)
        self.dir[:,1] = np.sin(t)

        # run through each point, determine intersecting segment and num hits
        for i in range(self.size):
            self._to_impact(wave[0,i],self.dir[i])

        # run through each time step and determine new wave location
        for i in range(1,self.n_steps):

            # determine new wave location
            wave[i] = wave[i-1] + self.step * self.dir


        # make wave global
        self.wave = wave


    def plot(self):
        """Method
        """

        # plot wave propogation
        plt.plot(self.shape[:,0],self.shape[:,1],"k")

        # plot each wave propogation
        for i in range(self.n_steps):
            plt.plot(self.wave[i,:,0],self.wave[i,:,1],"b")
        
        # show plot
        plt.axis("equal")
        plt.show()




waver = wave()
