

class BatchScript:
    """
    Class to create .sh batch script to submit jobs to clusters
    Attributes:
    --------------------
    --------------------

    _nodes: int
        number of nodes to use
    _cpus: int
        number of cpus per node to use
    _mem: int
        minimum memory for job in GB
    _jobarray: shape
        parameters for jobarray
    _jobname: string
        name of job to use as id
    _time: int
        timelimit for job in h
    """
    _nodes = 1
    _cpus = 4
    _mem = 16
    _jobarray = None
    _jobname = "diag"
    _time = 24

    def __init__(self, nodes = 1, cpus = 4, mem = 16, jobarray = None, jobname = "diag", time = 24):
        self._nodes = nodes
        self._cpus = cpus
        self._mem = mem
        self._jobarray = jobarray
        self._jobname = jobname
        self._time = time


# ------------------------------------ ATTRIBUTES
    @property
    def jobarray(self):
        return self._jobarray

# ------------------------------------ SETTERS
    @_nodes.setter
    def nodes(self, node_num):
        self._nodes = node_num


# ------------------------------------ METHODS

    def generate_script(self):
        with open ('sender.sh', 'w') as file:
            file.write(f'''\
                        #! /bin/bash
                        #SBATCH --job-name="${self._jobname}_L=12"
                        echo "more lines"
                        ''')