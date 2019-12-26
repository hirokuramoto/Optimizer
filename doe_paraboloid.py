from openmdao.api import IndepVarComp, Component, Problem, Group, SqliteRecorder
from paraboloid import ParaboloidExternalCode
from openmdao.drivers.latinhypercube_driver import OptimizedLatinHypercubeDriver
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

class DoeParaboloid(object):
    """docstring for DoeParaboloid."""
    def __init__(self, filepath):
        self._filepath = filepath

    def main(self):
        # データの読み込み
        data = pd.read_csv(filepath, header = 0)



        top = Problem(impl=impl)
        root = top.root = Group()

        for i in

        root.add('p1', IndepVarComp('a', 23.0), promotes=['a'])
        root.add('p2', IndepVarComp('b', 30.0), promotes=['b'])
        root.add('p3', IndepVarComp('c', 208.0), promotes=['c'])
        root.add('p4', IndepVarComp('d', 10.0), promotes=['d'])
        root.add('p5', IndepVarComp('r', 3.0), promotes=['r'])
        root.add('p', ParaboloidExternalCode(),promotes=['a', 'b', 'c', 'd', 'r', 'f_xy1', 'f_xy2', 'f_xy3'])


top.driver = OptimizedLatinHypercubeDriver(num_samples=5, seed=0,
        population=20, generations=4, norm_method=2, num_par_doe=1)
#top.driver = pyOptSparseDriver()
#top.driver.options['optimizer'] = 'NSGA2'
top.driver.add_desvar('a', lower=20.0, upper=40.0)
top.driver.add_desvar('b', lower=10.0, upper=50.0)
top.driver.add_desvar('c', lower=120.0, upper=220.0)
top.driver.add_desvar('d', lower=10.0, upper=50.0)
top.driver.add_desvar('r', lower=2.0, upper=4.0)
top.driver.add_objective('f_xy1')
top.driver.add_objective('f_xy2')
top.driver.add_objective('f_xy3')

recorder = SqliteRecorder('doe_paraboloid_seminer')
recorder.options['record_params'] = True
recorder.options['record_unknowns'] = True
recorder.options['record_resids'] = False
top.driver.add_recorder(recorder)

top.setup()
top.run()
top.cleanup()
