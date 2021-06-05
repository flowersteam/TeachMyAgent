from TeachMyAgent.environments.envs.bodies.AbstractBody import AbstractBody
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum

class WalkerAbstractBody(AbstractBody):
    '''
        Base class for walkers.
    '''
    def __init__(self, scale, motors_torque, nb_steps_under_water):
        '''
            Creates a walker, which cannot survive under water.

            Args:
                scale: Scale value used in the environment (to adapt the embodiment to its environment)
                motors_torque: Maximum torque the embodiment can use on its motors
                nb_steps_under_water: How many consecutive steps the embodiment can survive under water
        '''
        super(WalkerAbstractBody, self).__init__(scale, motors_torque)

        self.body_type = BodyTypesEnum.WALKER
        self.nb_steps_can_survive_under_water = nb_steps_under_water