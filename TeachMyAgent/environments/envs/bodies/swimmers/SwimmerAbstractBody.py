from TeachMyAgent.environments.envs.bodies.AbstractBody import AbstractBody
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum

class SwimmerAbstractBody(AbstractBody):
    '''
        Base class for swimmers.
    '''
    def __init__(self, scale, motors_torque, density, nb_steps_outside_water):
        '''
            Creates a swimmer, which cannot survive outside water.

            Args:
                scale: Scale value used in the environment (to adapt the embodiment to its environment)
                motors_torque: Maximum torque the embodiment can use on its motors
                density: Water density (in order to make the agent in a zero-gravity-like setup)
                nb_steps_outside_water: How many consecutive steps the embodiment can survive outside water
        '''
        super(SwimmerAbstractBody, self).__init__(scale, motors_torque)

        self.body_type = BodyTypesEnum.SWIMMER
        self.nb_steps_can_survive_outside_water = nb_steps_outside_water
        # set the embodiment's density to the same value as water so that it will be in a zero-gravity setup
        self.DENSITY = density - 0.01 # Make it a little lighter such that it slowly goes up when no action is done