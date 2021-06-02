from TeachMyAgent.environments.envs.bodies.AbstractBody import AbstractBody
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum

class AmphibianAbstractBody(AbstractBody):
    '''
        Base class for amphibians.
    '''
    def __init__(self, scale, motors_torque, density):
        '''
            Creates an amphibious embodiment allowed to go both under and outside water

            :param scale: Scale value used in the environment (to adapt the embodiment to its environment)
            :param motors_torque: Maximum torque the embodiment can use on its motors
            :param density: Water density (in order to make the agent in a zero-gravity-like setup)
        '''
        super(AmphibianAbstractBody, self).__init__(scale, motors_torque)

        self.body_type = BodyTypesEnum.AMPHIBIAN
        self.DENSITY = density # set the embodiment's density to the same value as water so that it will be in a zero-gravity setup