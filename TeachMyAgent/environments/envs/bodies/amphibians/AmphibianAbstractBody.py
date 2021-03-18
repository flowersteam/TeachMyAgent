from TeachMyAgent.environments.envs.bodies.AbstractBody import AbstractBody
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum

class AmphibianAbstractBody(AbstractBody):
    def __init__(self, scale, motors_torque, density):
        super(AmphibianAbstractBody, self).__init__(scale, motors_torque)

        self.body_type = BodyTypesEnum.AMPHIBIAN
        self.DENSITY = density # set the embodiment's density to the same value as water so that it will be in a zero-gravity setup