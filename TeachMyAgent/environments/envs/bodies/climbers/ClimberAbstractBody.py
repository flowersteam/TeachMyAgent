from TeachMyAgent.environments.envs.bodies.walkers.WalkerAbstractBody import WalkerAbstractBody
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum
import Box2D
from Box2D.b2 import circleShape, fixtureDef

class ClimberAbstractBody(WalkerAbstractBody):
    '''
        Base class for climbers.
    '''
    def __init__(self, scale, motors_torque, nb_steps_under_water):
        '''
            Creates a climber, which cannot survive under water and cannot touch ground.

            Args:
                scale: Scale value used in the environment (to adapt the embodiment to its environment)
                motors_torque: Maximum torque the embodiment can use on its motors
                nb_steps_under_water: How many consecutive steps the embodiment can survive under water
        '''
        super(ClimberAbstractBody, self).__init__(scale, motors_torque, nb_steps_under_water)

        self.body_type = BodyTypesEnum.CLIMBER
        self.sensors = []
        self.SENSOR_FD = fixtureDef(
            shape=circleShape(radius=0.05),
            density=1.0,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x1,
            isSensor=True
        )

    # States
    def get_state_size(self):
        '''
            Returns the size of the embodiment's state vector (classic state + sensors)
        '''
        return super(ClimberAbstractBody, self).get_state_size() + len(self.get_sensors_state())

    def get_sensors_state(self):
        '''
            Returns state vector sensors.

            For each sensor returns:
            - if it a collision is detected
            - if it is already grasping (i.e. it is attached to a joint)
        '''
        state = []
        for sensor in self.sensors:
            state.extend([1.0 if sensor.userData.has_contact else 0.0,
                          1.0 if sensor.userData.has_joint else 0.0])
        return state

    # Actions
    def get_action_size(self):
        '''
            Returns the size of the action space (classic action space + number of sensors).
        '''
        return super(ClimberAbstractBody, self).get_action_size() + len(self.sensors)

    # Draw
    def get_elements_to_render(self):
        '''
            Returns bodies that must be rendered in the `env.render` function (including sensors).
        '''
        return super(ClimberAbstractBody, self).get_elements_to_render() + self.sensors

    # Destroy
    def destroy(self, world):
        super(ClimberAbstractBody, self).destroy(world)  # Destroy the rest of the body as any other agent
        for sensor in self.sensors:
            world.DestroyBody(sensor) # Destroy sensor
        self.sensors = []




