from TeachMyAgent.environments.envs.bodies.walkers.WalkerAbstractBody import WalkerAbstractBody
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum
import Box2D
from Box2D.b2 import circleShape, fixtureDef

class ClimberAbstractBody(WalkerAbstractBody):
    def __init__(self, scale, motors_torque, nb_steps_under_water):
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
        return super(ClimberAbstractBody, self).get_state_size() + len(self.get_sensors_state())

    def get_sensors_state(self):
        state = []
        for sensor in self.sensors:
            state.extend([1.0 if sensor.userData.has_contact else 0.0,
                          1.0 if sensor.userData.has_joint else 0.0])
        return state

    # Actions
    def get_action_size(self):
        return super(ClimberAbstractBody, self).get_action_size() + len(self.sensors)

    # Draw
    def get_elements_to_render(self):
        return super(ClimberAbstractBody, self).get_elements_to_render() + self.sensors

    # Destroy
    def destroy(self, world):
        super(ClimberAbstractBody, self).destroy(world)  # Destroy the rest of the body as any other agent
        for sensor in self.sensors:
            world.DestroyBody(sensor) # Destroy sensor
        self.sensors = []




