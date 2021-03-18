import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomUserDataObjectTypes

class ClimbingDynamics(object):
    def before_step_climbing_dynamics(self, actions, body, world):
        for i in range(len(body.sensors)):
            action_to_check = actions[len(actions) - i - 1]
            sensor_to_check = body.sensors[len(body.sensors) - i - 1]
            if action_to_check > 0: # Check whether the sensor should grasp or release
                sensor_to_check.userData.ready_to_attach = True
            else:
                sensor_to_check.userData.ready_to_attach = False
                if sensor_to_check.userData.has_joint: # If release and it had a joint => destroy it
                    sensor_to_check.userData.has_joint = False
                    joint_to_destroy = next((_joint.joint for _joint in sensor_to_check.joints
                            if isinstance(_joint.joint, Box2D.b2RevoluteJoint)), None)
                    if joint_to_destroy is not None:
                        world.DestroyJoint(joint_to_destroy)

    def after_step_climbing_dynamics(self, contact_detector, world):
        # Add climbing joints if needed
        for sensor in contact_detector.contact_dictionaries:
            if len(contact_detector.contact_dictionaries[sensor]) > 0 and \
                    sensor.userData.ready_to_attach and not sensor.userData.has_joint:
                other_body = contact_detector.contact_dictionaries[sensor][0]

                # Check if still overlapping after solver
                # Super coarse yet fast way, mainly useful for creepers
                other_body_shape = other_body.fixtures[0].shape
                x_values = [v[0] for v in other_body_shape.vertices]
                y_values = [v[1] for v in other_body_shape.vertices]
                radius = sensor.fixtures[0].shape.radius + 0.01

                if sensor.worldCenter[0] + radius > min(x_values) and sensor.worldCenter[0] - radius < max(x_values) and \
                    sensor.worldCenter[1] + radius > min(y_values) and sensor.worldCenter[1] - radius < max(y_values):
                    rjd = revoluteJointDef(
                        bodyA=sensor,
                        bodyB=other_body,
                        anchor=sensor.worldCenter  # contact.worldManifold.points[0],
                    )

                    joint = world.CreateJoint(rjd)
                    joint.bodyA.userData.joint = joint
                    sensor.userData.has_joint = True
                else:
                    contact_detector.contact_dictionaries[sensor].remove(other_body)
                    if len(contact_detector.contact_dictionaries[sensor]) == 0:
                        sensor.userData.has_contact = False

class ClimbingContactDetector(contactListener):
    '''
    Store contacts between sensors and graspable surfaces in a dictionaries associated to the sensor.
    '''
    def __init__(self):
        super(ClimbingContactDetector, self).__init__()
        self.contact_dictionaries = {}
    def BeginContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        for idx, body in enumerate(bodies):
            if body.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR and body.userData.check_contact:
                other_body = bodies[(idx + 1) % 2]
                if other_body.userData.object_type == CustomUserDataObjectTypes.GRIP_TERRAIN or \
                                    other_body.userData.object_type == CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN:
                    body.userData.has_contact = True
                    if body in self.contact_dictionaries:
                        self.contact_dictionaries[body].append(other_body)
                    else:
                        self.contact_dictionaries[body] = [other_body]
                else:
                    return

    def EndContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        for idx, body in enumerate(bodies):
            other_body = bodies[(idx + 1) % 2]
            if body.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR and \
                    body.userData.check_contact and body.userData.has_contact:
                if other_body in self.contact_dictionaries[body]:
                    self.contact_dictionaries[body].remove(other_body)

                if len(self.contact_dictionaries[body]) == 0:
                    body.userData.has_contact = False

    def Reset(self):
        self.contact_dictionaries = {}