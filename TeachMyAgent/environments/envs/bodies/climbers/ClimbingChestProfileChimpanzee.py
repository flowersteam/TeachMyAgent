import numpy as np
from Box2D.b2 import edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, weldJointDef

from TeachMyAgent.environments.envs.bodies.climbers.ClimberAbstractBody import ClimberAbstractBody
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomBodyUserData, CustomMotorUserData, CustomBodySensorUserData

SPEED_HIP     = 4
SPEED_KNEE    = 6
SPEED_HAND    = 8

class ClimbingChestProfileChimpanzee(ClimberAbstractBody):
    def __init__(self, scale, motors_torque = 100, nb_steps_under_water = 600):
        super(ClimbingChestProfileChimpanzee, self).__init__(scale, motors_torque, nb_steps_under_water)
        self.LEG_DOWN = 12 / self.SCALE
        self.ARM_UP = 22 / self.SCALE
        self.LIMB_W, self.LIMB_H = 8 / self.SCALE, 28 / self.SCALE
        self.HAND_PART_W, self.HAND_PART_H = 4 / self.SCALE, 8 / self.SCALE
        self.LEG_H = self.LIMB_H
        self.TORQUE_PENALTY = 0.00035 / 5 # Legs + arms + hands
        self.BODY_HEIGHT = 45
        self.HEAD_HEIGHT = 20

        self.AGENT_WIDTH = 24 / self.SCALE
        self.AGENT_HEIGHT = self.BODY_HEIGHT / self.SCALE + \
                            self.HEAD_HEIGHT / self.SCALE + 0.2 + \
                            self.LEG_H * 2 - self.LEG_DOWN
        self.AGENT_CENTER_HEIGHT = self.LEG_H * 2 + self.LEG_DOWN

        self.remove_reward_on_head_angle = True

    def draw(self, world, init_x, init_y, force_to_center):
        head = world.CreateDynamicBody(
            position=(init_x, init_y + self.BODY_HEIGHT / self.SCALE / 2 + self.HEAD_HEIGHT / self.SCALE / 2 + 0.2),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in [
                    (-5, +10), (+5, +10),
                    (+5, -10), (-5, -10)]]),
                density=5.0,
                friction=0.1,
                categoryBits=0x20,
                maskBits=0x1
            )
        )
        head.color1 = (0.5, 0.4, 0.9)
        head.color2 = (0.3, 0.3, 0.5)
        head.ApplyForceToCenter((force_to_center, 0), True)

        head.userData = CustomBodyUserData(True, is_contact_critical=True, name="head")
        self.body_parts.append(head)
        self.reference_head_object = head

        body = world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in [
                    (-12, +25), (+12, +25),
                    (+8, -20), (-8, -20)]]),
                density=5.0,
                friction=0.1,
                categoryBits=0x20,
                maskBits=0x1  # collide only with ground
            )
        )
        body.color1 = (0.5, 0.4, 0.9)
        body.color2 = (0.3, 0.3, 0.5)

        body.userData = CustomBodyUserData(True, is_contact_critical=True, name="body")
        self.body_parts.append(body)
        
        rjd = revoluteJointDef(
            bodyA=head,
            bodyB=body,
            anchor=(init_x, init_y + self.BODY_HEIGHT / self.SCALE / 2),
            enableMotor=False,
            enableLimit=True,
            lowerAngle=-0.1 * np.pi,
            upperAngle=0.1 * np.pi,
        )

        world.CreateJoint(rjd)

        UPPER_LIMB_FD = fixtureDef(
            shape=polygonShape(box=(self.LIMB_W / 2, self.LIMB_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x1
        )

        LOWER_LIMB_FD = fixtureDef(
            shape=polygonShape(box=(0.8 * self.LIMB_W / 2, self.LIMB_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x1
        )

        HAND_PART_FD = fixtureDef(
            shape=polygonShape(box=(self.HAND_PART_W / 2, self.HAND_PART_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x1
        )

        # ARMS
        for j in [-1, -1]:
            upper = world.CreateDynamicBody(
                position=(init_x, init_y + self.LIMB_H / 2 + self.ARM_UP),
                # angle=(i * 0.05),
                fixtures=UPPER_LIMB_FD
            )
            upper.color1 = (0.6 - j / 10., 0.3 - j / 10., 0.5 - j / 10.)
            upper.color2 = (0.4 - j / 10., 0.2 - j / 10., 0.3 - j / 10.)
            rjd = revoluteJointDef(
                bodyA=body,
                bodyB=upper,
                anchor=(init_x, init_y + self.ARM_UP),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-0.75 * 2 * np.pi,
                upperAngle=0,
            )

            upper.userData = CustomBodyUserData(False, name="upper_arm")
            self.body_parts.append(upper)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(SPEED_HIP, False)
            self.motors.append(joint_motor)

            lower = world.CreateDynamicBody(
                position=(init_x, init_y + self.LIMB_H * 3 / 2 + self.ARM_UP),
                # angle=(i * 0.05),
                fixtures=LOWER_LIMB_FD
            )
            lower.color1 = (0.6 - j / 10., 0.3 - j / 10., 0.5 - j / 10.)
            lower.color2 = (0.4 - j / 10., 0.2 - j / 10., 0.3 - j / 10.)
            rjd = revoluteJointDef(
                bodyA=upper,
                bodyB=lower,
                anchor=(init_x, init_y + self.LIMB_H + self.ARM_UP),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=0,
                upperAngle=0.75 * np.pi,
            )

            lower.userData = CustomBodyUserData(False, name="lower_arm")
            self.body_parts.append(lower)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(SPEED_HIP, False)
            self.motors.append(joint_motor)

            # hand
            prev_part = lower
            initial_y = init_y + self.LIMB_H * 2 + self.ARM_UP
            angle_boundaries = [[-0.5, 0.5]]
            nb_hand_parts = 1
            for u in range(nb_hand_parts):
                hand_part = world.CreateDynamicBody(
                    position=(init_x, initial_y + self.HAND_PART_H / 2 + self.HAND_PART_H * u),
                    fixtures=HAND_PART_FD
                )

                hand_part.color1 = (0.6 - j / 10., 0.3 - j / 10., 0.5 - j / 10.)
                hand_part.color2 = (0.4 - j / 10., 0.2 - j / 10., 0.3 - j / 10.)
                rjd = revoluteJointDef(
                    bodyA=prev_part,
                    bodyB=hand_part,
                    anchor=(init_x, initial_y + self.HAND_PART_H * u),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=self.MOTORS_TORQUE,
                    motorSpeed=1,
                    lowerAngle=angle_boundaries[u][0] * np.pi,
                    upperAngle=angle_boundaries[u][1] * np.pi,
                )

                hand_part.userData = CustomBodyUserData(True, name="hand")
                self.body_parts.append(hand_part)

                joint_motor = world.CreateJoint(rjd)
                joint_motor.userData = CustomMotorUserData(SPEED_HAND,
                                                           True,
                                                           contact_body=hand_part)
                self.motors.append(joint_motor)

                prev_part = hand_part

            hand_sensor_position = (init_x, initial_y + self.HAND_PART_H * nb_hand_parts)
            hand_sensor_part = world.CreateDynamicBody(
                position=hand_sensor_position,
                fixtures=self.SENSOR_FD,
                userData = CustomBodySensorUserData(True, False, "hand_sensor")
            )
            hand_sensor_part.color1 = (1, 0, 0)#(0.6 - j / 10., 0.3 - j / 10., 0.5 - j / 10.)
            hand_sensor_part.color2 = (1, 0, 0)#(0.4 - j / 10., 0.2 - j / 10., 0.3 - j / 10.)

            self.sensors.append(hand_sensor_part)
            world.CreateJoint(weldJointDef(
                bodyA = prev_part,
                bodyB = hand_sensor_part,
                anchor = hand_sensor_position
            ))

