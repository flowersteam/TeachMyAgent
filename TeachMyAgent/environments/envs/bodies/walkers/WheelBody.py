import numpy as np
import math
from Box2D.b2 import edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener

from TeachMyAgent.environments.envs.bodies.walkers.WalkerAbstractBody import WalkerAbstractBody
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomBodyUserData, CustomMotorUserData

MAIN_BODY_POLYGONS = [
    [(-10, +10), (+10, +10),
    (+10, -10), (-10, -10)]
]
HULL_BOTTOM_WIDTH = 20
SPEED_HIP     = 4
SPEED_KNEE    = 6

class WheelBody(WalkerAbstractBody):
    def __init__(self, scale, motors_torque=500, body_scale=1, nb_steps_under_water=600,
                 reset_on_hull_critical_contact=False):
        super(WheelBody, self).__init__(scale, motors_torque, nb_steps_under_water)
        self.LEG_W, self.LEG_H = 4 / self.SCALE, 10 / self.SCALE
        self.TORQUE_PENALTY = 0.00035 / 2 # 4 legs = 2 pair of legs
        self.reset_on_hull_critical_contact = reset_on_hull_critical_contact

        self.body_scale = body_scale

        self.AGENT_WIDTH = HULL_BOTTOM_WIDTH * self.body_scale / self.SCALE
        self.AGENT_HEIGHT = 20 * self.body_scale / self.SCALE + \
                            self.LEG_H * 4
        self.AGENT_CENTER_HEIGHT = 20 * self.body_scale / self.SCALE / 2 + \
                                   self.LEG_H * 2

        self.remove_reward_on_head_angle = True

    def draw(self, world, init_x, init_y, force_to_center):
        ''' Circular body
        MAIN_BODY_FIXTURES = fixtureDef(
            shape=circleShape(radius=0.4),
            density=1.0,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x000F
        )
        '''
        MAIN_BODY_FIXTURES = [
            fixtureDef(
                shape=polygonShape(vertices=[(x * self.body_scale / self.SCALE, y * self.body_scale / self.SCALE) for x, y in polygon]),
                density=5.0,
                friction=0.1,
                categoryBits=0x20,
                maskBits=0x000F)
            for polygon in MAIN_BODY_POLYGONS
        ]

        LEG_FD = fixtureDef(
            shape=polygonShape(box=(self.LEG_W / 2, self.LEG_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x000F)

        LOWER_FD = fixtureDef(
            shape=polygonShape(box=(0.8 * self.LEG_W / 2, self.LEG_H / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x000F)

        main_body = world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=MAIN_BODY_FIXTURES
        )

        basis_color1 = (0.6, 0.3, 0.5)
        basis_color2 = (0.4, 0.2, 0.3)
        main_body.color1 = tuple([c - 0.1 for c in basis_color1])
        main_body.color2 = tuple([c - 0.1 for c in basis_color2])
        leg_color1 = tuple([c + 0.1 for c in basis_color1])
        leg_color2 = tuple([c + 0.1 for c in basis_color2])

        main_body.ApplyForceToCenter((force_to_center, 0), True)

        main_body.userData = CustomBodyUserData(True, is_contact_critical=self.reset_on_hull_critical_contact, name="main_body")
        self.body_parts.append(main_body)
        self.reference_head_object = main_body

        for j in [[0, -1, 0], [0, 1, 0],[-1, 0, np.pi/2], [1, 0, np.pi/2]]:
            x_position = j[0] * (HULL_BOTTOM_WIDTH * self.body_scale / self.SCALE / 2)
            y_position = j[1] * (HULL_BOTTOM_WIDTH * self.body_scale / self.SCALE / 2)
            # for i in [-1, +1]:
            leg_x = init_x + j[0] * (self.LEG_H / 2) + x_position
            leg_y = init_y + j[1] * (self.LEG_H / 2) + y_position
            leg = world.CreateDynamicBody(
                position=(leg_x, leg_y),
                angle=(j[2]),
                fixtures=LEG_FD
            )
            leg.color1 = leg_color1
            leg.color2 = leg_color2
            rjd = revoluteJointDef(
                bodyA=main_body,
                bodyB=leg,
                anchor=(init_x + x_position, init_y + y_position),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )

            leg.userData = CustomBodyUserData(False, name="leg")
            self.body_parts.append(leg)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(SPEED_HIP, False)
            self.motors.append(joint_motor)

            lower = world.CreateDynamicBody(
                position=(leg_x + j[0] * (self.LEG_H),
                          leg_y + j[1] * (self.LEG_H)),
                angle=(j[2]),
                fixtures=LOWER_FD
            )
            lower.color1 = leg_color1
            lower.color2 = leg_color2
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                anchor=(leg_x + j[0] * (self.LEG_H / 2),
                        leg_y + j[1] * (self.LEG_H / 2)),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )

            lower.userData = CustomBodyUserData(True, name="lower")
            self.body_parts.append(lower)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(
                SPEED_KNEE,
                True,
                contact_body=lower,
                angle_correction=1.0)
            self.motors.append(joint_motor)