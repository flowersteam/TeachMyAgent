import numpy as np
from Box2D.b2 import edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener

from TeachMyAgent.environments.envs.bodies.walkers.WalkerAbstractBody import WalkerAbstractBody
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomBodyUserData, CustomMotorUserData

MAIN_BODY_POLYGONS = [
    [(-10, +10), (+10, +10),
    (+10, -10), (-10, -10)]
]
MAIN_BODY_BOTTOM_WIDTH = 20
SPEED_HIP     = 4
SPEED_KNEE    = 6

class SpiderBody(WalkerAbstractBody):
    def __init__(self, scale, motors_torque=100, nb_pairs_of_legs=2, nb_steps_under_water=600,
                 reset_on_hull_critical_contact=False):
        super(SpiderBody, self).__init__(scale, motors_torque, nb_steps_under_water)
        self.LEG_DOWN = 4 / self.SCALE
        self.LEG_W, self.LEG_H = 6 / self.SCALE, 20 / self.SCALE
        self.reset_on_hull_critical_contact = reset_on_hull_critical_contact

        self.nb_pairs_of_legs = nb_pairs_of_legs

        self.TORQUE_PENALTY = 0.00035 / self.nb_pairs_of_legs

        # not exact but works
        self.AGENT_WIDTH = MAIN_BODY_BOTTOM_WIDTH / self.SCALE + \
                           self.LEG_H * 4
        self.AGENT_HEIGHT = 20 / self.SCALE + \
                            self.LEG_H * 2
        self.AGENT_CENTER_HEIGHT = self.LEG_H + self.LEG_DOWN

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
                shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in polygon]),
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


        for i in [+1, -1] * self.nb_pairs_of_legs:
            ##### First part of the leg #####
            upper_leg_angle = 0.25 * np.pi * i
            upper_leg_x_distance = np.sin(upper_leg_angle) * self.LEG_H / 2
            upper_leg_y_distance = np.cos(upper_leg_angle) * self.LEG_H / 2
            upper_leg_x = init_x - i * MAIN_BODY_BOTTOM_WIDTH / self.SCALE / 2 - upper_leg_x_distance
            upper_leg_y = init_y + upper_leg_y_distance - self.LEG_DOWN

            upper_leg = world.CreateDynamicBody(
                position=(upper_leg_x, upper_leg_y),
                angle=upper_leg_angle,
                fixtures=LEG_FD
            )
            upper_leg.color1 = leg_color1
            upper_leg.color2 = leg_color2

            rjd = revoluteJointDef(
                bodyA=main_body,
                bodyB=upper_leg,
                anchor=(init_x - i * MAIN_BODY_BOTTOM_WIDTH / self.SCALE / 2,
                        init_y - self.LEG_DOWN),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-0.1*np.pi,
                upperAngle=0.1*np.pi,
            )

            upper_leg.userData = CustomBodyUserData(False, name="upper_leg")
            self.body_parts.append(upper_leg)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(SPEED_HIP, False)
            self.motors.append(joint_motor)

            ##### Second part of the leg #####
            middle_leg_angle = 0.7 * np.pi * i
            middle_leg_x_distance = np.sin(middle_leg_angle) * self.LEG_H / 2
            middle_leg_y_distance = -np.cos(middle_leg_angle) * self.LEG_H / 2
            middle_leg_x = upper_leg_x - upper_leg_x_distance - middle_leg_x_distance
            middle_leg_y = upper_leg_y + upper_leg_y_distance - middle_leg_y_distance
            middle_leg = world.CreateDynamicBody(
                position=(middle_leg_x, middle_leg_y),
                angle=middle_leg_angle,
                fixtures=LEG_FD
            )
            middle_leg.color1 = leg_color1
            middle_leg.color2 = leg_color2

            rjd = revoluteJointDef(
                bodyA=upper_leg,
                bodyB=middle_leg,
                anchor=(upper_leg_x - upper_leg_x_distance,
                        upper_leg_y + upper_leg_y_distance),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-0.15*np.pi,
                upperAngle=0.15*np.pi,
            )

            middle_leg.userData = CustomBodyUserData(False, name="middle_leg")
            self.body_parts.append(middle_leg)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(SPEED_HIP, False)
            self.motors.append(joint_motor)

            ##### Third part of the leg #####
            lower_leg_angle = 0.9 * np.pi * i
            lower_leg_x_distance = np.sin(lower_leg_angle) * self.LEG_H / 2
            lower_leg_y_distance = -np.cos(lower_leg_angle) * self.LEG_H / 2
            lower_leg_x = middle_leg_x - middle_leg_x_distance - lower_leg_x_distance
            lower_leg_y = middle_leg_y - middle_leg_y_distance - lower_leg_y_distance
            lower_leg = world.CreateDynamicBody(
                position=(lower_leg_x, lower_leg_y),
                angle=lower_leg_angle,
                fixtures=LOWER_FD
            )
            lower_leg.color1 = leg_color1
            lower_leg.color2 = leg_color2

            rjd = revoluteJointDef(
                bodyA=middle_leg,
                bodyB=lower_leg,
                anchor=(middle_leg_x - middle_leg_x_distance,
                        middle_leg_y - middle_leg_y_distance ),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-0.2*np.pi,
                upperAngle=0.2*np.pi,
            )

            lower_leg.userData = CustomBodyUserData(True, name="lower_leg")
            self.body_parts.append(lower_leg)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(SPEED_KNEE,
                                                       True,
                                                       contact_body=lower_leg)
            self.motors.append(joint_motor)
