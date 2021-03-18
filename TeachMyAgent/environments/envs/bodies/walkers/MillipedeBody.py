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

class MillipedeBody(WalkerAbstractBody):
    def __init__(self, scale, motors_torque=200, nb_of_bodies=4, nb_steps_under_water=600,
                 reset_on_hull_critical_contact=False):
        super(MillipedeBody, self).__init__(scale, motors_torque, nb_steps_under_water)
        self.LEG_DOWN = 3 / self.SCALE  # 0 = center of hull
        self.LEG_W, self.LEG_H = 4 / self.SCALE, 10 / self.SCALE
        self.TORQUE_PENALTY = 0.00035
        self.reset_on_hull_critical_contact = reset_on_hull_critical_contact

        self.nb_of_bodies = nb_of_bodies
        self.TORQUE_PENALTY = 0.00035 / self.nb_of_bodies # 1 body = 1 pair of legs

        self.AGENT_WIDTH = MAIN_BODY_BOTTOM_WIDTH / self.SCALE * self.nb_of_bodies
        self.AGENT_HEIGHT = 20 / self.SCALE + \
                            self.LEG_H * 2 - self.LEG_DOWN
        self.AGENT_CENTER_HEIGHT = self.LEG_H * 2 + self.LEG_DOWN

    def draw(self, world, init_x, init_y, force_to_center):
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

        init_x = init_x - MAIN_BODY_BOTTOM_WIDTH / self.SCALE * self.nb_of_bodies / 2 # initial init_x is the middle of the whole body
        previous_main_body = None
        for j in range(self.nb_of_bodies):
            main_body_x = init_x + j * (MAIN_BODY_BOTTOM_WIDTH / self.SCALE)
            main_body = world.CreateDynamicBody(
                position=(main_body_x, init_y),
                fixtures=MAIN_BODY_FIXTURES
            )
            main_body.color1 = (0.5, 0.4, 0.9)
            main_body.color2 = (0.3, 0.3, 0.5)
            main_body.ApplyForceToCenter((force_to_center, 0), True)

            # self.body_parts.append(BodyPart(main_body, True, True))
            main_body.userData = CustomBodyUserData(True, is_contact_critical=False, name="body")
            self.body_parts.append(main_body)

            for i in [-1, +1]:
                leg = world.CreateDynamicBody(
                    position=(main_body_x, init_y - self.LEG_H / 2 - self.LEG_DOWN),
                    #angle=(i * 0.05),
                    fixtures=LEG_FD
                )
                leg.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
                leg.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
                rjd = revoluteJointDef(
                    bodyA=main_body,
                    bodyB=leg,
                    anchor=(main_body_x, init_y - self.LEG_DOWN),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=self.MOTORS_TORQUE,
                    motorSpeed=i,
                    lowerAngle=-0.8,
                    upperAngle=1.1,
                )

                leg.userData = CustomBodyUserData(False, name="leg")
                self.body_parts.append(leg)

                joint_motor = world.CreateJoint(rjd)
                joint_motor.userData = CustomMotorUserData(SPEED_HIP, False)
                self.motors.append(joint_motor)

                lower = world.CreateDynamicBody(
                    position=(main_body_x, init_y - self.LEG_H * 3 / 2 - self.LEG_DOWN),
                    #angle=(i * 0.05),
                    fixtures=LOWER_FD
                )
                lower.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
                lower.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
                rjd = revoluteJointDef(
                    bodyA=leg,
                    bodyB=lower,
                    anchor=(main_body_x, init_y - self.LEG_DOWN - self.LEG_H),
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

            if previous_main_body is not None:
                rjd = revoluteJointDef(
                    bodyA=previous_main_body,
                    bodyB=main_body,
                    anchor=(main_body_x - MAIN_BODY_BOTTOM_WIDTH / self.SCALE / 2, init_y),
                    enableMotor=False,
                    enableLimit=True,
                    lowerAngle=-0.05 * np.pi,
                    upperAngle=0.05 * np.pi,
                )

                world.CreateJoint(rjd)
            previous_main_body = main_body

        self.reference_head_object = previous_main_body  # set as head the rightmost body
        self.reference_head_object.is_contact_critical = self.reference_head_object
