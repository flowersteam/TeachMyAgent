import numpy as np
from Box2D.b2 import edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener

from TeachMyAgent.environments.envs.bodies.walkers.WalkerAbstractBody import WalkerAbstractBody
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomBodyUserData, CustomMotorUserData

HULL_POLYGONS = [
    [(-30, +9), (+6, +9), (+34, +1),
    (+34, -8), (-30, -8)]
]
HULL_BOTTOM_WIDTH = 64
SPEED_HIP     = 4
SPEED_KNEE    = 6

class ClassicBipedalBody(WalkerAbstractBody):
    def __init__(self, scale, motors_torque=80, nb_steps_under_water=600, reset_on_hull_critical_contact=False):
        super(ClassicBipedalBody, self).__init__(scale, motors_torque, nb_steps_under_water)
        self.LEG_DOWN = 3 / self.SCALE # 0 = center of hull
        self.LEG_W, self.LEG_H = 8 / self.SCALE, 34 / self.SCALE
        self.TORQUE_PENALTY = 0.00035
        self.reset_on_hull_critical_contact = reset_on_hull_critical_contact

        self.AGENT_WIDTH = HULL_BOTTOM_WIDTH / self.SCALE
        self.AGENT_HEIGHT = 17 / self.SCALE + \
                            self.LEG_H * 2 - self.LEG_DOWN
        self.AGENT_CENTER_HEIGHT = self.LEG_H * 2 + self.LEG_DOWN

    def draw(self, world, init_x, init_y, force_to_center):
        HULL_FIXTURES = [
            fixtureDef(
                shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in polygon]),
                density=5.0,
                friction=0.1,
                categoryBits=0x20,
                maskBits=0x000F)  # 0.99 bouncy
            for polygon in HULL_POLYGONS
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

        hull = world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=HULL_FIXTURES
        )
        hull.color1 = (0.5, 0.4, 0.9)
        hull.color2 = (0.3, 0.3, 0.5)
        hull.ApplyForceToCenter((force_to_center, 0), True)

        hull.userData = CustomBodyUserData(True, is_contact_critical=self.reset_on_hull_critical_contact, name="hull")
        self.body_parts.append(hull)
        self.reference_head_object = hull

        for i in [-1, +1]:
            leg = world.CreateDynamicBody(
                position=(init_x, init_y - self.LEG_H / 2 - self.LEG_DOWN),
                #angle=(i * 0.05),#2°
                fixtures=LEG_FD
            )
            leg.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            leg.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=hull,
                bodyB=leg,
                anchor=(init_x, init_y - self.LEG_DOWN),
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
                position=(init_x, init_y - self.LEG_H * 3 / 2 - self.LEG_DOWN),
                #angle=(i * 0.05), #2°
                fixtures=LOWER_FD
            )
            lower.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            lower.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                anchor=(init_x, init_y - self.LEG_DOWN - self.LEG_H),
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