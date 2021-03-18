import numpy as np
from Box2D.b2 import edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener

from TeachMyAgent.environments.envs.bodies.swimmers.SwimmerAbstractBody import SwimmerAbstractBody
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomBodyUserData, CustomMotorUserData

# Head
HULL_POLYGONS = [
    (-20, +12), (+6, +12),
     (+15, +4), (+15, -4),
     (+6, -12), (-20, -12)
]

BODY_P1 = [
    (-8, +9), (+8, +12),
     (+8, -12), (-8, -9)
]

BODY_P2 = [
    (-8, +4), (+8, +9),
     (+8, -9), (-8, -4)
]

# Tail
BODY_P3 = [
    (-4, +2), (+4, +4),
     (+4, -4), (-4, -2)
]

FIN = [
    (-1, -10), (-1, +10),
     (+1, +10), (+1, -10)
]

HULL_BOTTOM_WIDTH = 35
SPEED_HIP     = 4
SPEED_KNEE    = 6

class FishBody(SwimmerAbstractBody):
    def __init__(self, scale, density, motors_torque = 80, nb_steps_outside_water = 600):
        super(FishBody, self).__init__(scale, motors_torque, density, nb_steps_outside_water)
        self.TORQUE_PENALTY = 0.00035

        self.AGENT_WIDTH = HULL_BOTTOM_WIDTH / self.SCALE
        self.AGENT_HEIGHT = 18 /self.SCALE
        self.AGENT_CENTER_HEIGHT = 9 / self.SCALE

        self.remove_reward_on_head_angle = True

        self.fins = []
        self.tail = None

    def draw(self, world, init_x, init_y, force_to_center):
        init_y = init_y + 1
        #### HULL ####
        HULL_FD = fixtureDef(
            shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in HULL_POLYGONS]),
            density=self.DENSITY,
            friction=0.1,
            categoryBits=0x20,
            maskBits=0x000F)  # 0.99 bouncy

        hull = world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=HULL_FD
        )
        hull.color1 = (0.5, 0.4, 0.9)
        hull.color2 = (0.3, 0.3, 0.5)
        # hull.ApplyForceToCenter((force_to_center, 0), True)

        hull.userData = CustomBodyUserData(True, is_contact_critical=False, name="head")
        self.body_parts.append(hull)
        self.reference_head_object = hull

        #### P1 ####
        body_p1_x = init_x - 35 / 2 / self.SCALE - 16 / 2 / self.SCALE
        BODY_P1_FD = fixtureDef(
            shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in BODY_P1]),
            density=self.DENSITY,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x000F)

        body_p1 = world.CreateDynamicBody(
            position=(body_p1_x, init_y),
            fixtures=BODY_P1_FD
        )
        body_p1.color1 = (0.5, 0.4, 0.9)
        body_p1.color2 = (0.3, 0.3, 0.5)

        rjd = revoluteJointDef(
            bodyA=hull,
            bodyB=body_p1,
            anchor=(init_x - 35 / 2 / self.SCALE, init_y),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=self.MOTORS_TORQUE,
            motorSpeed=1,
            lowerAngle=-0.1 * np.pi,
            upperAngle=0.2 * np.pi,
        )

        body_p1.userData = CustomBodyUserData(True, name="body")
        self.body_parts.append(body_p1)

        joint_motor = world.CreateJoint(rjd)
        joint_motor.userData = CustomMotorUserData(SPEED_KNEE, True, contact_body=body_p1)
        self.motors.append(joint_motor)

        #### P2 ####
        body_p2_x = body_p1_x - 16 / 2 / self.SCALE - 16 / 2 / self.SCALE
        BODY_P2_FD = fixtureDef(
            shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in BODY_P2]),
            density=self.DENSITY,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x000F)

        body_p2 = world.CreateDynamicBody(
            position=(body_p2_x, init_y),
            fixtures=BODY_P2_FD
        )
        body_p2.color1 = (0.5, 0.4, 0.9)
        body_p2.color2 = (0.3, 0.3, 0.5)

        rjd = revoluteJointDef(
            bodyA=body_p1,
            bodyB=body_p2,
            anchor=(body_p1_x - 16 / 2 / self.SCALE, init_y),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=self.MOTORS_TORQUE,
            motorSpeed=1,
            lowerAngle=-0.15 * np.pi,
            upperAngle=0.15 * np.pi,
        )

        body_p2.userData = CustomBodyUserData(True, name="body")
        self.body_parts.append(body_p2)

        joint_motor = world.CreateJoint(rjd)
        joint_motor.userData = CustomMotorUserData(SPEED_KNEE, True, contact_body=body_p2)
        self.motors.append(joint_motor)

        #### P3 ####
        body_p3_x = body_p2_x - 16 / 2 / self.SCALE - 8 / 2 / self.SCALE
        BODY_P3_FD = fixtureDef(
            shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in BODY_P3]),
            density=self.DENSITY,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x000F)

        body_p3 = world.CreateDynamicBody(
            position=(body_p3_x, init_y),
            fixtures=BODY_P3_FD
        )
        body_p3.color1 = (0.5, 0.4, 0.9)
        body_p3.color2 = (0.3, 0.3, 0.5)

        rjd = revoluteJointDef(
            bodyA=body_p2,
            bodyB=body_p3,
            anchor=(body_p2_x - 16 / 2 / self.SCALE, init_y),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=self.MOTORS_TORQUE,
            motorSpeed=1,
            lowerAngle=-0.3 * np.pi,
            upperAngle=0.3 * np.pi,
        )

        body_p3.userData = CustomBodyUserData(True, name="body")
        self.body_parts.append(body_p3)
        self.tail = body_p3

        joint_motor = world.CreateJoint(rjd)
        joint_motor.userData = CustomMotorUserData(SPEED_KNEE, True, contact_body=body_p3)
        self.motors.append(joint_motor)

        #### FIN ####
        FIN_FD = fixtureDef(
            shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in FIN]),
            density=self.DENSITY,
            restitution=0.0,
            categoryBits=0x20,
            maskBits=0x000F)

        fin_positions = [
            # [init_x + 35 / 2 / self.SCALE / 2, init_y],
            # [init_x - 35 / 2 / self.SCALE / 2, init_y],
            [init_x, init_y - 22 / 2 / self.SCALE + 0.2],
            # [init_x - 35 / 2 / self.SCALE / 2, init_y - 22 / 2 / self.SCALE + 0.1],
        ]

        fin_angle = -0.2 * np.pi
        middle_fin_x_distance = np.sin(fin_angle) * 20 / 2 / self.SCALE
        middle_fin_y_distance = np.cos(fin_angle) * 20 / 2 / self.SCALE

        for fin_pos in fin_positions:
            current_fin_x = fin_pos[0] + middle_fin_x_distance
            current_fin_y = fin_pos[1] - middle_fin_y_distance

            fin = world.CreateDynamicBody(
                position=(current_fin_x, current_fin_y),
                fixtures=FIN_FD,
                angle=fin_angle
            )
            fin.color1 = (0.5, 0.4, 0.9)
            fin.color2 = (0.3, 0.3, 0.5)

            rjd = revoluteJointDef(
                bodyA=hull,
                bodyB=fin,
                anchor=(fin_pos[0], fin_pos[1]),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-0.3 * np.pi,
                upperAngle=0.2 * np.pi,
            )

            fin.userData = CustomBodyUserData(True, name="fin")
            self.body_parts.append(fin)
            self.fins.append(fin)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(SPEED_KNEE, True, contact_body=fin)
            self.motors.append(joint_motor)