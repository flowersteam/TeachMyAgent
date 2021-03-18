import numpy as np
from Box2D.b2 import edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener

from TeachMyAgent.environments.envs.bodies.AbstractBody import AbstractBody
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomBodyUserData, CustomMotorUserData

##### CURRENTLY NOT USABLE IN TEACH MY AGENT #####
SPEED_HIP     = 4
SPEED_KNEE    = 6
SPEED_HAND    = 8

class BackChimpanzee(AbstractBody):
    def __init__(self, scale, motors_torque = 500):
        super(BackChimpanzee, self).__init__(scale, motors_torque)
        self.LIMB_W, self.LIMB_H = 8 / self.SCALE, 28 / self.SCALE
        self.HAND_PART_W, self.HAND_PART_H = 4 / self.SCALE, 8 / self.SCALE
        self.LEG_H = self.LIMB_H
        self.TORQUE_PENALTY = 0.00035 / 5 # Legs + arms + hands
        self.BODY_HEIGHT = 47
        self.BODY_WIDTH = 30
        self.HEAD_HEIGHT = 20

        self.AGENT_WIDTH = 40 / self.SCALE
        self.AGENT_HEIGHT = self.BODY_HEIGHT / self.SCALE + \
                            self.HEAD_HEIGHT / self.SCALE + 0.2 + \
                            self.LEG_H * 2
        self.AGENT_CENTER_HEIGHT = self.BODY_HEIGHT / self.SCALE / 2 + \
                                   self.LEG_H * 2

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

        BODY_POLYGONS = [
            [(-20, +25), (+20, +25),
             (+4, -20), (-4, -20)],
            [(-15, -20), (+15, -20),
             (-15, -22), (+15, -22)]
        ]

        BODY_FIXTURES = [
            fixtureDef(
                shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in polygon]),
                density=5.0,
                friction=0.1,
                categoryBits=0x20,
                maskBits=0x1)
            for polygon in BODY_POLYGONS
        ]

        body = world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=BODY_FIXTURES
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

        # Left / Right side of body
        for i in [-1, +1]:
            current_x = init_x + i*self.BODY_WIDTH / self.SCALE / 2

            # LEG
            leg_top_y = init_y - self.BODY_HEIGHT / self.SCALE / 2
            upper = world.CreateDynamicBody(
                position=(current_x, leg_top_y - self.LIMB_H / 2),
                fixtures=UPPER_LIMB_FD
            )

            upper.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            upper.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=body,
                bodyB=upper,
                anchor=(current_x, leg_top_y),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=int(i < 0) * -0.75 * np.pi,
                upperAngle=int(i > 0) * 0.75 * np.pi,
            )

            upper.userData = CustomBodyUserData(False, name="upper_leg")
            self.body_parts.append(upper)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(SPEED_HIP, False)
            self.motors.append(joint_motor)

            lower = world.CreateDynamicBody(
                position=(current_x, leg_top_y - self.LIMB_H * 3 / 2),
                fixtures=LOWER_LIMB_FD
            )
            lower.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            lower.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=upper,
                bodyB=lower,
                anchor=(current_x, leg_top_y - self.LIMB_H),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=int(i > 0) * -0.7 * np.pi,
                upperAngle=int(i < 0) * 0.7 * np.pi,
            )

            lower.userData = CustomBodyUserData(True, name="lower_leg")
            self.body_parts.append(lower)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(
                SPEED_KNEE,
                True,
                contact_body=lower,
                angle_correction=1.0)
            self.motors.append(joint_motor)

            # ARM
            arm_top_y = init_y + self.BODY_HEIGHT / self.SCALE / 2

            upper = world.CreateDynamicBody(
                position=(current_x, arm_top_y - self.LIMB_H / 2),
                fixtures=UPPER_LIMB_FD
            )
            upper.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            upper.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=body,
                bodyB=upper,
                anchor=(current_x, arm_top_y),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=int(i < 0) * -0.9 * np.pi,
                upperAngle=int(i > 0) * 0.9 * np.pi,
            )

            upper.userData = CustomBodyUserData(False, name="upper_arm")
            self.body_parts.append(upper)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(SPEED_HIP, False)
            self.motors.append(joint_motor)

            lower = world.CreateDynamicBody(
                position=(current_x, arm_top_y - self.LIMB_H * 3 / 2),
                fixtures=LOWER_LIMB_FD
            )

            lower.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            lower.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=upper,
                bodyB=lower,
                anchor=(current_x, arm_top_y - self.LIMB_H),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=int(i < 0) * -0.8 * np.pi,
                upperAngle=int(i > 0) * 0.8 * np.pi,
            )

            lower.userData = CustomBodyUserData(False, name="lower_arm")
            self.body_parts.append(lower)

            joint_motor = world.CreateJoint(rjd)
            joint_motor.userData = CustomMotorUserData(
                SPEED_KNEE,
                False)
            self.motors.append(joint_motor)

            # hand
            prev_part = lower
            initial_hand_y = arm_top_y - self.LIMB_H * 2
            angle_boundaries = [[-0.4, 0.4], [-0.5, 0.5], [-0.8, 0.8]]
            for u in range(3):
                hand_part = world.CreateDynamicBody(
                    position=(current_x, initial_hand_y - self.HAND_PART_H / 2 - self.HAND_PART_H * u),
                    fixtures=HAND_PART_FD
                )

                hand_part.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
                hand_part.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
                rjd = revoluteJointDef(
                    bodyA=prev_part,
                    bodyB=hand_part,
                    anchor=(current_x, initial_hand_y - self.HAND_PART_H * u),
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
                joint_motor.userData = CustomMotorUserData(
                    SPEED_HAND,
                    True,
                    contact_body=hand_part)
                self.motors.append(joint_motor)

                prev_part = hand_part

