import numpy as np

class AbstractBody(object):
    def __init__(self, scale, motors_torque):
        self.SCALE = scale
        self.MOTORS_TORQUE = motors_torque

        self.body_parts = [] # list of objects constituting the body
        self.motors = [] # list of motors

    # States
    def get_state_size(self):
        return len(self.get_motors_state())

    def get_motors_state(self):
        state = []
        for motor in self.motors:
            motor_info = motor.userData
            if motor_info.check_contact:
                state.extend([
                    motor.angle + motor_info.angle_correction,
                    motor.speed / motor_info.speed_control,
                    1.0 if motor_info.contact_body.userData.has_contact else 0.0 # If motor add check_contact=True, check the contact of the associated contact_body
                ])
            else:
                state.extend([
                    motor.angle + motor_info.angle_correction,
                    motor.speed / motor_info.speed_control
                ])
        return state

    # Actions
    def get_action_size(self):
        return len(self.motors)

    def activate_motors(self, action):
        for i in range(len(self.motors)):
            self.motors[i].motorSpeed  = float(self.motors[i].userData.speed_control * np.sign(action[i]))
            self.motors[i].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[i]), 0, 1))

    # Draw
    def draw(self, world, init_x, init_y, force_to_center):
        pass

    def get_elements_to_render(self):
        return self.body_parts

    # Destroy
    def destroy(self, world):
        for body_part in self.body_parts:
            world.DestroyBody(body_part)
        self.body_parts = []
        self.motors = []





