import numpy as np

class AbstractBody(object):
    '''
        Base class for all embodiments.
    '''
    def __init__(self, scale, motors_torque):
        '''
            Creates an embodiment.

            Args:
                scale: Scale value used in the environment (to adapt the embodiment to its environment)
                motors_torque: Maximum torque the embodiment can use on its motors
        '''
        self.SCALE = scale
        self.MOTORS_TORQUE = motors_torque

        self.body_parts = [] # list of objects constituting the body
        self.motors = [] # list of motors

    # States
    def get_state_size(self):
        '''
            Returns the size of the embodiment's state vector
        '''
        return len(self.get_motors_state())

    def get_motors_state(self):
        '''
            Returns state vector of motors.

            For each motor returns:
            - its angle
            - its speed
            - if `motor.userData` has `check_contact` to True, returns whether the associated body checking contact has contact
        '''
        state = []
        for motor in self.motors:
            motor_info = motor.userData
            if motor_info.check_contact:
                state.extend([
                    motor.angle + motor_info.angle_correction,
                    motor.speed / motor_info.speed_control,
                    1.0 if motor_info.contact_body.userData.has_contact else 0.0 # If motor has check_contact=True, check the contact of the associated contact_body
                ])
            else:
                state.extend([
                    motor.angle + motor_info.angle_correction,
                    motor.speed / motor_info.speed_control
                ])
        return state

    # Actions
    def get_action_size(self):
        '''
            Returns the size of the action space.
        '''
        return len(self.motors)

    def activate_motors(self, action):
        '''
            Activate motors given a vector of actions (between -1 and 1).

            Sets `motorSpeed` to `speed_control * sign(action)`.
            Sets `maxMotorTorque` to `MOTORS_TORQUE * abs(action)` (with `MOTORS_TORQUE` the torque set in the constructor).
        '''
        for i in range(len(self.motors)):
            self.motors[i].motorSpeed  = float(self.motors[i].userData.speed_control * np.sign(action[i]))
            self.motors[i].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[i]), 0, 1))

    # Draw
    def draw(self, world, init_x, init_y, force_to_center):
        '''
            Creates fixtures and bodies in the Box2D world.
        '''
        pass

    def get_elements_to_render(self):
        '''
            Returns bodies that must be rendered in the `env.render` function.
        '''
        return self.body_parts

    # Destroy
    def destroy(self, world):
        for body_part in self.body_parts:
            world.DestroyBody(body_part)
        self.body_parts = []
        self.motors = []





