import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from copy import copy
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomUserDataObjectTypes
import numpy as np

class WaterDynamics(object):
    '''
    Calculate forces to simulate in a simplified way water physics.
    Most of the forces implemented here are taken from https://www.iforce2d.net/b2dtut/buoyancy except the push force implemented by Clément Romac.
    '''
    def __init__(self, gravity, drag_mod=0.25, lift_mod=0.25, push_mod=0.05, max_drag=2000, max_lift=500, max_push=20):
        self.gravity = gravity
        self.drag_mod = drag_mod
        self.lift_mod = lift_mod
        self.max_drag = max_drag
        self.max_lift = max_lift
        self.push_mod = push_mod
        self.max_push = max_push

    def compute_centroids(self, vectors):
        count = len(vectors)
        assert count >= 3

        c = Box2D.b2Vec2(0, 0)
        area = 0

        ref_point = Box2D.b2Vec2(0, 0)
        inv3 = 1/3

        for i in range(count):
            # Triangle vertices
            p1 = ref_point
            p2 = vectors[i]
            p3 = vectors[i+1] if i+1 < count else vectors[0]

            e1 = p2 - p1
            e2 = p3 - p1

            d = Box2D.b2Cross(e1, e2)

            triangle_area = 0.5 * d
            area += triangle_area

            # Area weighted centroid
            c += triangle_area * inv3 * (p1 + p2 + p3)

        if area > Box2D.b2_epsilon:
            c *= 1/area
        else:
            area = 0

        return c, area

    def inside(self, cp1, cp2, p):
        return (cp2.x-cp1.x)*(p.y-cp1.y) > (cp2.y-cp1.y)*(p.x-cp1.x)

    def intersection(self, cp1, cp2, s, e):
        dc = Box2D.b2Vec2(cp1.x - cp2.x, cp1.y - cp2.y)
        dp = Box2D.b2Vec2(s.x - e.x, s.y - e.y)
        n1 = cp1.x * cp2.y - cp1.y * cp2.x
        n2 = s.x * e.y - s.y * e.x
        n3 = 1.0 / (dc.x * dp.y - dc.y * dp.x)
        return Box2D.b2Vec2((n1*dp.x - n2*dc.x) * n3,
                            (n1*dp.y - n2*dc.y) * n3)

    def find_intersection(self, fixture_A, fixture_B):
        # assert polygons TODO
        output_vertices = []
        polygon_A = fixture_A.shape
        polygon_B = fixture_B.shape

        #fill 'subject polygon' from fixtureA polygon
        for i in range(len(polygon_A.vertices)):
            output_vertices.append(fixture_A.body.GetWorldPoint(polygon_A.vertices[i]))

        #fill 'clip polygon' from fixtureB polygon
        clip_polygon = []
        for i in range(len(polygon_B.vertices)):
            clip_polygon.append(fixture_B.body.GetWorldPoint(polygon_B.vertices[i]))

        cp1 = clip_polygon[len(clip_polygon) - 1]
        for j in range(len(clip_polygon)):
            cp2 = clip_polygon[j]
            if len(output_vertices) == 0:
                break

            input_list = copy(output_vertices)
            output_vertices.clear()

            s = input_list[len(input_list) - 1]
            for i in range(len(input_list)):
                e = input_list[i]
                if self.inside(cp1, cp2, e):
                    if not self.inside(cp1, cp2, s):
                        output_vertices.append(self.intersection(cp1, cp2, s, e))
                    output_vertices.append(e)
                elif self.inside(cp1, cp2, s):
                    output_vertices.append(self.intersection(cp1, cp2, s, e))
                s = e
            cp1 = cp2
        return len(output_vertices) != 0, output_vertices

    def calculate_forces(self, fixture_pairs):
        for pair in fixture_pairs:
            density = pair[0].density

            has_intersection, intersection_points = self.find_intersection(pair[0], pair[1])

            if has_intersection:
                centroid, area = self.compute_centroids(intersection_points)

                # apply buoyancy force
                displaced_mass = pair[0].density * area
                buoyancy_force = displaced_mass * -self.gravity
                pair[1].body.ApplyForce(force=buoyancy_force, point=centroid, wake=True)

                # apply complex drag
                for i in range(len(intersection_points)):
                    v0 = intersection_points[i]
                    v1 = intersection_points[(i+1)%len(intersection_points)]
                    mid_point = 0.5 * (v0 + v1)

                    ##### DRAG
                    # find relative velocity between object and fluid at edge midpoint
                    vel_dir = pair[1].body.GetLinearVelocityFromWorldPoint(mid_point) - \
                              pair[0].body.GetLinearVelocityFromWorldPoint(mid_point)
                    vel = vel_dir.Normalize()

                    edge = v1 - v0
                    edge_length = edge.Normalize()
                    normal = Box2D.b2Cross(-1, edge)
                    drag_dot = Box2D.b2Dot(normal, vel_dir)
                    if drag_dot >= 0: # normal points backwards - this is not a leading edge
                        # apply drag
                        drag_mag = drag_dot * self.drag_mod * edge_length * density * vel * vel
                        drag_mag = min(drag_mag, self.max_drag)
                        drag_force = drag_mag * -vel_dir
                        pair[1].body.ApplyForce(force=drag_force,
                                                point=mid_point,
                                                wake=True)

                        # apply lift
                        lift_dot = Box2D.b2Dot(edge, vel_dir)
                        lift_mag = drag_dot * lift_dot * self.lift_mod * edge_length * density * vel * vel
                        lift_mag = min(lift_mag, self.max_lift)
                        lift_dir = Box2D.b2Cross(1, vel_dir)
                        lift_force = lift_mag * lift_dir
                        pair[1].body.ApplyForce(force=lift_force,
                                                point=mid_point,
                                                wake=True)
                    ##### PUSH
                    # Apply a linear force to an object linked to a rotation joint applying torque.
                    # Torque and angular inertia are used to calculate the magnitude of the linear force

                    body_to_check = pair[1].body
                    # Simplification /!\
                    # joint = pair[1].body.joints[0].joint
                    # joints_to_check = [joint_edge.joint for joint_edge in body_to_check.joints]
                    joints_to_check = [joint_edge.joint for joint_edge in body_to_check.joints if
                                 joint_edge.joint.bodyB == body_to_check]

                    for joint in joints_to_check:
                        if joint.lowerLimit < joint.angle < joint.upperLimit:
                            torque = joint.GetMotorTorque(60)

                            # Calculate angular inertia of the object
                            moment_of_inertia = body_to_check.inertia
                            angular_velocity = body_to_check.angularVelocity
                            angular_inertia = moment_of_inertia * angular_velocity

                            # Calculate the force applied to the object
                            world_center = body_to_check.worldCenter
                            anchor = joint.anchorB
                            lever_vector = world_center - anchor # vector from pivot to point of application of the force
                            force_applied_at_center = Box2D.b2Cross(lever_vector, -torque)

                            push_dot = Box2D.b2Dot(normal, force_applied_at_center)
                            if push_dot > 0:
                                vel = torque + angular_inertia
                                push_mag = push_dot * self.push_mod * edge_length * density * vel * vel # Wrong approximation /!\
                                # push_mag = min(push_mag, self.max_push)
                                push_force = np.clip(push_mag * -force_applied_at_center, -self.max_push, self.max_push)
                                body_to_check.ApplyForce(force=push_force,
                                                        point=joint.anchorB,#body_to_check.worldCenter,
                                                        wake=True)

class WaterContactDetector(contactListener):
    '''
    Store fixtures having contact with water.
    '''
    def __init__(self):
        super(WaterContactDetector, self).__init__()
        self.fixture_pairs = []
    def BeginContact(self, contact):
        if contact.fixtureA.body.userData.object_type == CustomUserDataObjectTypes.WATER and \
                        contact.fixtureB.body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT:
            self.fixture_pairs.append((contact.fixtureA,
                                       contact.fixtureB))
        elif contact.fixtureB.body.userData.object_type == CustomUserDataObjectTypes.WATER and \
                        contact.fixtureA.body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT:
            self.fixture_pairs.append((contact.fixtureB,
                                       contact.fixtureA))

    def EndContact(self, contact):
        if contact.fixtureA.body.userData.object_type == CustomUserDataObjectTypes.WATER and \
                        contact.fixtureB.body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT:
            self.fixture_pairs.remove((contact.fixtureA,
                                       contact.fixtureB))
        elif contact.fixtureB.body.userData.object_type == CustomUserDataObjectTypes.WATER and \
                        contact.fixtureA.body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT:
            self.fixture_pairs.remove((contact.fixtureB,
                                       contact.fixtureA))

    def Reset(self):
        self.fixture_pairs = []