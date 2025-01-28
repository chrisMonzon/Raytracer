import sys
from math import sqrt, atan2, pi, sin, cos, exp
from PIL import Image


image = None
width = 0
height = 0
spheres = []
colors = []
rays = []
suns = []
bulbs = []
planes = []
vertices = []
triangles = []
filename = ""
expose_v = -1
eye = (0, 0, 0)
g_forward = (0, 0, -1)
g_up = (0, 1, 0)
g_right = (1, 0, 0)
forward = (0, 0, -1)
up = (0, 1, 0)
right = (1, 0, 0)
fisheye = 0
panorama = 0
aa_val = 1.0

def ab_add(a, b):
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))

def ab_sub(a, b):
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))

def ab_mul(a, b):
    return (float(a[0]) * float(b[0]), float(a[1]) * float(b[1]), float(a[2]) * float(b[2]))

def ab_dot(a, b):
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])

def ac_add(a, c):
    return (a[0] + float(c), a[1] + float(c), a[2] + float(c))

def ac_mul(a, c):
    return (a[0] * float(c), a[1] * float(c), a[2] * float(c))

def ac_div(a, c):
    return (a[0] / float(c), a[1] / float(c), a[2] / float(c))

def vector_length_squared(v):
    # print("vls: ", v, type(v[0]))

    return sum(float(a) * float(a) for a in v)

def distPoints(a, b):
    return sqrt((float(a[0]) - float(b[0])) ** 2.0 + (float(a[1]) - float(b[1])) ** 2.0 + (float(a[2]) - float(b[2])) ** 2.0)

def normalize(v):
    length = sqrt(vector_length_squared(v))
    # print("v = ", v, "length = ", length)
    return tuple(a / float(length) for a in v)

def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )
def perpendicular_to_edge(edge, reference=(0, 0, 1)):
    edge = normalize(edge)
    reference = normalize(reference)
    
    # Check if the edge is parallel to the reference
    if abs(ab_dot(edge, reference)) > 0.999:  # Almost parallel
        # Use a different reference vector
        reference = (0, 1, 0) if edge != (0, 1, 0) else (1, 0, 0)
    
    # Compute a perpendicular vector using the cross product
    perpendicular = cross(edge, reference)
    return normalize(perpendicular)

def parse_file(file_path):
    print("Parse file called")
    global image, width, height, colors, spheres, rays, filename, suns, planes, triangles, bulbs
    global expose_v, eye, forward, up, right
    global eye, g_forward, g_up, aa_val
    global fisheye, panorama
    rays = []
    suns = []
    spheres = []
    planes = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the PNG line
    png_line = lines[0].strip().split()
    width = int(png_line[1])
    height = int(png_line[2])
    filename = png_line[3]
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    # current_color = (255, 255, 255)  # Default white color
    current_color = (1, 1, 1)  # Default white color
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        if parts[0] == "color":
            current_color = tuple((float(c)) for c in parts[1:])
        elif parts[0] == "sphere":
            sphere = [(float(parts[1]), float(parts[2]), float(parts[3])), float(parts[4]), current_color]
            # (center), radius, (color)
            print("sphere = ",sphere)
            spheres.append(sphere)
        elif parts[0] == "plane":
            plane = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), current_color]
            # (center), radius, (color)
            print("plane = ",plane)
            planes.append(plane)
            # spheres.append(plane)
        elif parts[0] == "xyz":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            print("appending vertex =", (float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "tri":
            plane = []
            # plane = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1, current_color]
            for i in range(1,4):
                if (int(parts[i]) >= 0):
                    plane.append(int(parts[i]) - 1)
                else:
                    plane.append(int(parts[i]))
            plane.append(current_color)
            # planes.append(plane)
            triangles.append(plane)
            print("appending triangle =", plane)
            # bruh
        elif parts[0] == "sun":
            ray = [eye, (float(parts[1]), float(parts[2]), float(parts[3])), current_color]
            print("appending sun =", ray)
            suns.append(ray)
        elif parts[0] == "bulb":
            bulb = [(float(parts[1]), float(parts[2]), float(parts[3])), current_color]
            print("appending bulb =", bulb)
            bulbs.append(bulb)
        elif parts[0] == "expose":
            expose_v = float(parts[1])
        elif parts[0] == "up":
            up = (float(parts[1]), float(parts[2]), float(parts[3]))
            right = normalize(cross(forward, up))
            up = normalize(cross(right, forward))
        elif parts[0] == "eye":
            eye = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif parts[0] == "forward":
            forward = (float(parts[1]), float(parts[2]), float(parts[3]))
            right = normalize(cross(forward, up))
            up = normalize(cross(right, forward))
        elif parts[0] == "fisheye":
            fisheye = 1
        elif parts[0] == "panorama":
            panorama = 1
        elif parts[0] == "aa":
            aa_val = int(parts[1])

def get_distance(a, b):
    # print("get dist called")
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def apply_color(intersection_point, normal, sphere, light_src):
    global suns
    # total_intensity = (0,0,0)
    # sun_dir = normalize(sun[1])
    # sun_color = sun[2]
    # intensity = max(0, ab_dot(normal, sun_dir))
    # total_intensity = ab_add(total_intensity, ac_mul(sun_color, intensity))
    light_dir = (0, 0, 0)
    light_color = (0, 0, 0)
    distance_factor = 1.0

    if len(light_src) == 3: # sun
        light_dir = normalize(light_src[1])  # Dirfction of sunlight
        light_color = light_src[2]
    else: # bulb
        light_dir = normalize(ab_sub(light_src[0], intersection_point))  # Point light direction
        light_color = light_src[1]
        distance = distPoints(light_src[0], intersection_point)
        distance_factor = 1.0 / (distance * distance)  # Intensity decreases with square of distance

    # Compute light intensity and apply distance factor
    intensity = max(0.0, ab_dot(normal, light_dir) * distance_factor)
    total_intensity = ac_mul(light_color, intensity)

    toRet = (0,0,0)
    if len(sphere) == 3: # sphere
        toRet = ab_mul(sphere[2], total_intensity)
    elif (len(sphere) == 5): # infinite plane
        toRet = ab_mul(sphere[4], total_intensity)
    elif (len(sphere) == 4): # triangle
        toRet = ab_mul(sphere[3], total_intensity)
    toRet = (toRet[0], toRet[1], toRet[2])
    
    # print("toRetColor=", toRet)
    return toRet

def ray_plane_intersect(plane, ray):
    global vertices
    r0 = (ray[0][0], ray[0][1], ray[0][2])
    rd = normalize((ray[1][0], ray[1][1], ray[1][2]))
    if (len(plane) == 5): # normal plane
        n = (float(plane[0]), float(plane[1]), float(plane[2]))
        # n = normalize(float(plane[0]), float(plane[1]), float(plane[2]))
        p = ac_div(ac_mul(n, -plane[3]), plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
        if ab_dot(rd, n) == 0:
            # if abs(ab_dot(rd, n)) < 10**(-10):
            return None, None
        t = ab_dot(ab_sub(p, r0), n) / ab_dot(rd, n)
        if t > 0:
            return t, ab_add(r0, ac_mul(rd, t))
        else:
            return None, None
    else: # triangle
        # print("triangle intersection")
        p0 = vertices[plane[0]]
        p1 = vertices[plane[1]]
        p2 = vertices[plane[2]]
        print("vertices = ", p0, p1, p2)
        n = cross(ab_sub(p1, p0), ab_sub(p2, p0))
        if ab_dot(n, rd) > 0:
            n = (-n[0], -n[1], -n[2])
        denom = ab_dot(rd, n)
        if abs(denom) < 10**(-6):  # Ray is parallel to the plane
            return None, None
            # Compute the distance along the ray to the plane
        t = ab_dot(ab_sub(p0, r0), n) / denom
        if t <= 0:  
            return None, None
        
        intersection_point = ab_add(r0, ac_mul(rd, t))
        
        a1 = cross(ab_sub(p2, p0), n)  # Perpendicular vector for edge p2 -> p0
        a2 = cross(ab_sub(p1, p0), n)  # Perpendicular vector for edge p1 -> p0

        e1 = ac_div(a1, ab_dot(a1, ab_sub(p1, p0)))
        e2 = ac_div(a2, ab_dot(a2, ab_sub(p2, p0)))

        # Compute Barycentric coordinates
        b1 = ab_dot(e1, ab_sub(intersection_point, p0))
        b2 = ab_dot(e2, ab_sub(intersection_point, p0))
        b0 = 1 - b1 - b2  # Enforce sum to 1

        epsilon = 1e-6
        if b0 >= -epsilon and b1 >= -epsilon and b2 >= -epsilon:
            return t, intersection_point
        else:
            return None, None  # Intersection point is outside the triangle
    return None, None
    

def ray_sphere_intersect(sphere, ray):
    # print("ray sphere intersect called")
    center = (sphere[0][0], sphere[0][1], sphere[0][2])
    radius = sphere[1]
    r0 = (ray[0][0], ray[0][1], ray[0][2])
    # print("ray = ", ray)
    rd = normalize((ray[1][0], ray[1][1], ray[1][2]))
    inside = vector_length_squared(ab_sub(center, r0))
    cminusr = ab_sub(center, r0)
    tc = ab_dot(cminusr, rd) / sqrt(vector_length_squared(rd))
    if (not (inside  < radius**2)and tc < 0):
        # print("RSI 1-> None, tc =", tc, "r2 =", radius**2, "ns =", inside)
        return None, None
    d2 = vector_length_squared(ab_sub(ab_add(r0, ac_mul(rd, tc)), center))
    if (not (inside  < radius**2) and d2 > radius**2):
        # print("RSI 2-> None")
        return None, None
    # print("r2 = ", radius**2, "d2=", d2)
    t_offset = sqrt(abs(radius**2 - d2)) / sqrt(vector_length_squared(rd))
    t = tc - t_offset
    if (inside < radius**2):
        t = tc + t_offset
    else:
        t = tc - t_offset
    intersection_point = ab_add(r0, ac_mul(rd, t))
    # print("Found intersection at depth", t, " at", intersection_point)
    return t, intersection_point

def find_hit_object(ray):
    # print("Find hit object called")
    global spheres, planes, triangles
    closest_t = float('inf')
    closest_sphere = None
    int_point_final = (0,0,0)
    for sphere in spheres:
        t, int_point = ray_sphere_intersect(sphere, ray)
        if t is not None and t < closest_t:
            closest_t = t
            closest_sphere = sphere
            int_point_final = int_point
    
    # Check intersections with planes
    for plane in planes:
        t, int_point = ray_plane_intersect(plane, ray)
        if t is not None and t < closest_t:
            closest_t = t
            closest_sphere = plane
            int_point_final = int_point
    
    for triangle in triangles:
        t, int_point = ray_plane_intersect(triangle, ray)
        if t is not None and t < closest_t:
            closest_t = t
            closest_sphere = triangle
            int_point_final = int_point
    return closest_t, int_point_final, closest_sphere

def to_srgb(L_linear):
    global expose_v
    if (expose_v != -1):
        L_linear = 1 - exp(-expose_v * L_linear)
    if L_linear <= 0.0031308:
        L_linear = 12.92 * L_linear
    else:
        L_linear = 1.055 * (L_linear ** (1 / 2.4)) - 0.055
    return L_linear

def raytrace(ray):
    # print("Raytrace called")
    global suns
    epsilon = 1 * 10**(-5)
    # sun[0] = light source origin, sun[1] = light direction
    t, intersection_point, object_hit = find_hit_object(ray)
    normal = (0,0,0)
    if object_hit:
        if (len(object_hit) == 3): # object is a sphere
            # print(len(object_hit))
            center = object_hit[0]
        
            radius = object_hit[1]
            normal = ac_mul(ab_sub(intersection_point, center), (1.0/float(radius)))
            if (ab_dot(ray[1], normal) > 0):
                normal = (-normal[0], -normal[1], -normal[2])
            normal = normalize(normal)
        else: # object is a plane
            # normal = (object_hit[0], object_hit[1], object_hit[2])
            normal = normalize((object_hit[0], object_hit[1], object_hit[2]))
        final_color = (0, 0, 0)
        is_in_shadow = False
        for sun in suns: 
            light_dir = normalize(sun[1])  # Direction to the light source
            light_ray = (ab_add(intersection_point, ac_mul(normal, epsilon)), light_dir)
            light_distance = distPoints(intersection_point, sun[0])  # Distance to light source

            # Check for any objects blocking the light
            shadow_t, _, shadow_sphere = find_hit_object(light_ray)
            if (shadow_sphere and shadow_t > epsilon and shadow_t < light_distance):
                is_in_shadow = True
            else:
                final_color = ab_add(final_color, apply_color(intersection_point, normal, object_hit, sun))
        for bulb in bulbs:
            light_dir = normalize(ab_sub(bulb[0], intersection_point))  # Direction to the bulb
            light_distance = distPoints(bulb[0], intersection_point)  # Distance to the bulb
            light_ray = (ab_add(intersection_point, ac_mul(normal, epsilon)), light_dir)

            # Check for shadows
            shadow_t, _, shadow_object = find_hit_object(light_ray)
            if shadow_object and shadow_t > epsilon and shadow_t < light_distance:
                is_in_shadow = True
            else:
                final_color = ab_add(final_color, apply_color(intersection_point, normal, object_hit, bulb))

        

        final_color = (to_srgb(final_color[0]), to_srgb(final_color[1]), to_srgb(final_color[2]))
        return final_color
    # print("Returning NONE in rt")
    return None



def createRay(x, y):
    global width, height
    global eye, forward, up, right
    global fisheye
    f = forward
    u = up
    r = right
    rayOrigin = eye

    sx = float(2.0 * x - width) / float(max(width, height))
    sy = float(height - 2.0 * y) / float(max(width, height))

    ang = 0.35
    sinAng = sin(ang)
    cosAng = cos(ang)

    if (fisheye == 1):
        if (sx ** 2 + sy ** 2 > 1):
            return None
        c0 = 1 - sx ** 2 - sy ** 2
        if c0 >= 0:
            c = sqrt(c0)
            f = ac_mul(f, c)

    sxr = ac_mul(r, sx)
    syu = ac_mul(u, sy)
    rayDir = ab_add(ab_add(f, sxr), syu)
    return [rayOrigin, normalize(rayDir), (0,0,0)]


def render():
    # print("render called")
    global image, filename, panorama, forward, eye, aa_val
    if (aa_val != 1.0):
        render_aa()
        return
    for y in range(height):
        for x in range(width):
            ray = (0,0,0)
            if panorama == 1:
                # forward = (0,0,0)
                longitude = (-x / float(width)) * 360.0
                latitude = (y / float(height)) * 180.0
                theta = longitude * (pi / 180.0)
                phi = latitude * (pi / 180.0)   
                ray_dir = (
                    sin(phi) * sin(theta), 
                    cos(phi),              
                    sin(phi) * cos(theta)  
                )
                ray = [eye, normalize(ray_dir), (0,0,0)]
            else:
                ray = createRay(x, y)
            if (ray != None):
                color = raytrace(ray)
                if (color != None):
                    # print("linear_color = ", linear_color, x, y)
                    color = tuple(min(1.0, max(0.0, c)) for c in color)
                    color = ac_mul(color, 255)
                    color = (int(color[0]),int(color[1]),int(color[2]))
                    # print("COLOR = ", color)
                    image.putpixel((x, y), color)
    image.save(filename)
    print("Image saved as", filename)
import random

def render_aa():
    global image, filename, panorama, forward, eye, aa_val

    for y in range(height):
        for x in range(width):
            total_color = [0.0, 0.0, 0.0]
            total_alpha = 0.0
            rays_per_pixel = int(aa_val)

            # Generate random rays within the pixel bounds
            for _ in range(rays_per_pixel):
                sub_x = x + random.uniform(0, 1)  # Random float within the pixel width
                sub_y = y + random.uniform(0, 1)  # Random float within the pixel height

                ray = createRay(sub_x, sub_y)
                if ray is not None:
                    color = raytrace(ray)
                    if color is not None:
                        # Add the color contribution
                        total_color[0] += color[0]
                        total_color[1] += color[1]
                        total_color[2] += color[2]
                        total_alpha += 1.0  # Increment alpha for hit rays
                    else:
                        # Missed rays only contribute to alpha
                        total_alpha += 0.0

            # Average the colors
            num_rays = rays_per_pixel
            if total_alpha > 0:
                avg_color = [c / total_alpha for c in total_color]  # Normalize RGB by hits
            else:
                avg_color = [0.0, 0.0, 0.0]  # No hits = fully transparent

            avg_alpha = total_alpha / num_rays
            final_color = tuple(min(255, max(0, int(c * 255))) for c in avg_color)
            final_alpha = int(avg_alpha * 255)

            image.putpixel((x, y), (final_color[0],final_color[1], final_color[2],final_alpha))

    # Save the image
    image.save(filename)
    print("Image saved as", filename)


# Main Execution
if __name__ == "__main__":
    input_file = sys.argv[1]
    parse_file(input_file)
    render()
