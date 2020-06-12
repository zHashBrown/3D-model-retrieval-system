import bpy
import os.path
import math
import sys

C = bpy.context
D = bpy.data
scene = D.scenes['Scene']

# cameras: a list of camera positions
# a camera position is defined by two parameters: (theta, phi),
# where we fix the "r" of (r, theta, phi) in spherical coordinate system.

# 5 orientations: front, right, back, left, top
fixed_view = 60
inter = 30
cameras = [(fixed_view, i) for i in range(0, 360, inter)]  # 这会生成360/30=12个视角图片

# 12 orientations around the object with 30-deg elevation
# cameras = [(60, i) for i in range(0, 360, 30)]

render_setting = scene.render

# output image size = (W, H)
w = 600 * 2
h = 600 * 2
render_setting.resolution_x = w
render_setting.resolution_y = h


def main():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:]

    if len(argv) != 2:
        print('phong.py args: <3d mesh path> <image dir>')
        exit(-1)

    model = argv[0]
    image_dir = argv[1]

    # blender has no native support for off files
    install_off_addon()

    init_camera()
    fix_camera_to_origin()

    do_model(model, image_dir)


def install_off_addon():
    filepath = os.path.dirname(__file__) + '\import_off.py'
    bpy.ops.wm.addon_install(
        overwrite=False,
        filepath=filepath
    )
    bpy.ops.wm.addon_enable(module='import_off')


def init_camera():
    cam = D.objects['Camera']
    # select the camera object
    scene.objects.active = cam
    cam.select = True

    # set the rendering mode to orthogonal and scale
    C.object.data.type = 'ORTHO'
    C.object.data.ortho_scale = 2.


def fix_camera_to_origin():
    origin_name = 'Origin'

    # create origin
    try:
        origin = D.objects[origin_name]
    except KeyError:
        bpy.ops.object.empty_add(type='SPHERE')
        D.objects['Empty'].name = origin_name
        origin = D.objects[origin_name]

    origin.location = (0, 0, 0)

    cam = D.objects['Camera']
    scene.objects.active = cam
    cam.select = True

    if 'Track To' not in cam.constraints:
        bpy.ops.object.constraint_add(type='TRACK_TO')

    cam.constraints['Track To'].target = origin
    cam.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    cam.constraints['Track To'].up_axis = 'UP_Y'


def do_model(path, image_dir):
    ext = path.split('.')[-1].lower()
    name = load_model(path)
    center_model(name)
    normalize_model(name)
    image_subdir = os.path.join(image_dir, name)

    for i, c in enumerate(cameras):
        move_camera(c)
        render()
        save(image_subdir, '%s.%d' % (name, i), ext)

    delete_model(name)


def load_model(path):
    d = os.path.dirname(path)
    ext = path.split('.')[-1].lower()

    name = os.path.basename(path).split('.')[0]
    # handle weird object naming by Blender for stl files
    # if ext == 'stl':
    #    name = name.title().replace('_', ' ')

    if name not in D.objects:
        print('loading :' + name)

        with open('./tmp/' + name + '.' + ext + '.txt', 'w') as file_handle:  # 新建保存图像路径的txt
            file_handle.write('-1\n12\n')  # 写入

        if ext == 'stl':
            bpy.ops.import_mesh.stl(filepath=path, directory=d,
                                    filter_glob='*.stl')

        ##########
        elif ext == 'ply':
            bpy.ops.import_mesh.ply(filepath=path, directory=d,
                                    filter_glob="*.ply")
        elif ext == '3ds':
            bpy.ops.import_scene.autodesk_3ds(filepath=path, filter_glob="*.3ds")
        elif ext == 'fbx':
            bpy.ops.import_scene.fbx(filepath=path, directory=d, filter_glob="*.fbx")
        elif ext == 'x3d':
            bpy.ops.import_scene.x3d(filepath=path, filter_glob="*.x3d")
        ##########

        elif ext == 'off':
            bpy.ops.import_mesh.off(filepath=path, filter_glob='*.off')
        elif ext == 'obj':
            bpy.ops.import_scene.obj(filepath=path, filter_glob='*.obj')
        else:
            print('Currently .{} file type is not supported.'.format(ext))
            exit(-1)
    return name


def delete_model(name):
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.name.startswith(name):
            ob.select = True
        else:
            ob.select = False
    bpy.ops.object.delete()


def center_model(name):
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
    try:
        D.objects[name].location = (0, 0, 0)
    except:
        D.objects[0].location = (0, 0, 0)
        print('except！！！')


def normalize_model(name):
    try:
        obj = D.objects[name]
    except:
        obj = D.objects[0]
        print('except！！！')
    dim = obj.dimensions
    print('original dim:' + str(dim))
    if max(dim) > 0:
        dim = dim / max(dim)
    obj.dimensions = dim

    print('new dim:' + str(dim))


def move_camera(coord):
    def deg2rad(deg):
        return deg * math.pi / 180.

    r = 3.
    theta, phi = deg2rad(coord[0]), deg2rad(coord[1])
    loc_x = r * math.sin(theta) * math.cos(phi)
    loc_y = r * math.sin(theta) * math.sin(phi)
    loc_z = r * math.cos(theta)

    D.objects['Camera'].location = (loc_x, loc_y, loc_z)


def render():
    bpy.ops.render.render()


def save(image_dir, name, ext):
    path = os.path.join(image_dir, name + '.png')
    D.images['Render Result'].save_render(filepath=path)
    print('save to ' + path)

    with open('./tmp/' + name.split('.')[-2] + '.' + ext + '.txt', 'a') as file_handle:
        file_handle.write(path + '\n')  # 写入


if __name__ == '__main__':
    main()
