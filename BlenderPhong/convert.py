import bpy
import sys
import os, glob

C = bpy.context
D = bpy.data
scene = D.scenes['Scene']

# ply
def load_model(path):
    d = os.path.dirname(path)
    ext = path.split('.')[-1].lower()

    name = os.path.basename(path).split('.')[0]
    # handle weird object naming by Blender for stl files
    #if ext == 'stl':            # ？？？？？？？？？？
     #   name = name.title().replace('_', ' ')

    if name not in D.objects:
        print('loading :' + name)

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


def convert(path, image_dir):

    name = load_model(path)
    center_model(name)
    normalize_model(name)

    bpy.ops.export_scene.obj(filepath=image_dir + '\\' + name + '.obj', check_existing=True,
                             filter_glob="*.obj;*.mtl", use_selection=True, use_animation=False,
                             use_edges=True, use_normals=False, use_uvs=True,
                             use_materials=True, use_triangles=False, use_nurbs=False, use_vertex_groups=False,
                             # use_blen_objects=True, group_by_object=False, group_by_material=False,
                             use_blen_objects=False, group_by_object=False, group_by_material=False, # 不分组不设置o属性
                             keep_vertex_order=False, global_scale=1, axis_forward='-Z', axis_up='Y', path_mode='AUTO')
    delete_model(name)
    return True


def install_off_addon():
    filepath = os.path.dirname(__file__) + '\import_off.py'
    bpy.ops.wm.addon_install(
        overwrite=False,
        filepath=filepath
    )
    bpy.ops.wm.addon_enable(module='import_off')


def center_model(name):
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
    try:
        D.objects[name].location = (0, 0, 0)
    except:
        D.objects[0].location = (0, 0, 0)


def normalize_model(name):
    try:
        obj = D.objects[name]
    except:
        obj = D.objects[0]
    dim = obj.dimensions
    print('original dim:' + str(dim))
    if max(dim) > 0:
        dim = dim / max(dim)
    obj.dimensions = dim

    print('new dim:' + str(dim))


def delete_model(name):
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.name.startswith(name):
            ob.select = True
        else:
            ob.select = False
    bpy.ops.object.delete()


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

    convert(model, image_dir)


if __name__ == '__main__':
    main()