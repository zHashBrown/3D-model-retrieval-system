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


def convert(path, image_dir, target_format):
    name = load_model(path)
    center_model(name)
    normalize_model(name)

    if target_format == 'stl':
        bpy.ops.export_mesh.stl(filepath=image_dir + '\\' + name + '.stl', check_existing=True, axis_forward='Y', axis_up='Z', filter_glob="*.stl",
                                use_selection=False, global_scale=1.0, use_scene_unit=False, ascii=False,
                                use_mesh_modifiers=True, batch_mode='OFF')
    elif target_format == 'ply':
        bpy.ops.export_mesh.ply(filepath=image_dir + '\\' + name + '.ply', check_existing=True, axis_forward='Y', axis_up='Z', filter_glob="*.ply",
                                use_mesh_modifiers=True, use_normals=True, use_uv_coords=True, use_colors=True,
                                global_scale=1.0)
    elif target_format == '3ds':
        bpy.ops.export_scene.autodesk_3ds(filepath=image_dir + '\\' + name + '.3ds', check_existing=True, axis_forward='Y', axis_up='Z',
                                          filter_glob="*.3ds", use_selection=False)
    elif target_format == 'fbx':
        bpy.ops.export_scene.fbx(filepath=image_dir + '\\' + name + '.fbx', check_existing=True, axis_forward='-Z', axis_up='Y', filter_glob="*.fbx",
                                 version='BIN7400', ui_tab='MAIN', use_selection=False, global_scale=1.0,
                                 apply_unit_scale=True, bake_space_transform=False,
                                 object_types={'ARMATURE', 'CAMERA', 'EMPTY', 'LAMP', 'MESH', 'OTHER'},
                                 use_mesh_modifiers=True, mesh_smooth_type='OFF', use_mesh_edges=False,
                                 use_tspace=False, use_custom_props=False, add_leaf_bones=True, primary_bone_axis='Y',
                                 secondary_bone_axis='X', use_armature_deform_only=False, armature_nodetype='NULL',
                                 bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=True,
                                 bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True,
                                 bake_anim_step=1.0, bake_anim_simplify_factor=1.0, use_anim=True,
                                 use_anim_action_all=True, use_default_take=True, use_anim_optimize=True,
                                 anim_optimize_precision=6.0, path_mode='AUTO', embed_textures=False,
                                 batch_mode='OFF', use_batch_own_dir=True, use_metadata=True)
    elif target_format == 'x3d':
        bpy.ops.export_scene.x3d(filepath=image_dir + '\\' + name + '.x3d', check_existing=True, axis_forward='Z', axis_up='Y', filter_glob="*.x3d",
                                 use_selection=False, use_mesh_modifiers=True, use_triangulate=False, use_normals=False,
                                 use_compress=False, use_hierarchy=True, name_decorations=True, use_h3d=False,
                                 global_scale=1.0, path_mode='AUTO')
    ##########

    elif target_format == 'off':
        bpy.ops.export_mesh.off(filepath=image_dir + '\\' + name + '.off', filter_glob='*.off')
    elif target_format == 'obj':
        bpy.ops.export_scene.obj(filepath=image_dir + '\\' + name + '.obj', check_existing=True,
                                 filter_glob="*.obj;*.mtl", use_selection=True, use_animation=False,
                                 use_edges=True, use_normals=False, use_uvs=True,
                                 use_materials=True, use_triangles=False, use_nurbs=False, use_vertex_groups=False,
                                 # use_blen_objects=True, group_by_object=False, group_by_material=False,
                                 use_blen_objects=False, group_by_object=False, group_by_material=False,  # 不分组不设置o属性
                                 keep_vertex_order=False, global_scale=1, axis_forward='-Z', axis_up='Y',
                                 path_mode='AUTO')
    else:
        print('Currently .{} file type is not supported.'.format(ext))
        exit(-1)

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

    if len(argv) != 3:
        print('phong.py args: <3d mesh path> <image dir>')
        exit(-1)

    model = argv[0]
    image_dir = argv[1]
    target_format = argv[2]

    # blender has no native support for off files
    install_off_addon()

    convert(model, image_dir, target_format)


if __name__ == '__main__':
    main()
