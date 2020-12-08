import bpy
import bmesh
from mathutils import *
import math
import bpy_extras
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view
import random
import pickle
import os
from pathlib import Path
import json

project_root = f'{Path.home()}/work/roboaugen'

materials_model = [
    {
        'name': 'plastic-gray',
        'metallic': 0.1,   
        'diffuse_color': (0.9, 0.9, 0.9, 1.0),
        'roughness': 0.1
        
    },
    {
        'name': 'plastic-red',
        'metallic': 0.1,   
        'diffuse_color': (0.9, 0.2, 0.2, 1.0),
        'roughness': 0.3
    }
    ,
    {
        'name': 'metallic-green',
        'metallic': 0.9,   
        'diffuse_color': (0.2, 0.9, 0.2, 1.0),
        'roughness': 0.6
    },
    {
        'name': 'metallic-blue',
        'metallic': 0.9,   
        'diffuse_color': (0.2, 0.2, 0.9, 1.0),
        'roughness': 0.9
    },
    {
        'name': 'wood',
        'metallic': 0.1,   
        'texture': 'Wood-2093.jpg',
        'roughness': 0.9
    },
    {
        'name': 'concrete',
        'metallic': 0.1,   
        'texture': 'Concrete-0561.jpg',
        'roughness': 0.9
    },
    {
        'name': 'cracks',
        'metallic': 0.1,   
        'texture': 'Cracks_008.jpg',
        'roughness': 0.9
    },
    {
        'name': 'ground',
        'metallic': 0.1,   
        'texture': 'Ground-0179.jpg',
        'roughness': 0.9
    },
    {
        'name': 'metal',
        'metallic': 0.9,   
        'texture': 'Metal-4126.jpg',
        'roughness': 0.1
    },
    {
        'name': 'tarp',
        'metallic': 0.5,   
        'texture': 'Tarp-0486.jpg',
        'roughness': 0.5
    },
    {
        'name': 'dirt',
        'metallic': 0.05,   
        'texture': 'thumb-large-dirt_7805.jpg',
        'roughness': 0.95
    }
]


def render(image_path):
    if image_path is not None:
        bpy.data.scenes['Scene'].render.filepath = image_path
    bpy.context.scene.render.engine= 'CYCLES'
    bpy.context.scene.render.preview_pixel_size = '1'
    bpy.context.scene.render.bake_samples = 1024 * 100
    bpy.ops.render.render(write_still=True)

def get_projected_points_from_vertices(vertices):
    camera = bpy.data.objects['Camera']
    scene = bpy.context.scene
    vertices_projected = [ bpy_extras.object_utils.world_to_camera_view(scene, camera, v) if v is not None else None for v in vertices]
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        scene.render.resolution_x * render_scale,
        scene.render.resolution_y * render_scale,
    )
    coords = [ (  (co_2d.x * render_size[0]), render_size[1] - (co_2d.y * render_size[1]) ) if co_2d is not None else None for co_2d in vertices_projected ]
    return coords

def get_projected_bounding_box(all_vertices=False):
    vertices = []
    if all_vertices:
        vertices = bpy.data.objects['Model'].data.vertices
        vertices = [v.co for v in vertices]
    else:
        bound_box = bpy.data.objects['Model'].bound_box
        vertices = [Vector((v[0], v[1], v[2])) for v in bound_box]
    
    print(vertices)
    mat = bpy.data.objects['Model'].matrix_world
    vertices = [mat @ v for v in vertices]
    print(vertices)
    projected_vertices = get_projected_points_from_vertices(vertices)

    return projected_vertices, vertices

def get_object_height():
    bound_box = bpy.data.objects['Model'].bound_box
    vertices = [Vector((v[0], v[1], v[2])) for v in bound_box]
    mat = bpy.data.objects['Model'].matrix_world
    vertices = [mat @ v for v in vertices]
    min_z, max_z = None, None
    for v in vertices:
        if min_z is None or min_z > v[2]:
            min_z = v[2]
        if max_z is None or max_z < v[2]:
            max_z = v[2]
    return max_z - min_z

def move_camera(distance=30, height=0):
    camera = bpy.data.objects['Camera']
    rot_quat = Vector((1.0, 0., 0.)).to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    object_height = get_object_height()
    camera.location = Vector((distance, 0., height + (object_height / 2.0)))

def rotate_camera(degrees=30):
    camera = bpy.data.objects['Camera']
    camera.rotation_euler.rotate_axis("Y", math.radians(degrees))

def rotate_object(degrees):
    object =  bpy.data.objects['Model']
    object.rotation_euler = (math.pi / 2., 0.0, math.radians(degrees))

def move_object(x, y, z):
    object =  bpy.data.objects['Model']
    object.location = Vector((x, y, z))

def scale_object(scale):
    object = bpy.data.objects['Model']
    object.scale = [scale, scale, scale]

def move_light(x, y, z):
    bpy.data.objects['Lamp'].location = Vector((x, y, z))

def set_light_intensity(energy=100):
    bpy.data.lights['Lamp'].energy = energy

def change_camera_focal_length(lens=35):
     bpy.data.cameras['Camera'].clip_end = 10000000
     bpy.data.cameras['Camera'].lens = lens

def recreate_light_source():
    lights = bpy.data.lights
    if 'Lamp' in lights:
        lights.remove(lights['Lamp'], do_unlink=True)        
    objects = bpy.data.objects
    if 'Lamp' in objects:
        objects.remove(objects['Lamp'], do_unlink=True)
    light_data = lights.new(name="Lamp", type='POINT')
    light_data.energy = 100
    light_object = objects.new(name="Lamp", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    light_object.location = (0, 0, 20)
    dg = bpy.context.evaluated_depsgraph_get() 
    dg.update()

def add_object(scale):    
    objs = bpy.data.objects
    if 'Model' in objs:
        objs.remove(objs['Model'], do_unlink=True)
    bpy.ops.import_mesh.stl(filepath = f"{project_root}/test/data/model.stl",
        filter_glob="*.stl", files=[{"name":"model.stl", "name":"model.stl"}],
        directory=f"{project_root}/test/data")
    scale_object(scale)
    bpy.ops.transform.rotate(value = -1.5708, orient_axis= 'X')
    change_camera_focal_length(35)
    move_object(0., 0., 0.)
    rotate_object(degrees=0.0)

def BVHTreeAndVertices( bbox_vertices ):
    polygons = [
        (0, 1, 5),
        (0, 4, 5),
        (1, 2, 5),
        (2, 5, 6),
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (3, 4, 7),
        (2, 3, 6),
        (3, 6, 7),
        (5, 6, 7),
        (4, 5, 7)]
    bvh = BVHTree.FromPolygons( bbox_vertices, polygons )
    return bvh

def get_visible_bounding_box(bbox_vertices):
    scene = bpy.context.scene
    cam = bpy.data.objects['Camera']
    obj = bpy.data.objects['Model']
    bvh = BVHTreeAndVertices( bbox_vertices )
    visible_vertices = []
    for i, v in enumerate( bbox_vertices ):
        co2D = world_to_camera_view( scene, cam, v )
        if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0:
            location, normal, index, distance = bvh.ray_cast( cam.location, (v - cam.location).normalized() )
            error = (v - location).length if location else None
            if error and error < 0.01:
                #print("error above bounds", error)
                visible_vertices.append(v)
            elif not error:
                #print("adding vertex withour location")
                visible_vertices.append(v)
            else:
                #print("error below bounds", error)
                visible_vertices.append(None)
        else:
            visible_vertices.append(None)
    del bvh
    return visible_vertices

def set_background(id):
    background = f'//../data/backgrounds/{id}.jpg'
    bpy.data.images['background'].filepath = background

def set_material(id):
    material = materials_model[id]['name']
    obj = bpy.data.objects['Model']
    mat = bpy.data.materials[material]
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    obj.material_slots[0].material = mat
    
    # Smart UV Project
    if mat.use_nodes: # check if material has texture
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()
        bpy.ops.object.editmode_toggle()

def cast_shadow(cast):
    bpy.data.objects['Plane'].hide_render = not cast

def set_background_type(use_image):
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links
    [links.remove(link) for link in links]
    if use_image:
        links.new(nodes['Image'].outputs[0], nodes['Scale'].inputs[0])
        links.new(nodes['Scale'].outputs[0], nodes['Alpha Over'].inputs[1])
        links.new(nodes['Render Layers'].outputs[0], nodes['Alpha Over'].inputs[2])
        links.new(nodes['Alpha Over'].outputs[0], nodes['Composite'].inputs[0])
    else:
        links.new(nodes['Render Layers'].outputs[0], nodes['Composite'].inputs[0])    

def to_map(light_position, cast_shadow_id, background_id, material_id, \
    focal, object_position, object_rotation, camera_height, camera_distance, \
    light_intensity, camera_rotation, use_background, object_scale):        
    return {
            'light_position': light_position,
            'cast_shadow_id': cast_shadow_id,
            'background_id': background_id,
            'material_id': material_id,
            'focal': focal,
            'object_position': object_position,
            'object_rotation': object_rotation,
            'camera_height': camera_height,
            'camera_distance': camera_distance,
            'light_intensity': light_intensity,
            'camera_rotation': camera_rotation,
            'use_background': use_background,
            'resolution': [scene.render.resolution_x, scene.render.resolution_y],
            'object_scale': object_scale
        }

def get_background_ids():
    backgrounds_folder = f'{project_root}/data/backgrounds'
    ids = os.listdir(backgrounds_folder)
    return ids

def randomize_scene(sample):
    object_rotation = random.randint(0,45)
    object_x = random.randint(-40, 20)
    object_y = random.randint(-20, 20)
    object_z = random.randint(0, 0)
    object_position = [object_x, object_y, object_z]
    camera_distance = 60
    camera_height = random.randint(object_z, 30)
    camera_rotation = 0
    light_x = object_x + random.randint(-30,30)
    light_y = object_y + random.randint(-30,30)
    light_z = random.randint(50,80)
    light_position = [light_x, light_y, light_z]
    light_intensity = random.randint(100 * light_z, 5000 * light_z)
    focal = random.randint(25,40)
    background_id = random.randint(0, len(get_background_ids()) - 1)
    material_id = random.randint(0, len(materials_model) - 1)
    cast_shadow_id = random.randint(0, 1)
    use_background = True
    object_scale = 0.1 + (random.random() * 0.2)
    return to_map(light_position, cast_shadow_id, background_id, material_id, \
        focal, object_position, object_rotation, camera_height, camera_distance, \
        light_intensity, camera_rotation, use_background, object_scale)
        


def apply_scene_data(scene_data):
    add_object(scene_data['object_scale'])
    set_background_type(scene_data['use_background'])
    set_background(scene_data['background_id'])
    cast_shadow(scene_data['cast_shadow_id'] == 1)
    set_material(scene_data['material_id'])
    set_light_intensity(energy=scene_data['light_intensity'])
    change_camera_focal_length(scene_data['focal'])
    move_light(
        x=scene_data['light_position'][0],
        y=scene_data['light_position'][1],
        z=scene_data['light_position'][2])
    rotate_camera(degrees=scene_data['camera_rotation'])
    move_camera(distance=scene_data['camera_distance'],
        height=scene_data['camera_height'])
    move_object(\
        scene_data['object_position'][0], 
        scene_data['object_position'][1],
        scene_data['object_position'][2])
    rotate_object(degrees=scene_data['object_rotation'])

def get_projected_visible_vertices():
    _, bbox_vertices = get_projected_bounding_box(False)
    visible_vertices = get_visible_bounding_box(bbox_vertices)
    projected_visible_vertices = get_projected_points_from_vertices(visible_vertices)
    bbox_vertices = get_projected_points_from_vertices(bbox_vertices)
    return bbox_vertices, projected_visible_vertices

def get_next_sample_id(material_name):
    folder = f"{project_root}/data/train/{material_name}"
    if os.path.exists(folder):
        ids = os.listdir(folder)
        if len(ids) > 0:
            ids = [int(id) for id in ids]
            ids.sort(reverse = True)
            return ids[0] + 1
        else:
            return 0
    else:
        return 0

def write_json(folder_path, filename, data):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    data_path = f"{folder_path}/{filename}.json"
    with open(data_path, 'w') as outfile:
        data_string = json.dumps(data, indent=4, sort_keys=True)
        outfile.write(data_string)
    apply_scene_data(data)
    render(f"{folder_path}/{filename}.png")
    data['use_background'] = False
    data['cast_shadow_id'] = 0
    apply_scene_data(data)
    render(f"{folder_path}/{filename}_no_background.png")


def save_data(sample_dir, data):
    write_json(sample_dir, 'data', data)

def generate_random_samples(samples=100):
    sample_dirs = []
    for sample in range(samples):        
        data = randomize_scene(sample)        
        apply_scene_data(data)
        render(f"/tmp/blender_generator.png")
        bbox_vertices, projected_visible_vertices = get_projected_visible_vertices()
        if len([v for v  in projected_visible_vertices if v is not None]) > 0:
            data['projected_visible_vertices'] = projected_visible_vertices
            data['bbox_vertices'] = bbox_vertices
            material_name = materials_model[data['material_id']]['name']
            sample_id = get_next_sample_id(material_name)
            folder_name = f'{project_root}/data/train/{material_name}'
            if not os.path.exists(folder_name):            
                os.makedirs(folder_name)
            sample_dir = f'{folder_name}/{sample_id}'
            save_data(sample_dir, data)
            sample_dirs.append(sample_dir)
            sample_id += 1
    return sample_dirs

def display_sample(sample_id):
    data_sample_path = f'{project_root}/data/train/{sample_id}/data.json'
    if(os.path.exists(data_sample_path)):
        with open(data_sample_path, 'r') as f:
            data = f.read()
            data = json.loads(data)
            apply_scene_data(data)
            render(None)

def create_materials():    
    materials = bpy.data.materials
    for material in materials:
        bpy.data.materials.remove(material)        
    for material_model in materials_model:
        material = materials.new(name=material_model['name'])
        if 'metallic' in material_model:
            material.metallic = material_model['metallic']
        if 'roughness' in material_model:
            material.roughness = material_model['roughness']
        if 'specular_intensity' in material_model:
            material.specular_intensity = material_model['specular_intensity']
        if 'diffuse_color' in material_model:
            material.diffuse_color = material_model['diffuse_color']
        if 'specular_color' in material_model:
            material.specular_color = material_model['specular_color']
        if 'texture' in material_model:
            material.use_nodes = True
            bsdf = material.node_tree.nodes["Principled BSDF"]
            texture_image = material.node_tree.nodes.new('ShaderNodeTexImage')
            texture_image.image = \
                bpy.data.images.load( \
                    f"{project_root}/data/textures/{material_model['texture']}")
            material.node_tree.links.new(bsdf.inputs['Base Color'], texture_image.outputs['Color'])
            
        
if __name__ == "__main__":
    recreate_light_source()
    create_materials()
    scene = bpy.context.scene
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    sample_dirs = generate_random_samples(100)
    print(sample_dirs)
    
    #display_sample(14)
    
    
    